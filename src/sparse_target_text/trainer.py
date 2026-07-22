from __future__ import annotations

import copy
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .filtering import deterministic_top_k
from .losses import qlike_logvol, sparse_hurdle_loss
from .models import SparseHurdleCorrector

STATE_COLUMNS = ["baseline_prediction", "stock_residual_prediction", "p_prediction", "market_mean_prediction",
                 "market_prediction_dispersion", "baseline_vs_market", "known_error_mean_5",
                 "known_abs_error_mean_5", "known_error_mean_22", "known_abs_error_mean_22"]
_PREDICTION_ROW_ID = "prediction_row_id"


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(value: str) -> torch.device:
    if value == "auto": value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    return torch.device(value)


def variant_events(events: pd.DataFrame, variant: str, top_k: int) -> tuple[pd.DataFrame, str]:
    if variant == "T1_all_target":
        return events.loc[events.basic_filter_pass].copy(), "all"
    if variant == "T2_hard_top1":
        return deterministic_top_k(events, top_k), "deterministic_topk"
    if variant == "T3_sparse_hurdle":
        return events.loc[events.hard_filter_pass].copy(), "learned_topk"
    raise ValueError(f"Unknown variant {variant}")


def _state_stats(baseline: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train = baseline.loc[baseline.analysis_split.eq("train"), STATE_COLUMNS].to_numpy(np.float32)
    mean = np.nanmean(train, axis=0); scale = np.nanstd(train, axis=0); scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _tensor_bundle(baseline: pd.DataFrame, events: pd.DataFrame, cfg: dict, device: torch.device,
                   state_mean: np.ndarray | None = None, state_scale: np.ndarray | None = None,
                   event_type_map: dict[str, int] | None = None,
                   impact_thresholds: dict[int, float] | None = None) -> dict:
    rows = baseline.reset_index(drop=True).copy(); rows[_PREDICTION_ROW_ID] = np.arange(len(rows), dtype=np.int64)
    if _PREDICTION_ROW_ID in events:
        raise ValueError(f"Event frame contains reserved column {_PREDICTION_ROW_ID}")
    row_lookup = rows[["date", "ticker", "horizon", "fold_id", "analysis_split", _PREDICTION_ROW_ID]].rename(
        columns={"date": "effective_date"}
    )
    edges = events.merge(row_lookup, on=["effective_date", "ticker"], how="inner", validate="many_to_many")
    if edges.empty: raise ValueError("No target-company events align to baseline rows")
    edge_row = edges[_PREDICTION_ROW_ID].to_numpy(dtype=np.int64)
    if edge_row.min() < 0 or edge_row.max() >= len(rows):
        raise ValueError("Event-to-prediction row mapping is out of bounds")
    if state_mean is None or state_scale is None: state_mean, state_scale = _state_stats(rows)
    state = (rows[STATE_COLUMNS].to_numpy(np.float32) - state_mean) / state_scale
    ticker_map = {str(t): i for i, t in enumerate(cfg["data"]["tickers"])}
    if event_type_map is None:
        event_types = ["other"] + sorted(set(events.event_type.astype(str)) - {"other"})
        event_type_map = {x: i for i, x in enumerate(event_types)}
    event_meta = np.column_stack([
        edges.semantic_novelty.astype(float).clip(0, 2) / 2.0, edges.catalyst_score.astype(float),
        edges.entity_relevance.astype(float), edges.timestamp_confidence.astype(float),
        np.log1p(edges.word_count.astype(float)) / np.log(513.0),
    ]).astype(np.float32)
    impact_q = float(cfg["training"].get("impact_label_quantile", 0.80))
    thresholds = impact_thresholds or rows.loc[rows.analysis_split.eq("train")].groupby("horizon").baseline_error.apply(lambda x: x.abs().quantile(impact_q)).to_dict()
    labels = np.asarray([abs(e) >= thresholds.get(int(h), float("inf")) for e, h in zip(rows.baseline_error, rows.horizon)], dtype=np.float32)
    bundle = {"rows": rows, "edges": edges, "state_mean": state_mean, "state_scale": state_scale,
              "ticker_map": ticker_map, "event_type_map": event_type_map, "impact_thresholds": thresholds,
              "embedding": torch.as_tensor(np.stack(edges.embedding.map(lambda x: np.asarray(x, dtype=np.float32))), device=device),
              "event_meta": torch.as_tensor(event_meta, device=device),
              "event_type": torch.as_tensor(edges.event_type.astype(str).map(event_type_map).fillna(0).to_numpy(), dtype=torch.long, device=device),
              "edge_row": torch.as_tensor(edge_row, dtype=torch.long, device=device),
              "edge_train_mask": torch.as_tensor(edges.analysis_split.eq("train").to_numpy(), device=device),
              "row_state": torch.as_tensor(state, device=device),
              "row_stock": torch.as_tensor(rows.ticker.astype(str).map(ticker_map).to_numpy(), dtype=torch.long, device=device),
              "row_horizon": torch.as_tensor(rows.horizon.to_numpy(), dtype=torch.long, device=device),
              "actual": torch.as_tensor(rows.actual_logvol.to_numpy(), dtype=torch.float32, device=device),
              "baseline_prediction": torch.as_tensor(rows.baseline_prediction.to_numpy(), dtype=torch.float32, device=device),
              "impact_label": torch.as_tensor(labels, dtype=torch.float32, device=device)}
    return bundle


def _forward(net: SparseHurdleCorrector, bundle: dict, selection_mode: str, top_k: int) -> dict:
    return net(bundle["embedding"], bundle["event_meta"], bundle["event_type"], bundle["edge_row"],
               bundle["row_state"], bundle["row_stock"], bundle["row_horizon"], selection_mode, top_k)


def fit_variant(variant: str, baseline: pd.DataFrame, all_events: pd.DataFrame, cfg: dict,
                device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    events, selection_mode = variant_events(all_events, variant, int(cfg["model"]["top_k_events"]))
    bundle = _tensor_bundle(baseline, events, cfg, device)
    net = SparseHurdleCorrector(bundle["embedding"].shape[1], len(STATE_COLUMNS), len(cfg["data"]["tickers"]),
                                tuple(cfg["data"]["horizons"]), cfg).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=float(cfg["training"]["learning_rate"]),
                                  weight_decay=float(cfg["training"]["weight_decay"]))
    train_mask = torch.as_tensor(bundle["rows"].analysis_split.eq("train").to_numpy(), device=device)
    val_mask = torch.as_tensor(bundle["rows"].analysis_split.eq("validation").to_numpy(), device=device)
    best_state = copy.deepcopy(net.state_dict()); best_qlike = float("inf"); stale = 0; history = []
    for epoch in range(int(cfg["training"]["max_epochs"])):
        net.train(); optimizer.zero_grad(); output = _forward(net, bundle, selection_mode, int(cfg["model"]["top_k_events"]))
        final = bundle["baseline_prediction"] + output["correction"]
        train_event_mask = train_mask & output["has_event"]
        if not bool(train_event_mask.any()):
            raise ValueError(f"{variant} has no selected training events after filtering/alignment")
        loss, parts = sparse_hurdle_loss(bundle["actual"][train_event_mask], final[train_event_mask], output["correction"][train_event_mask],
                                         output["hurdle_probability"][train_event_mask], bundle["impact_label"][train_event_mask],
                                         output["edge_gate"][bundle["edge_train_mask"]], cfg)
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), float(cfg["training"]["gradient_clip_norm"])); optimizer.step()
        net.eval()
        with torch.no_grad():
            val_output = _forward(net, bundle, selection_mode, int(cfg["model"]["top_k_events"]))
            val_qlike = float(qlike_logvol(bundle["actual"][val_mask],
                              (bundle["baseline_prediction"] + val_output["correction"])[val_mask]).cpu())
        history.append({"epoch": epoch, "validation_qlike": val_qlike,
                        **{k: float(v.detach().cpu()) for k, v in parts.items()}})
        if val_qlike < best_qlike - float(cfg["training"].get("minimum_improvement", 1e-7)):
            best_qlike = val_qlike; best_state = copy.deepcopy(net.state_dict()); stale = 0
        else:
            stale += 1
        if stale >= int(cfg["training"]["early_stopping_patience"]): break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad(): output = _forward(net, bundle, selection_mode, int(cfg["model"]["top_k_events"]))
    predictions, edge_output = outputs_to_frames(variant, bundle, output)
    checkpoint = {"variant": variant, "selection_mode": selection_mode, "model_state": best_state,
                  "state_mean": bundle["state_mean"], "state_scale": bundle["state_scale"],
                  "ticker_map": bundle["ticker_map"], "event_type_map": bundle["event_type_map"],
                  "impact_thresholds": bundle["impact_thresholds"], "embedding_dim": int(bundle["embedding"].shape[1]),
                  "config": cfg}
    diagnostics = {"variant": variant, "best_validation_qlike": best_qlike, "epochs": len(history), "history": history,
                   "train_events": int(len(events.loc[events.effective_date.isin(bundle["rows"].loc[train_mask.cpu().numpy(), "date"])])),
                   "validation_events": int(len(events.loc[events.effective_date.isin(bundle["rows"].loc[val_mask.cpu().numpy(), "date"])]))}
    return predictions, edge_output, diagnostics, checkpoint


def outputs_to_frames(variant: str, bundle: dict, output: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = bundle["rows"].drop(columns=_PREDICTION_ROW_ID).copy()
    predictions["model"] = variant; predictions["correction"] = output["correction"].detach().cpu().numpy()
    predictions["final_residual_prediction"] = predictions.stock_residual_prediction + predictions.correction
    predictions["residual_prediction"] = predictions.final_residual_prediction
    predictions["final_prediction"] = predictions.p_prediction + predictions.final_residual_prediction
    if not np.allclose(predictions.final_prediction, predictions.baseline_prediction + predictions.correction, atol=1e-6):
        raise AssertionError("Late-fusion prediction identity failed")
    predictions["hurdle_probability"] = output["hurdle_probability"].detach().cpu().numpy()
    predictions["has_selected_event"] = output["has_event"].detach().cpu().numpy().astype(int)
    predictions["high_baseline_error_label_using_train_threshold"] = bundle["impact_label"].detach().cpu().numpy().astype(int)
    edge_columns = ["event_id", "original_event_id", "news_date", "effective_date", "ticker", "category", "event_type",
                    "text_hash", "original_text_hash", "chunk_index", "chunk_count", "token_count", "horizon", "fold_id",
                    "analysis_split", "row_id", _PREDICTION_ROW_ID, "semantic_novelty", "catalyst_score", "entity_relevance",
                    "timestamp_confidence", "word_count", "payload_source_ticker", "wrong_ticker_payload"]
    edge = bundle["edges"][[c for c in edge_columns if c in bundle["edges"]]].copy()
    edge["model"] = variant; edge["edge_gate"] = output["edge_gate"].detach().cpu().numpy()
    edge["selected"] = output["selected_mask"].detach().cpu().numpy().astype(int)
    return predictions, edge


def save_checkpoint(checkpoint: dict, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); torch.save(checkpoint, path)


def predict_checkpoint(checkpoint_path: str | Path, baseline: pd.DataFrame, all_events: pd.DataFrame,
                       device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]; variant = checkpoint["variant"]
    events, selection_mode = variant_events(all_events, variant, int(cfg["model"]["top_k_events"]))
    bundle = _tensor_bundle(baseline, events, cfg, device,
                            np.asarray(checkpoint["state_mean"], dtype=np.float32),
                            np.asarray(checkpoint["state_scale"], dtype=np.float32),
                            checkpoint["event_type_map"], checkpoint["impact_thresholds"])
    net = SparseHurdleCorrector(int(checkpoint["embedding_dim"]), len(STATE_COLUMNS), len(cfg["data"]["tickers"]),
                                tuple(cfg["data"]["horizons"]), cfg).to(device)
    net.load_state_dict(checkpoint["model_state"]); net.eval()
    with torch.no_grad(): output = _forward(net, bundle, selection_mode, int(cfg["model"]["top_k_events"]))
    return outputs_to_frames(variant, bundle, output)
