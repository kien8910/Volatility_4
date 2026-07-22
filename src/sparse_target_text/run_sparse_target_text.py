from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import torch
import yaml

from .config import load_config
from .data import (attach_embeddings_and_novelty, build_event_embedding_cache, load_g5_oof_baseline,
                   load_target_event_candidates, shuffle_event_payload_within_day)
from .evaluator import add_losses, gate_diagnostics, metric_table, select_variant
from .trainer import fit_variant, predict_checkpoint, resolve_device, save_checkpoint, seed_everything


def _config_fingerprint(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _baseline_frame(baseline: pd.DataFrame, split_name: str | None = None) -> pd.DataFrame:
    out = baseline.copy()
    if split_name is not None: out = out.loc[out.analysis_split.eq(split_name)].copy()
    out["model"] = "M0_stock_only"; out["correction"] = 0.0
    out["final_residual_prediction"] = out.stock_residual_prediction
    out["residual_prediction"] = out.final_residual_prediction
    out["final_prediction"] = out.p_prediction + out.final_residual_prediction
    out["hurdle_probability"] = 0.0; out["has_selected_event"] = 0
    return out


def _load(cfg: dict, include_test: bool, require_embeddings: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = load_g5_oof_baseline(cfg, include_locked_test=include_test)
    events = load_target_event_candidates(cfg, baseline)
    if require_embeddings: events = attach_embeddings_and_novelty(events, cfg)
    return baseline, events


def _write_validation_outputs(cfg: dict, predictions: list[pd.DataFrame], edges: list[pd.DataFrame], diagnostics: list[dict]) -> None:
    outdir = Path(cfg["output"]["directory"]); outdir.mkdir(parents=True, exist_ok=True)
    all_predictions = add_losses(pd.concat(predictions, ignore_index=True))
    all_predictions.to_parquet(outdir / "predictions_development.parquet", index=False)
    validation = all_predictions.loc[all_predictions.analysis_split.eq("validation")]
    by_model = metric_table(validation, ["model"]); by_horizon = metric_table(validation, ["model", "horizon"])
    by_ticker = metric_table(validation, ["model", "ticker"])
    by_model.to_csv(outdir / "metrics_validation_by_model.csv", index=False)
    by_horizon.to_csv(outdir / "metrics_validation_by_horizon.csv", index=False)
    by_ticker.to_csv(outdir / "metrics_validation_by_ticker.csv", index=False)
    edge_frame = pd.concat(edges, ignore_index=True) if edges else pd.DataFrame()
    if len(edge_frame): edge_frame.to_parquet(outdir / "event_selection_values.parquet", index=False)
    gates = gate_diagnostics(edge_frame); gates.to_csv(outdir / "gate_diagnostics.csv", index=False)
    pd.DataFrame([{k: v for k, v in row.items() if k != "history"} for row in diagnostics]).to_csv(
        outdir / "training_diagnostics.csv", index=False)


def run(cfg: dict, mode: str, device_name: str, include_locked_test: bool = False,
        checkpoint_path: str | None = None) -> None:
    outdir = Path(cfg["output"]["directory"]); checkpoints = Path(cfg["output"]["checkpoint_directory"])
    outdir.mkdir(parents=True, exist_ok=True); checkpoints.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name); seed_everything(int(cfg["training"]["seed"]))
    if mode == "validate-data":
        baseline, events = _load(cfg, include_test=False, require_embeddings=False)
        summary = {"baseline_rows": len(baseline), "event_candidates": len(events),
                   "basic_filter_events": int(events.basic_filter_pass.sum()), "hard_filter_events": int(events.hard_filter_pass.sum()),
                   "train_rows": int(baseline.analysis_split.eq("train").sum()),
                   "validation_rows": int(baseline.analysis_split.eq("validation").sum()),
                   "news_lag_sessions": int(cfg["information_cutoff"]["news_lag_sessions"]),
                   "locked_test_loaded": False}
        (outdir / "data_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8"); print(summary); return
    if mode in {"build-embedding-cache", "build-locked-test-cache"}:
        include_test = mode == "build-locked-test-cache"
        if include_test:
            selected_file = outdir / "selected_variant.yaml"
            if not selected_file.exists():
                raise FileNotFoundError("Freeze validation selection before building locked-test embeddings")
            selected = yaml.safe_load(selected_file.read_text(encoding="utf-8"))
            if selected.get("config_fingerprint") != _config_fingerprint(cfg):
                raise ValueError("Current config differs from the frozen validation-selected config")
        baseline, events = _load(cfg, include_test=include_test, require_embeddings=False)
        cache = build_event_embedding_cache(events, cfg, str(device))
        print({"cached_rows": len(cache), "locked_test_included": include_test}); return
    if mode == "train-validation":
        baseline, events = _load(cfg, include_test=False, require_embeddings=True)
        predictions, edge_frames, diagnostics = [_baseline_frame(baseline)], [], []
        for variant in ["T1_all_target", "T2_hard_top1", "T3_sparse_hurdle"]:
            pred, edge, diag, checkpoint = fit_variant(variant, baseline, events, cfg, device)
            predictions.append(pred); edge_frames.append(edge); diagnostics.append(diag)
            save_checkpoint(checkpoint, checkpoints / f"{variant}.pt")
        placebo_path = outdir / "predictions_placebo.parquet"
        if placebo_path.exists():
            manifest = json.loads((outdir / "placebo_manifest.json").read_text(encoding="utf-8"))
            if manifest.get("config_fingerprint") != _config_fingerprint(cfg):
                raise ValueError("Existing placebo artifacts were built with a different config")
            predictions.append(pd.read_parquet(placebo_path))
            edge_path = outdir / "event_selection_placebo.parquet"
            if edge_path.exists(): edge_frames.append(pd.read_parquet(edge_path))
        _write_validation_outputs(cfg, predictions, edge_frames, diagnostics); return
    if mode == "run-placebo":
        baseline, events = _load(cfg, include_test=False, require_embeddings=True)
        shuffled, shuffle_diag = shuffle_event_payload_within_day(events, int(cfg["placebo"]["seed"]))
        predictions, edge_frames, diagnostics = [], [], []
        for variant in ["T1_all_target", "T2_hard_top1", "T3_sparse_hurdle"]:
            pred, edge, diag, checkpoint = fit_variant(variant, baseline, shuffled, cfg, device)
            placebo_name = f"{variant}_placebo"; pred["model"] = placebo_name; edge["model"] = placebo_name
            diag.update(shuffle_diag); diag["model"] = placebo_name
            predictions.append(pred); edge_frames.append(edge); diagnostics.append({k: v for k, v in diag.items() if k != "history"})
            save_checkpoint(checkpoint, checkpoints / f"{placebo_name}.pt")
        add_losses(pd.concat(predictions, ignore_index=True)).to_parquet(outdir / "predictions_placebo.parquet", index=False)
        pd.concat(edge_frames, ignore_index=True).to_parquet(outdir / "event_selection_placebo.parquet", index=False)
        (outdir / "placebo_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        (outdir / "placebo_manifest.json").write_text(
            json.dumps({"config_fingerprint": _config_fingerprint(cfg)}, indent=2), encoding="utf-8")
        return
    if mode == "select-config":
        metrics = pd.read_csv(outdir / "metrics_validation_by_model.csv")
        ticker = pd.read_csv(outdir / "metrics_validation_by_ticker.csv")
        gates = pd.read_csv(outdir / "gate_diagnostics.csv")
        selection = select_variant(metrics, ticker, gates, cfg)
        selection["config_fingerprint"] = _config_fingerprint(cfg)
        if selection["selected_variant"] != "M0_stock_only":
            frozen_checkpoint = checkpoints / f"{selection['selected_variant']}.pt"
            selection["checkpoint_path"] = str(frozen_checkpoint)
            selection["checkpoint_sha256"] = _file_sha256(frozen_checkpoint)
        with (outdir / "selected_variant.yaml").open("w", encoding="utf-8") as fh: yaml.safe_dump(selection, fh, sort_keys=False)
        (outdir / "run_summary.json").write_text(json.dumps(selection, indent=2), encoding="utf-8"); print(selection); return
    if mode == "evaluate-locked-test":
        selected_file = outdir / "selected_variant.yaml"
        selected = yaml.safe_load(selected_file.read_text(encoding="utf-8"))
        if selected.get("config_fingerprint") != _config_fingerprint(cfg):
            raise ValueError("Current config differs from the frozen validation-selected config")
        if selected["selected_variant"] == "M0_stock_only":
            baseline = load_g5_oof_baseline(cfg, include_locked_test=True)
            predictions = _baseline_frame(baseline, "test")
        else:
            baseline, events = _load(cfg, include_test=True, require_embeddings=True)
            test = baseline.loc[baseline.analysis_split.eq("test")].copy()
            path = checkpoint_path or selected.get("checkpoint_path") or str(checkpoints / f"{selected['selected_variant']}.pt")
            if _file_sha256(path) != selected.get("checkpoint_sha256"):
                raise ValueError("Checkpoint hash differs from the validation-selected checkpoint")
            predictions, edges = predict_checkpoint(path, test, events, device)
            edges.to_parquet(outdir / "event_selection_test.parquet", index=False)
        predictions = add_losses(predictions); predictions.to_parquet(outdir / "predictions_locked_test.parquet", index=False)
        metric_table(predictions, ["model"]).to_csv(outdir / "metrics_locked_test.csv", index=False); return
    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=["validate-data", "build-embedding-cache", "build-locked-test-cache",
                                                          "train-validation", "run-placebo", "select-config", "evaluate-locked-test"])
    parser.add_argument("--device", default="auto"); parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args(); run(load_config(args.config), args.mode, args.device, checkpoint_path=args.checkpoint)


if __name__ == "__main__": main()
