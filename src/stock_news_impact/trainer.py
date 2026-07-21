from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.stock_news_impact.checkpointing import save_step7_checkpoint
from src.stock_news_impact.evaluator import prediction_level_from_event_rows
from src.stock_news_impact.factorized_gate import FactorizedGateModel, FixedGateModel
from src.stock_news_impact.losses import gate_regularization, gated_forecast_loss


@dataclass(frozen=True)
class Step7RunConfig:
    model: str
    initial_probability: float = 0.10
    fixed_gate: float = 0.10
    relevance_hidden: tuple[int, ...] = (32, 16)
    usage_weight: float = 0.001
    correction_weight: float = 0.001
    utility_weight: float = 0.0
    common_utility_weight: float = 0.0

    @property
    def config_id(self) -> str:
        if self.model == "S2_FixedSmallGate":
            return f"{self.model}__g{self.fixed_gate:g}"
        if self.model == "S0_StockOnly_G5":
            return self.model
        hidden = "-".join(str(x) for x in self.relevance_hidden)
        base = f"{self.model}__p{self.initial_probability:g}__rh{hidden}__uw{self.usage_weight:g}__cw{self.correction_weight:g}"
        if self.model == "S5_UtilityFactorizedGate":
            base += f"__utw{self.utility_weight:g}__cutw{self.common_utility_weight:g}"
        return base

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "config_id": self.config_id,
            "initial_probability": self.initial_probability,
            "fixed_gate": self.fixed_gate,
            "relevance_hidden": list(self.relevance_hidden),
            "usage_weight": self.usage_weight,
            "correction_weight": self.correction_weight,
            "utility_weight": self.utility_weight,
            "common_utility_weight": self.common_utility_weight,
        }


class GateDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, columns: dict[str, list[str]]):
        self.frame = frame.reset_index(drop=True)
        self.columns = columns

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[idx]
        return {
            "event": torch.tensor(row[self.columns["event"]].to_numpy(dtype=np.float32)),
            "relation": torch.tensor(row[self.columns["relation"]].to_numpy(dtype=np.float32)),
            "stock": torch.tensor(row[self.columns["stock"]].to_numpy(dtype=np.float32)),
            "market": torch.tensor(row[self.columns["market"]].to_numpy(dtype=np.float32)),
            "utility_context": torch.tensor(row[self.columns.get("utility_context", [])].to_numpy(dtype=np.float32)),
            "horizon": torch.tensor(int(row["horizon"]), dtype=torch.long),
            "actual_logvol": torch.tensor(float(row["actual_logvol"]), dtype=torch.float32),
            "stock_prediction": torch.tensor(float(row["stock_prediction"]), dtype=torch.float32),
            "p_prediction": torch.tensor(float(row["p_prediction"]), dtype=torch.float32),
            "stock_residual_prediction": torch.tensor(float(row["stock_residual_prediction"]), dtype=torch.float32),
            "news_correction_proxy": torch.tensor(float(row["news_correction_proxy"]), dtype=torch.float32),
            "utility_label": torch.tensor(int(row.get("utility_label", -1)), dtype=torch.long),
            "event_common_utility_label": torch.tensor(int(row.get("event_common_utility_label", -1)), dtype=torch.long),
            "row_index": torch.tensor(idx, dtype=torch.long),
        }


def _market_input(batch: dict[str, torch.Tensor], run_cfg: Step7RunConfig, device: torch.device) -> torch.Tensor:
    market = batch["market"].to(device)
    if run_cfg.model != "S5_UtilityFactorizedGate":
        return market
    utility_context = batch["utility_context"].to(device)
    if utility_context.shape[-1] == 0:
        return market
    return torch.cat([market, utility_context], dim=-1)


def build_model(run_cfg: Step7RunConfig, columns: dict[str, list[str]], cfg: dict):
    if run_cfg.model == "S2_FixedSmallGate":
        return FixedGateModel(run_cfg.fixed_gate)
    return FactorizedGateModel(
        event_dim=len(columns["event"]),
        relation_dim=len(columns["relation"]),
        stock_dim=len(columns["stock"]),
        market_dim=len(columns["market"]) + (len(columns.get("utility_context", [])) if run_cfg.model == "S5_UtilityFactorizedGate" else 0),
        horizons=list(cfg["target"]["horizons"]),
        reliability_hidden_dim=int(cfg["gate"]["reliability_hidden_dim"]),
        relevance_hidden_dims=list(run_cfg.relevance_hidden),
        horizon_embedding_dim=int(cfg["gate"]["horizon_embedding_dim"]),
        dropout=float(cfg["gate"].get("dropout", 0.1)),
        initial_probability=float(run_cfg.initial_probability),
    )


def predict_event_rows(model, frame: pd.DataFrame, columns: dict[str, list[str]], cfg: dict, device: torch.device, run_cfg: Step7RunConfig) -> pd.DataFrame:
    ds = GateDataset(frame, columns)
    loader = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)
    gates = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["event"].to(device),
                batch["relation"].to(device),
                batch["stock"].to(device),
                _market_input(batch, run_cfg, device),
                batch["horizon"].to(device),
            )
            gates.append({k: v.detach().cpu().numpy() for k, v in out.items()})
    gate_frame = pd.concat([pd.DataFrame(g) for g in gates], ignore_index=True)
    out = pd.concat([frame.reset_index(drop=True), gate_frame], axis=1)
    out["model"] = run_cfg.model
    out["news_correction"] = out["news_correction_proxy"].astype(float)
    out["gated_correction"] = out["final_gate"].astype(float) * out["news_correction"].astype(float)
    return out


def _utility_supervision_loss(final_gate: torch.Tensor, batch: dict[str, torch.Tensor], run_cfg: Step7RunConfig, device: torch.device) -> torch.Tensor:
    loss = final_gate.new_tensor(0.0)
    if run_cfg.utility_weight > 0:
        labels = batch["utility_label"].to(device)
        mask = labels.ne(-1)
        if bool(mask.any()):
            target = labels[mask].float()
            loss = loss + float(run_cfg.utility_weight) * torch.nn.functional.binary_cross_entropy(final_gate[mask], target)
    if run_cfg.common_utility_weight > 0:
        labels = batch["event_common_utility_label"].to(device)
        mask = labels.ne(-1)
        if bool(mask.any()):
            target = labels[mask].float()
            loss = loss + float(run_cfg.common_utility_weight) * torch.nn.functional.binary_cross_entropy(final_gate[mask], target)
    return loss


def train_gate_model(run_cfg: Step7RunConfig, train_frame: pd.DataFrame, val_frame: pd.DataFrame, columns: dict[str, list[str]], cfg: dict, device: torch.device, ckpt_dir: str | Path):
    if run_cfg.model == "S0_StockOnly_G5":
        val = val_frame.copy()
        val["model"] = run_cfg.model
        val["reliability_gate"] = 0.0
        val["stock_relevance_gate"] = 0.0
        val["horizon_gate"] = 0.0
        val["final_gate"] = 0.0
        val["news_correction"] = val["news_correction_proxy"].astype(float)
        val["gated_correction"] = 0.0
        return val, {"best_epoch": 0, "best_metric": np.nan}
    model = build_model(run_cfg, columns, cfg).to(device)
    if run_cfg.model == "S2_FixedSmallGate":
        return predict_event_rows(model, val_frame, columns, cfg, device, run_cfg), {"best_epoch": 0, "best_metric": np.nan}
    train_ds = GateDataset(train_frame, columns)
    loader = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["gate_learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    best_metric = float("inf")
    best_epoch = 0
    patience = int(cfg["training"]["early_stopping_patience"])
    stale = 0
    for epoch in range(int(cfg["training"]["max_epochs"])):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(
                batch["event"].to(device),
                batch["relation"].to(device),
                batch["stock"].to(device),
                _market_input(batch, run_cfg, device),
                batch["horizon"].to(device),
            )
            correction = out["final_gate"] * batch["news_correction_proxy"].to(device)
            final_prediction = batch["stock_prediction"].to(device) + correction
            loss = gated_forecast_loss(final_prediction, batch["actual_logvol"].to(device))
            loss = loss + gate_regularization(out["final_gate"], correction, run_cfg.usage_weight, run_cfg.correction_weight)
            loss = loss + _utility_supervision_loss(out["final_gate"], batch, run_cfg, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("gradient_clip_norm", 1.0)))
            optimizer.step()
        pred = prediction_level_from_event_rows(predict_event_rows(model, val_frame, columns, cfg, device, run_cfg), cfg)
        metric = float(pred["qlike_loss"].mean())
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            stale = 0
            save_step7_checkpoint(Path(ckpt_dir) / "best.pt", model, optimizer, epoch, metric, run_cfg.as_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    return predict_event_rows(model, val_frame, columns, cfg, device, run_cfg), {"best_epoch": best_epoch, "best_metric": best_metric}
