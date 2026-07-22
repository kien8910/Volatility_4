from __future__ import annotations
import numpy as np
import pandas as pd
from src.graph.metrics import qlike_from_logvol


def add_losses(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    out["qlike_loss"] = qlike_from_logvol(out.actual_logvol, out.final_prediction)[0]
    out["absolute_error"] = (out.actual_logvol - out.final_prediction).abs()
    out["squared_error"] = (out.actual_logvol - out.final_prediction).square()
    return out


def metric_table(predictions: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    return (predictions.groupby(groups, dropna=False).agg(n=("qlike_loss", "size"), qlike=("qlike_loss", "mean"),
            mae=("absolute_error", "mean"), mse=("squared_error", "mean"),
            mean_abs_correction=("correction", lambda x: x.abs().mean()),
            correction_std=("correction", "std")).reset_index())


def gate_diagnostics(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame(columns=["model", "analysis_split", "horizon", "n_edges", "selection_rate", "gate_mean"])
    return (edges.groupby(["model", "analysis_split", "horizon"], dropna=False).agg(
        n_edges=("edge_gate", "size"), selection_rate=("selected", "mean"), gate_mean=("edge_gate", "mean"),
        gate_std=("edge_gate", "std"), novelty_selected=("semantic_novelty", lambda x: x[edges.loc[x.index, "selected"].eq(1)].mean()))
        .reset_index())


def select_variant(metrics: pd.DataFrame, ticker_metrics: pd.DataFrame, gates: pd.DataFrame, cfg: dict) -> dict:
    q = metrics.set_index("model").qlike.to_dict()
    real_variants = ["T1_all_target", "T2_hard_top1", "T3_sparse_hurdle"]
    required = ["M0_stock_only"] + real_variants + [f"{x}_placebo" for x in real_variants]
    missing = [x for x in required if x not in q]
    if missing: raise ValueError(f"Cannot select configuration; missing validation models {missing}")
    best_variant = min(real_variants, key=lambda x: q[x]); matching_placebo = f"{best_variant}_placebo"
    ticker = ticker_metrics.pivot(index="ticker", columns="model", values="qlike")
    breadth = int((ticker[best_variant] < ticker["M0_stock_only"]).sum())
    gate = gates.loc[(gates.model.eq(best_variant)) & gates.analysis_split.eq("validation")]
    selection_rate = float(np.average(gate.selection_rate, weights=gate.n_edges)) if len(gate) else np.nan
    beats_all = q[best_variant] < q["M0_stock_only"] and q[best_variant] < q[matching_placebo]
    gate_ok = (best_variant != "T3_sparse_hurdle" or
               (np.isfinite(selection_rate) and selection_rate <= float(cfg["selection"]["max_event_selection_rate"])))
    if beats_all and breadth >= int(cfg["selection"]["minimum_improved_tickers"]) and gate_ok:
        decision = "GO"
    elif q[best_variant] < q[matching_placebo] and q[best_variant] < q["M0_stock_only"]:
        decision = "WEAK-GO"
    else:
        decision = "NO-GO"
    return {"decision": decision, "best_real_variant": best_variant,
            "matching_placebo": matching_placebo,
            "selected_variant": best_variant if decision != "NO-GO" else "M0_stock_only",
            "validation_qlike": q, "improved_tickers": breadth, "validation_event_selection_rate": selection_rate,
            "locked_test_used": False}
