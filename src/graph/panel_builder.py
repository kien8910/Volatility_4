from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GraphPanels:
    dates: pd.DatetimeIndex
    tickers: list[str]
    horizons: list[int]
    residual_state: np.ndarray
    raw_state: np.ndarray
    target_residual: np.ndarray
    target_actual: np.ndarray
    p_prediction: np.ndarray
    target_dates: np.ndarray
    split: np.ndarray
    fold_id: np.ndarray


@dataclass(frozen=True)
class GraphSampleTable:
    sample_dates: pd.DatetimeIndex
    tickers: list[str]
    horizons: list[int]
    residual_windows: np.ndarray
    raw_windows: np.ndarray
    target_residual: np.ndarray
    target_actual: np.ndarray
    p_prediction: np.ndarray
    target_dates: np.ndarray
    split: np.ndarray
    fold_id: np.ndarray
    lookback: int


def _pivot_state(state: pd.DataFrame, value_col: str, tickers: list[str]) -> pd.DataFrame:
    pivot = state.pivot(index="date", columns="ticker", values=value_col).sort_index()
    return pivot.reindex(columns=tickers)


def _complete_target_panel(targets: pd.DataFrame, dates: pd.DatetimeIndex, tickers: list[str], horizons: list[int]):
    n_dates, n_tickers, n_h = len(dates), len(tickers), len(horizons)
    residual = np.full((n_dates, n_tickers, n_h), np.nan, dtype=np.float32)
    actual = np.full_like(residual, np.nan)
    p_pred = np.full_like(residual, np.nan)
    target_dates = np.empty((n_dates, n_h), dtype="datetime64[ns]")
    target_dates[:] = np.datetime64("NaT")
    split = np.full(n_dates, "", dtype=object)
    fold_id = np.full(n_dates, -1, dtype=np.int64)

    date_pos = {date: idx for idx, date in enumerate(dates)}
    ticker_pos = {ticker: idx for idx, ticker in enumerate(tickers)}
    horizon_pos = {int(h): idx for idx, h in enumerate(horizons)}

    for row in targets.itertuples(index=False):
        d = pd.Timestamp(row.date)
        h = int(row.horizon)
        if d not in date_pos or row.ticker not in ticker_pos or h not in horizon_pos:
            continue
        i, j, k = date_pos[d], ticker_pos[row.ticker], horizon_pos[h]
        residual[i, j, k] = float(row.residual_target)
        actual[i, j, k] = float(row.actual_target)
        p_pred[i, j, k] = float(row.p_prediction)
        target_dates[i, k] = np.datetime64(pd.Timestamp(row.target_date).to_datetime64())
        split[i] = str(row.base_split)
        fold_id[i] = int(getattr(row, "fold_id", -1)) if pd.notna(getattr(row, "fold_id", -1)) else -1

    return residual, actual, p_pred, target_dates, split, fold_id


def build_panels(state: pd.DataFrame, targets: pd.DataFrame, tickers: list[str], horizons: list[int]) -> GraphPanels:
    residual_state = _pivot_state(state, "residual_state_h1", tickers)
    raw_state = _pivot_state(state, "actual_logvol_gk", tickers)
    common_dates = residual_state.index.intersection(raw_state.index)
    residual_state = residual_state.loc[common_dates]
    raw_state = raw_state.loc[common_dates]

    target_res, target_actual, p_pred, target_dates, split, fold_id = _complete_target_panel(
        targets, common_dates, tickers, horizons
    )
    return GraphPanels(
        dates=pd.DatetimeIndex(common_dates),
        tickers=tickers,
        horizons=horizons,
        residual_state=residual_state.to_numpy(dtype=np.float32),
        raw_state=raw_state.to_numpy(dtype=np.float32),
        target_residual=target_res,
        target_actual=target_actual,
        p_prediction=p_pred,
        target_dates=target_dates,
        split=split,
        fold_id=fold_id,
    )


def build_sample_table(panels: GraphPanels, lookback: int) -> GraphSampleTable:
    residual_windows = []
    raw_windows = []
    y_res, y_actual, p_pred, target_dates = [], [], [], []
    dates, split, fold_id = [], [], []

    for end in range(lookback - 1, len(panels.dates)):
        start = end - lookback + 1
        res_window = panels.residual_state[start : end + 1]
        raw_window = panels.raw_state[start : end + 1]
        if not np.isfinite(res_window).all() or not np.isfinite(raw_window).all():
            continue
        if not np.isfinite(panels.target_residual[end]).all():
            continue
        if not np.isfinite(panels.target_actual[end]).all() or not np.isfinite(panels.p_prediction[end]).all():
            continue
        if panels.split[end] not in {"train", "validation", "test"}:
            continue
        residual_windows.append(res_window.T)
        raw_windows.append(raw_window.T)
        y_res.append(panels.target_residual[end])
        y_actual.append(panels.target_actual[end])
        p_pred.append(panels.p_prediction[end])
        target_dates.append(panels.target_dates[end])
        dates.append(panels.dates[end])
        split.append(panels.split[end])
        fold_id.append(panels.fold_id[end])

    if not dates:
        raise ValueError(f"No complete graph samples could be built for lookback={lookback}.")

    return GraphSampleTable(
        sample_dates=pd.DatetimeIndex(dates),
        tickers=panels.tickers,
        horizons=panels.horizons,
        residual_windows=np.asarray(residual_windows, dtype=np.float32),
        raw_windows=np.asarray(raw_windows, dtype=np.float32),
        target_residual=np.asarray(y_res, dtype=np.float32),
        target_actual=np.asarray(y_actual, dtype=np.float32),
        p_prediction=np.asarray(p_pred, dtype=np.float32),
        target_dates=np.asarray(target_dates, dtype="datetime64[ns]"),
        split=np.asarray(split, dtype=object),
        fold_id=np.asarray(fold_id, dtype=np.int64),
        lookback=lookback,
    )


def split_indices(samples: GraphSampleTable, fold_id: int | None = None) -> dict[str, np.ndarray]:
    if fold_id is None or fold_id < 0:
        return {
            "train": np.flatnonzero(samples.split == "train"),
            "validation": np.flatnonzero(samples.split == "validation"),
            "test": np.flatnonzero(samples.split == "test"),
            "development": np.flatnonzero(np.isin(samples.split, ["train", "validation"])),
        }
    train = np.flatnonzero((samples.split == "train") | ((samples.split == "validation") & (samples.fold_id < fold_id)))
    validation = np.flatnonzero((samples.split == "validation") & (samples.fold_id == fold_id))
    return {
        "train": train,
        "validation": validation,
        "test": np.flatnonzero(samples.split == "test"),
        "development": np.flatnonzero(np.isin(samples.split, ["train", "validation"])),
    }

