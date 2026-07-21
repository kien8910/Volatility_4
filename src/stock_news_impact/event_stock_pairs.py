from __future__ import annotations

import numpy as np
import pandas as pd


def _date_horizon_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions[["date", "target_date", "ticker", "horizon", "split", "fold_id", "seed"]].drop_duplicates()


def build_event_stock_pairs(
    events: pd.DataFrame,
    stock_predictions: pd.DataFrame,
    tickers: list[str],
    spillover_mode: str = "target_only",
    adjacency: np.ndarray | None = None,
    max_pairs: int | None = None,
) -> pd.DataFrame:
    base = _date_horizon_rows(stock_predictions).copy()
    base["date"] = pd.to_datetime(base["date"])
    events = events.copy()
    events["date"] = pd.to_datetime(events["date"])
    by_date = {pd.Timestamp(d): g for d, g in base.groupby("date", sort=False)}
    rows: list[dict] = []
    ticker_pos = {ticker: i for i, ticker in enumerate(tickers)}
    for event in events.itertuples(index=False):
        date = pd.Timestamp(event.date)
        if date not in by_date:
            continue
        if event.event_scope in {"macro", "sector"} or spillover_mode == "broadcast_to_all":
            target_tickers = tickers
        else:
            context = str(event.context_ticker)
            if context not in ticker_pos:
                continue
            target_tickers = [context]
            if spillover_mode == "target_plus_graph_neighbors" and adjacency is not None:
                weights = adjacency[ticker_pos[context]]
                target_tickers = [tickers[i] for i, w in enumerate(weights) if float(w) > 0 or tickers[i] == context]
        day_targets = by_date[date]
        for target_ticker in target_tickers:
            sub = day_targets.loc[day_targets["ticker"].astype(str).eq(target_ticker)]
            for row in sub.itertuples(index=False):
                direct = int(str(event.context_ticker) == target_ticker and str(event.context_ticker) != "")
                graph_weight = 1.0 if direct else 0.0
                graph_distance = 0 if direct else (1 if event.event_scope in {"macro", "sector"} else 99)
                rows.append(
                    {
                        "event_id": event.event_id,
                        "date": date,
                        "target_date": pd.Timestamp(row.target_date),
                        "source_ticker": event.source_ticker,
                        "target_ticker": target_ticker,
                        "context_ticker": event.context_ticker,
                        "hierarchy": event.hierarchy,
                        "event_scope": event.event_scope,
                        "horizon": int(row.horizon),
                        "split": row.split,
                        "fold_id": int(row.fold_id),
                        "seed": int(row.seed),
                        "is_direct_target": direct,
                        "static_graph_weight": graph_weight,
                        "static_graph_distance": graph_distance,
                        "placebo_type": "real",
                    }
                )
                if max_pairs is not None and len(rows) >= int(max_pairs):
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows).sort_values(["date", "event_id", "target_ticker", "horizon", "fold_id", "seed"]).reset_index(drop=True)


def validate_pairs(pairs: pd.DataFrame) -> None:
    required = ["event_id", "date", "target_date", "target_ticker", "horizon", "fold_id", "seed", "is_direct_target"]
    missing = [col for col in required if col not in pairs.columns]
    if missing:
        raise ValueError(f"Event-stock pairs missing columns: {missing}")
    if not (pd.to_datetime(pairs["target_date"]) > pd.to_datetime(pairs["date"])).all():
        raise ValueError("Step 7 event-stock pairs contain target_date <= event date.")
