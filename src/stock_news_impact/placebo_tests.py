from __future__ import annotations

import numpy as np
import pandas as pd


def wrong_ticker_placebo(pairs: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    out = pairs.copy()
    pos = {ticker: i for i, ticker in enumerate(tickers)}
    out["target_ticker"] = out["target_ticker"].map(lambda x: tickers[(pos.get(str(x), 0) + 1) % len(tickers)])
    out["is_direct_target"] = 0
    out["placebo_type"] = "wrong_ticker"
    return out


def time_shift_placebo(events: pd.DataFrame, shift_days: int) -> pd.DataFrame:
    out = events.copy()
    out["date"] = pd.to_datetime(out["date"]) + pd.to_timedelta(int(shift_days), unit="D")
    out["event_id"] = out["event_id"].astype(str) + f"_shift_{shift_days}"
    return out


def random_ticker_placebo(pairs: pd.DataFrame, tickers: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    out = pairs.copy()
    out["target_ticker"] = rng.choice(tickers, size=len(out))
    out["is_direct_target"] = 0
    out["placebo_type"] = f"random_ticker_{seed}"
    return out
