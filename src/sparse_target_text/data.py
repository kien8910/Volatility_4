from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import yaml

from src.news.embedding_cache import EmbeddingCache
from src.news.text_encoder import build_text_encoder
from src.news.text_preprocessing import normalize_text, stable_text_hash
from .filtering import hard_filter_events
from .chunking import load_tokenizer, tokenizer_chunk_events

BASE_KEYS = ["date", "target_date", "ticker", "horizon", "fold_id"]


def _require(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [c for c in columns if c not in frame]
    if missing:
        raise ValueError(f"{label} missing required columns {missing}; found {list(frame.columns)}")


def load_g5_oof_baseline(cfg: dict, include_locked_test: bool = False) -> pd.DataFrame:
    filters = None if include_locked_test else [("split", "=", "validation")]
    raw = pd.read_parquet(cfg["data"]["baseline_predictions_path"], filters=filters)
    with Path(cfg["data"]["baseline_config_path"]).open("r", encoding="utf-8") as fh:
        selected_config = str(yaml.safe_load(fh)["config_id"])
    _require(raw, BASE_KEYS + ["split", "seed", "config_id", "actual_logvol", "p_prediction",
                               "residual_actual", "residual_prediction", "final_prediction"], "Step4 baseline")
    raw = raw.loc[raw.config_id.astype(str).eq(selected_config)].copy()
    baseline_seeds = [int(x) for x in cfg["data"].get("baseline_seeds", [42])]
    raw = raw.loc[raw.seed.isin(baseline_seeds)].copy()
    train_folds = set(int(x) for x in cfg["data"].get("analysis_train_folds", [1, 2, 3]))
    validation_fold = int(cfg["data"].get("analysis_validation_fold", 4))
    raw["analysis_split"] = np.where(raw.split.eq("test"), "test",
                              np.where(raw.fold_id.eq(validation_fold), "validation",
                              np.where(raw.fold_id.isin(train_folds), "train", "exclude")))
    raw = raw.loc[raw.analysis_split.ne("exclude")]
    if not include_locked_test:
        raw = raw.loc[raw.analysis_split.ne("test")]
    group = BASE_KEYS + ["analysis_split"]
    if raw.groupby(group).actual_logvol.nunique().max() != 1:
        raise ValueError("Actual outcomes disagree across Step4 seeds")
    baseline = raw.groupby(group, as_index=False).agg(
        actual_logvol=("actual_logvol", "first"), p_prediction=("p_prediction", "first"),
        residual_actual=("residual_actual", "first"), stock_residual_prediction=("residual_prediction", "mean"),
        baseline_prediction=("final_prediction", "mean"), baseline_prediction_seed_sd=("final_prediction", "std"),
        ensemble_seed_count=("seed", "nunique"))
    baseline = baseline.loc[baseline.horizon.isin([int(x) for x in cfg["data"]["horizons"]])].copy()
    baseline["baseline_error"] = baseline.actual_logvol - baseline.baseline_prediction
    baseline["date"] = pd.to_datetime(baseline.date); baseline["target_date"] = pd.to_datetime(baseline.target_date)
    if not (baseline.target_date > baseline.date).all():
        raise ValueError("target_date must be strictly after forecast-origin date")
    val_start = baseline.loc[baseline.analysis_split.eq("validation"), "date"].min()
    test_start = pd.Timestamp(cfg["data"]["locked_test_start"])
    eligible = ~((baseline.analysis_split.eq("train") & baseline.target_date.ge(val_start)) |
                 (baseline.analysis_split.eq("validation") & baseline.target_date.ge(test_start)))
    baseline = baseline.loc[eligible | baseline.analysis_split.eq("test")].copy()
    if baseline.duplicated(BASE_KEYS).any():
        raise ValueError("Seed-ensembled baseline is not unique on forecast keys")
    return add_origin_known_state_features(baseline)


def add_origin_known_state_features(baseline: pd.DataFrame) -> pd.DataFrame:
    out = baseline.sort_values(["date", "ticker", "horizon"]).copy()
    market = out.groupby(["date", "horizon"]).baseline_prediction.agg(["mean", "std"]).rename(
        columns={"mean": "market_mean_prediction", "std": "market_prediction_dispersion"}).reset_index()
    out = out.merge(market, on=["date", "horizon"], how="left", validate="many_to_one")
    out["baseline_vs_market"] = out.baseline_prediction - out.market_mean_prediction
    for window in [5, 22]:
        out[f"known_error_mean_{window}"] = 0.0; out[f"known_abs_error_mean_{window}"] = 0.0
    for (_, _), idx in out.groupby(["ticker", "horizon"], sort=False).groups.items():
        positions = list(idx); group = out.loc[positions].sort_values("date")
        history: list[tuple[pd.Timestamp, float]] = []
        values: dict[int, tuple[float, float, float, float]] = {}
        for row in group.itertuples():
            known = [err for available, err in history if available < pd.Timestamp(row.date)]
            vals = []
            for window in [5, 22]:
                recent = known[-window:]
                vals.extend([float(np.mean(recent)) if recent else 0.0,
                             float(np.mean(np.abs(recent))) if recent else 0.0])
            values[row.Index] = tuple(vals)
            history.append((pd.Timestamp(row.target_date), float(row.baseline_error)))
        for row_idx, vals in values.items():
            out.loc[row_idx, ["known_error_mean_5", "known_abs_error_mean_5",
                              "known_error_mean_22", "known_abs_error_mean_22"]] = vals
    return out.sort_values(["date", "ticker", "horizon"]).reset_index(drop=True)


def _effective_dates(events: pd.DataFrame, baseline: pd.DataFrame, lag: int) -> pd.Series:
    result = pd.Series(pd.NaT, index=events.index, dtype="datetime64[ns]")
    calendars = {ticker: np.asarray(sorted(group.date.unique()), dtype="datetime64[ns]")
                 for ticker, group in baseline.groupby("ticker")}
    for ticker, idx in events.groupby("ticker").groups.items():
        calendar = calendars.get(str(ticker))
        if calendar is None or not len(calendar):
            continue
        dates = events.loc[idx, "news_date"].to_numpy(dtype="datetime64[ns]")
        side = "left" if lag == 0 else "right"
        positions = np.searchsorted(calendar, dates, side=side) + max(0, lag - 1)
        valid = positions < len(calendar)
        result.loc[np.asarray(list(idx))[valid]] = pd.to_datetime(calendar[positions[valid]])
    return result


def load_target_event_candidates(cfg: dict, baseline: pd.DataFrame) -> pd.DataFrame:
    columns = ["row_id", "date", "ticker", "hierarchy", "category", "text", "text_hash",
               "is_missing", "is_duplicate_within_date"]
    news = pd.read_parquet(cfg["data"]["news_long_path"], columns=columns,
                           filters=[("hierarchy", "=", "target_company")])
    _require(news, ["date", "ticker", "hierarchy", "category", "text"], "news_long")
    news = news.loc[news.hierarchy.astype(str).eq("target_company") & news.ticker.isin(cfg["data"]["tickers"])].copy()
    news["news_date"] = pd.to_datetime(news.date).dt.normalize()
    # Avoid mapping arbitrarily old events to the first available OOF origin.
    news = news.loc[news.news_date.between(baseline.date.min() - pd.Timedelta(days=7), baseline.date.max())].copy()
    news["text"] = news.text.map(lambda x: normalize_text(x)[0])
    missing_flag = news["is_missing"].astype(bool) if "is_missing" in news else pd.Series(False, index=news.index)
    news = news.loc[news.text.ne("") & ~missing_flag].copy()
    news["text_hash"] = news.get("text_hash", news.text.map(stable_text_hash)).fillna(news.text.map(stable_text_hash)).astype(str)
    quarantine_path = cfg["data"].get("leakage_quarantine_path")
    if quarantine_path and Path(quarantine_path).exists():
        quarantine = pd.read_parquet(quarantine_path)
        hash_column = next((c for c in ["text_hash", "hash"] if c in quarantine), None)
        if hash_column:
            news = news.loc[~news.text_hash.isin(set(quarantine[hash_column].dropna().astype(str)))].copy()
    timestamp_col = cfg["information_cutoff"].get("timestamp_column")
    require_timestamp = bool(cfg["information_cutoff"].get("require_publication_timestamp", False))
    if timestamp_col and timestamp_col in news:
        news["publication_timestamp"] = pd.to_datetime(news[timestamp_col], errors="coerce", utc=True)
    else:
        if require_timestamp:
            raise ValueError("Publication timestamp is required but unavailable in news_long")
        news["publication_timestamp"] = pd.NaT
    lag = int(cfg["information_cutoff"].get("news_lag_sessions",
              cfg["information_cutoff"].get("news_lag_observed_days", 1)))
    news["effective_date"] = pd.NaT
    timestamp_known = news.publication_timestamp.notna()
    if bool(cfg["information_cutoff"].get("use_timestamp_cutoff_when_available", True)) and timestamp_known.any():
        cutoff_hour = int(str(cfg["information_cutoff"].get("forecast_cutoff_local_time", "16:00")).split(":")[0])
        cutoff_minute = int(str(cfg["information_cutoff"].get("forecast_cutoff_local_time", "16:00")).split(":")[1])
        timezone = str(cfg["information_cutoff"].get("market_timezone", "America/New_York"))
        timestamped = news.loc[timestamp_known].copy()
        local_time = timestamped.publication_timestamp.dt.tz_convert(timezone)
        timestamped["news_date"] = local_time.dt.tz_localize(None).dt.normalize()
        after_cutoff = (local_time.dt.hour > cutoff_hour) | ((local_time.dt.hour == cutoff_hour) &
                                                              (local_time.dt.minute >= cutoff_minute))
        timestamped.loc[after_cutoff, "news_date"] += pd.Timedelta(days=1)
        news.loc[timestamp_known, "effective_date"] = _effective_dates(timestamped, baseline, 0)
    missing_timestamp = ~timestamp_known
    if missing_timestamp.any():
        news.loc[missing_timestamp, "effective_date"] = _effective_dates(news.loc[missing_timestamp], baseline, lag)
    news["effective_date"] = pd.to_datetime(news.effective_date)
    news = news.loc[news.effective_date.notna()].copy()
    if lag > 0 and missing_timestamp.any() and not (news.loc[news.publication_timestamp.isna(), "effective_date"] >
                                                    news.loc[news.publication_timestamp.isna(), "news_date"]).all():
        raise ValueError("Lagged news alignment failed: effective_date must follow news_date")
    news["event_id"] = [hashlib.sha256(f"{d}|{t}|{c}|{h}".encode()).hexdigest()[:24]
                        for d, t, c, h in zip(news.news_date, news.ticker, news.category, news.text_hash)]
    return hard_filter_events(news, cfg)


def build_event_embedding_cache(events: pd.DataFrame, cfg: dict, device: str) -> pd.DataFrame:
    cache = EmbeddingCache(cfg["text_encoder"]["cache_dir"])
    existing = cache.read()
    encoder_name = str(cfg["text_encoder"]["model_name"]); pooling = str(cfg["text_encoder"]["pooling_method"])
    max_length = int(cfg["text_encoder"]["max_length"])
    tokenizer = load_tokenizer(cfg)
    events = tokenizer_chunk_events(events, cfg, tokenizer)
    request = events[["text_hash", "text"]].drop_duplicates("text_hash").copy()
    existing_keys = set()
    if not existing.empty:
        sub = existing.loc[(existing.encoder_name.astype(str).eq(encoder_name)) &
                           (existing.pooling_method.astype(str).eq(pooling)) &
                           (existing.max_length.astype(int).eq(max_length))]
        existing_keys = set(sub.text_hash.astype(str))
    missing = request.loc[~request.text_hash.astype(str).isin(existing_keys)]
    if len(missing):
        encoder = build_text_encoder(cfg, device)
        vectors = encoder.encode(missing.text.astype(str).tolist(), pooling_method=pooling)
        now = pd.Timestamp.utcnow().isoformat()
        add = pd.DataFrame({"text_hash": missing.text_hash.astype(str), "hierarchy": "target_company",
                            "encoder_name": encoder_name, "pooling_method": pooling, "max_length": max_length,
                            "embedding_dim": vectors.shape[1], "embedding": list(vectors), "missing_mask": 0,
                            "created_at": now})
        existing = pd.concat([existing, add], ignore_index=True).drop_duplicates(
            ["encoder_name", "text_hash", "pooling_method", "max_length"], keep="last")
        cache.write(existing.reset_index(drop=True)); cache._write_manifest(existing)
    return cache.read()


def attach_embeddings_and_novelty(events: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    events = tokenizer_chunk_events(events, cfg)
    cache = EmbeddingCache(cfg["text_encoder"]["cache_dir"]).read()
    encoder_name = str(cfg["text_encoder"]["model_name"]); pooling = str(cfg["text_encoder"]["pooling_method"])
    max_length = int(cfg["text_encoder"]["max_length"])
    cache = cache.loc[(cache.encoder_name.astype(str).eq(encoder_name)) &
                      (cache.pooling_method.astype(str).eq(pooling)) &
                      (cache.max_length.astype(int).eq(max_length))]
    out = events.merge(cache[["text_hash", "embedding", "embedding_dim"]].drop_duplicates("text_hash"),
                       on="text_hash", how="left", validate="many_to_one")
    if out.embedding.isna().any():
        missing = int(out.embedding.isna().sum())
        raise ValueError(f"{missing} event embeddings are missing; run --mode build-embedding-cache first")
    vectors = np.stack(out.embedding.map(lambda x: np.asarray(x, dtype=np.float32)))
    normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    novelty = np.ones(len(out), dtype=np.float32)
    for ticker, idx in out.groupby("ticker", sort=False).groups.items():
        ordered = sorted(idx, key=lambda i: (out.loc[i, "effective_date"], out.loc[i, "event_id"]))
        history = np.zeros(normalized.shape[1], dtype=np.float64); n = 0; pending: list[int] = []; current = None
        for i in ordered:
            date = pd.Timestamp(out.loc[i, "effective_date"])
            if current is not None and date != current:
                for j in pending: history += normalized[j]; n += 1
                pending = []
            if n:
                centroid = history / n
                novelty[i] = np.clip(1.0 - normalized[i].dot(centroid) / max(np.linalg.norm(centroid), 1e-12), 0.0, 2.0)
            pending.append(i); current = date
    out["semantic_novelty"] = novelty
    out["embedding"] = list(vectors)
    return out


def shuffle_event_payload_within_day(events: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(seed); out = events.copy()
    payload = ["text_hash", "embedding", "event_type", "catalyst_score", "semantic_novelty", "word_count"]
    changed = 0; wrong_ticker = 0; eligible = 0
    out["payload_source_ticker"] = out.ticker.astype(str)
    for _, idx in out.groupby("effective_date", sort=False).groups.items():
        idx = np.asarray(list(idx))
        if len(idx) <= 1: continue
        target_tickers = out.loc[idx, "ticker"].astype(str).to_numpy()
        shifts = np.arange(1, len(idx)); rng.shuffle(shifts)
        shift = int(max(shifts, key=lambda s: np.sum(target_tickers != np.roll(target_tickers, s))))
        source = np.roll(idx, shift); source_tickers = out.loc[source, "ticker"].astype(str).to_numpy()
        original = out.loc[idx, "text_hash"].to_numpy(copy=True)
        out.loc[idx, payload] = out.loc[source, payload].to_numpy()
        out.loc[idx, "payload_source_ticker"] = source_tickers
        changed += int(np.sum(original != out.loc[idx, "text_hash"].to_numpy()))
        wrong_ticker += int(np.sum(target_tickers != source_tickers)); eligible += len(idx)
    out["placebo_type"] = "within_day_payload_shuffle"
    out["wrong_ticker_payload"] = out.payload_source_ticker.ne(out.ticker).astype(int)
    return out, {"payload_changed_rate": float(changed / max(len(out), 1)),
                 "wrong_ticker_rate_on_shuffle_eligible": float(wrong_ticker / max(eligible, 1)),
                 "shuffle_eligible_rate": float(eligible / max(len(out), 1))}
