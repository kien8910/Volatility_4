from __future__ import annotations

import importlib.util
import json
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class FitResult:
    y_pred: np.ndarray
    hyperparams: dict
    status: str = "ok"
    training_time_sec: float = 0.0
    inference_time_sec: float = 0.0
    param_count: int = 0
    message: str = ""


def _timed_predict(model, x_val):
    start = time.perf_counter()
    pred = model.predict(x_val)
    return pred, time.perf_counter() - start


def mean_global(train: pd.DataFrame, val: pd.DataFrame, target: str, **_) -> FitResult:
    start = time.perf_counter()
    mu = float(train[target].mean())
    return FitResult(np.full(len(val), mu), {}, training_time_sec=time.perf_counter() - start)


def mean_ticker(train: pd.DataFrame, val: pd.DataFrame, target: str, **_) -> FitResult:
    start = time.perf_counter()
    fallback = float(train[target].mean())
    means = train.groupby("ticker")[target].mean().to_dict()
    pred = val["ticker"].map(means).fillna(fallback).to_numpy(dtype=float)
    return FitResult(pred, {}, training_time_sec=time.perf_counter() - start)


def last_value(train: pd.DataFrame, val: pd.DataFrame, target: str, **_) -> FitResult:
    return FitResult(val["logvol_gk"].to_numpy(dtype=float), {})


def fit_sklearn(train: pd.DataFrame, val: pd.DataFrame, target: str, features: list[str], factory: Callable[[], object], hyper: dict) -> FitResult:
    start = time.perf_counter()
    model = factory()
    model.fit(train[features].to_numpy(dtype=float), train[target].to_numpy(dtype=float))
    training = time.perf_counter() - start
    pred, infer = _timed_predict(model, val[features].to_numpy(dtype=float))
    params = 0
    estimator = model[-1] if hasattr(model, "__getitem__") else model
    if hasattr(estimator, "coef_"):
        params += int(np.asarray(estimator.coef_).size)
    if hasattr(estimator, "intercept_"):
        params += int(np.asarray(estimator.intercept_).size)
    return FitResult(pred, hyper, training_time_sec=training, inference_time_sec=infer, param_count=params)


def har_ols(train: pd.DataFrame, val: pd.DataFrame, target: str, **_) -> FitResult:
    return fit_sklearn(train, val, target, ["har_d", "har_w", "har_m"], LinearRegression, {})


def har_ridge(train: pd.DataFrame, val: pd.DataFrame, target: str, alpha: float, **_) -> FitResult:
    return fit_sklearn(train, val, target, ["har_d", "har_w", "har_m"], lambda: Ridge(alpha=alpha), {"alpha": alpha})


def ar_model(train: pd.DataFrame, val: pd.DataFrame, target: str, p: int, **_) -> FitResult:
    features = [f"lag_{i}" for i in range(p)]
    return fit_sklearn(train, val, target, features, LinearRegression, {"p": p})


def linear_history(train: pd.DataFrame, val: pd.DataFrame, target: str, lookback: int, **_) -> FitResult:
    features = [f"lag_{i}" for i in range(lookback)]
    return fit_sklearn(train, val, target, features, lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)), {"lookback": lookback})


def mlp_history(train: pd.DataFrame, val: pd.DataFrame, target: str, lookback: int, seed: int, hidden=(64, 32), max_iter: int = 120, **_) -> FitResult:
    features = [f"lag_{i}" for i in range(lookback)]
    return fit_sklearn(
        train,
        val,
        target,
        features,
        lambda: make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=tuple(hidden), activation="relu", early_stopping=True, random_state=seed, max_iter=max_iter, n_iter_no_change=10),
        ),
        {"lookback": lookback, "seed": seed, "hidden": list(hidden), "max_iter": max_iter},
    )


def dlinear_history(train: pd.DataFrame, val: pd.DataFrame, target: str, lookback: int, seed: int, max_epochs: int, patience: int, learning_rate: float, **_) -> FitResult:
    import torch

    features = [f"lag_{i}" for i in range(lookback)]
    x_train = train[features].to_numpy(dtype=np.float32)
    y_train = train[target].to_numpy(dtype=np.float32).reshape(-1, 1)
    x_val = val[features].to_numpy(dtype=np.float32)
    y_val = val[target].to_numpy(dtype=np.float32).reshape(-1)
    torch.manual_seed(seed)
    model = torch.nn.Linear(lookback, 1)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    xt = torch.from_numpy(x_train)
    yt = torch.from_numpy(y_train)
    xv = torch.from_numpy(x_val)
    best_state = None
    best = np.inf
    stale = 0
    start = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
        with torch.no_grad():
            pred_val = model(xv).numpy().reshape(-1)
        ratio = np.maximum(np.exp(np.clip(2.0 * y_val, -60.0, 60.0)), 1.0e-12) / np.maximum(np.exp(np.clip(2.0 * pred_val, -60.0, 60.0)), 1.0e-12)
        val_qlike = float(np.mean(ratio - np.log(ratio) - 1.0))
        if val_qlike < best - 1.0e-8:
            best = val_qlike
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    training = time.perf_counter() - start
    start = time.perf_counter()
    with torch.no_grad():
        pred = model(torch.from_numpy(x_val)).numpy().reshape(-1)
    infer = time.perf_counter() - start
    params = sum(p.numel() for p in model.parameters())
    return FitResult(
        pred,
        {"lookback": lookback, "seed": seed, "max_epochs": max_epochs, "patience": patience, "learning_rate": learning_rate},
        training_time_sec=training,
        inference_time_sec=infer,
        param_count=int(params),
    )


def garch_family(train: pd.DataFrame, val: pd.DataFrame, target: str, horizon: int, family: str, distribution: str, epsilon: float, **_) -> FitResult:
    if importlib.util.find_spec("arch") is None:
        return FitResult(
            np.full(len(val), np.nan),
            {"family": family, "distribution": distribution},
            status="failed",
            message="Python package 'arch' is not installed; GARCH-family baselines were not run.",
        )
    from arch import arch_model

    start = time.perf_counter()
    returns = train["log_return"].dropna().to_numpy(dtype=float) * 100.0
    if len(returns) < 100:
        return FitResult(np.full(len(val), np.nan), {"family": family, "distribution": distribution}, status="failed", message="Too few returns for GARCH.")
    vol = "GARCH"
    o = 1 if family == "GJR-GARCH" else 0
    model = arch_model(returns, mean="Constant", vol=vol, p=1, o=o, q=1, dist=distribution, rescale=False)
    fit = model.fit(disp="off")
    training = time.perf_counter() - start
    start = time.perf_counter()
    fc = fit.forecast(horizon=horizon, reindex=False)
    var = float(fc.variance.iloc[-1, horizon - 1]) / (100.0 ** 2)
    pred = np.full(len(val), 0.5 * np.log(max(var, epsilon)))
    infer = time.perf_counter() - start
    return FitResult(pred, {"family": family, "distribution": distribution}, training_time_sec=training, inference_time_sec=infer, param_count=len(fit.params))


def hyper_json(hyper: dict) -> str:
    return json.dumps(hyper, sort_keys=True)
