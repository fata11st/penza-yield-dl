"""Табличные модели на годовых признаках (E05 — MLP, E06 — TabPFN).

Постановка как у эталонов E01 («по окончании вегетации»), отдельная модель на культуру. Цель — log(y/ma3),
прогноз ŷ = ma3 · exp(r̂), как у моделей сезона: все нейросетевые модели предсказывают отклонение от якоря.
Стандартизация признаков и цели — по окну обучения.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from penza_yield import data
from penza_yield.features import FEATURE_SETS, build_features


def crop_frames() -> dict[str, pd.DataFrame]:
    annual, monthly = data.load_annual(), data.load_monthly(check=False)
    frames = {}
    for t in data.TARGETS:
        f = build_features(annual, monthly, t)
        f["r"] = np.log(f["y"] / f["ma3"])
        frames[t] = f
    return frames


def split(frame: pd.DataFrame, year: int, features: list[str], allow_nan: bool = False):
    """Окно обучения — все годы < year с определённой целью (и признаками, если NaN не допускаются)."""
    need = ["r"] if allow_nan else ["r", *features]
    train = frame[frame.index < year].dropna(subset=need)
    test = frame.loc[[year]]
    if train.index.max() >= year:
        raise ValueError("Утечка: в обучении год не раньше прогнозируемого")
    return train, test


def fit_predict_mlp(frame, year, features, hidden: int, alpha: float, seeds) -> float:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    train, test = split(frame, year, features)
    sx = StandardScaler().fit(train[features])
    r_mu, r_sd = train.r.mean(), train.r.std() + 1e-8
    x_tr, x_te = sx.transform(train[features]), sx.transform(test[features])
    y_tr = (train.r.to_numpy() - r_mu) / r_sd
    preds = [MLPRegressor(hidden_layer_sizes=(hidden,), activation="tanh", alpha=alpha, solver="lbfgs",
                          max_iter=2000, random_state=s).fit(x_tr, y_tr).predict(x_te)[0] for s in seeds]
    return float(test.ma3.iloc[0] * np.exp(np.mean(preds) * r_sd + r_mu))


def fit_predict_tabpfn(frame, year, features, seeds, device: str = "auto") -> float:
    from tabpfn import TabPFNRegressor

    train, test = split(frame, year, features, allow_nan=True)  # TabPFN сам обрабатывает пропуски (NDVI до 2000 г.)
    r_mu, r_sd = train.r.mean(), train.r.std() + 1e-8
    y_tr = (train.r.to_numpy() - r_mu) / r_sd
    preds = []
    for s in seeds:
        model = TabPFNRegressor(random_state=s, device=device)
        model.fit(train[features].to_numpy(float), y_tr)
        preds.append(float(model.predict(test[features].to_numpy(float))[0]))
    return float(test.ma3.iloc[0] * np.exp(np.mean(preds) * r_sd + r_mu))


__all__ = ["crop_frames", "split", "fit_predict_mlp", "fit_predict_tabpfn", "FEATURE_SETS"]
