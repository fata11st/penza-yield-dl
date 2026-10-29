"""Эталонные модели: наивный прогноз, линейный тренд, Ridge с экспоненциальным взвешиванием лет."""
from __future__ import annotations

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from penza_yield.validation import exponential_weights


def predict_naive(train, test) -> float:
    """Урожайность прошлого года."""
    return float(test["lag1"].iloc[0])


def predict_trend(train, test) -> float:
    """Линейный тренд урожайности по годам окна обучения."""
    model = LinearRegression().fit(train.index.to_numpy()[:, None], train["y"].to_numpy())
    return float(model.predict(test.index.to_numpy()[:, None])[0])


def predict_ridge(train, test, features: list[str], alpha: float, decay: float) -> float:
    """Ridge: стандартизация по окну обучения, веса лет decay ** (t_last − t)."""
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    w = exponential_weights(train.index, decay)
    model.fit(train[features].to_numpy(), train["y"].to_numpy(), ridge__sample_weight=w)
    return float(model.predict(test[features].to_numpy())[0])


def usable_rows(frame, features: list[str]):
    """Строки без пропусков в признаках и цели (например, NDVI появляется только с 2000 г.)."""
    return frame.dropna(subset=[*features, "y"])


RIDGE_GRID = {
    "alpha": [0.1, 1.0, 10.0, 100.0],
    "decay": [0.8, 0.9, 1.0],
}


def ridge_configs(feature_sets: dict[str, list[str]]):
    for fs in feature_sets:
        for a in RIDGE_GRID["alpha"]:
            for d in RIDGE_GRID["decay"]:
                yield {"features": fs, "alpha": a, "decay": d}


def config_key(cfg: dict) -> str:
    return f"{cfg['features']}|a={cfg['alpha']}|d={cfg['decay']}"

