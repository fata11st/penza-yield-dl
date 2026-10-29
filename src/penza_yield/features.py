"""Годовые признаки для эталонных моделей.

Все признаки — детерминированные преобразования: собственная урожайность только прошлых лет (лаги, скользящее
среднее), погода и почва текущего сезона (постановка «прогноз по окончании вегетации»), тренд.
Обучаемых параметров здесь нет, поэтому построение признаков по всему датасету не даёт утечки;
масштабирование выполняется внутри окна обучения (в модели).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from penza_yield.data import MonthlyData

# Наборы признаков: ключ — имя набора в конфигурациях и отчётах
FEATURE_SETS = {
    # Только инерция агросистемы и тренд
    "ar": ["lag1", "lag2", "ma3", "time_index"],
    # + агрометеорология сезона (метеостанция и ERA5-Land; есть за все годы)
    "ar_climate": ["lag1", "lag2", "ma3", "time_index",
                   "t_summer", "p_summer", "p_veget", "t_may", "swc_spring", "pet_summer"],
    # + NDVI (только с 2000 г.)
    "ar_climate_ndvi": ["lag1", "lag2", "ma3", "time_index",
                        "t_summer", "p_summer", "p_veget", "t_may", "swc_spring", "pet_summer",
                        "ndvi_peak", "ndvi_jul_aug"],
}


def _season(m: MonthlyData, var: str, months: list[int], how: str = "mean") -> pd.Series:
    x = m.values[:, [k - 1 for k in months], m.variables.index(var)]
    with warnings.catch_warnings():  # годы без NDVI ожидаемо дают пустые срезы
        warnings.simplefilter("ignore", RuntimeWarning)
        agg = {"max": np.nanmax, "sum": np.nansum}.get(how, np.nanmean)(x, axis=1)
    # год без единого значения — NaN (nansum иначе дал бы 0)
    agg = np.where(np.isnan(x).all(axis=1), np.nan, agg)
    return pd.Series(agg, index=m.years)


def build_features(annual: pd.DataFrame, monthly: MonthlyData, target: str) -> pd.DataFrame:
    """Таблица признаков для одной культуры, индекс — год. Столбец 'y' — целевая переменная."""
    y = annual[target]
    f = pd.DataFrame(index=annual.index)
    f["y"] = y
    f["lag1"] = y.shift(1)
    f["lag2"] = y.shift(2)
    f["ma3"] = y.shift(1).rolling(3).mean()
    f["time_index"] = annual.index - annual.index.min()

    f["t_summer"] = _season(monthly, "mean_temp", [6, 7, 8])
    f["p_summer"] = _season(monthly, "rf_sum", [6, 7, 8], "sum")
    f["p_veget"] = _season(monthly, "rf_sum", [5, 6, 7, 8, 9], "sum")
    f["t_may"] = _season(monthly, "mean_temp", [5])
    f["swc_spring"] = _season(monthly, "swc_7_28cm", [4, 5, 6])
    f["pet_summer"] = _season(monthly, "pet_mm", [6, 7, 8], "sum")
    f["ndvi_peak"] = _season(monthly, "ndvi", [6, 7, 8], "max")
    f["ndvi_jul_aug"] = _season(monthly, "ndvi", [7, 8])
    return f
