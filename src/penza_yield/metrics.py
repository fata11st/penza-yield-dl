"""Метрики точности и статистическое сравнение прогнозов."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def mape(y_true, y_pred) -> float:
    """Средняя абсолютная процентная ошибка, %."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if np.any(y_true == 0):
        raise ValueError("MAPE не определена при нулевых фактических значениях")
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


@dataclass(frozen=True)
class DMResult:
    statistic: float
    p_value: float
    mean_loss_diff: float  # < 0: первая модель точнее


def diebold_mariano(y_true, pred_a, pred_b, h: int = 1, loss: str = "ape") -> DMResult:
    """Тест Диболда–Мариано с поправкой Харви–Лейборна–Ньюболда для малых выборок.

    H0: прогнозы A и B одинаково точны. Двусторонний тест, распределение Стьюдента с n−1 степенями свободы.
    loss: "ape" — абсолютная процентная ошибка (согласуется с MAPE), "se" — квадрат ошибки, "ae" — модуль ошибки.
    """
    y = np.asarray(y_true, float)
    ea, eb = y - np.asarray(pred_a, float), y - np.asarray(pred_b, float)
    if loss == "ape":
        d = np.abs(ea / y) - np.abs(eb / y)
    elif loss == "se":
        d = ea**2 - eb**2
    elif loss == "ae":
        d = np.abs(ea) - np.abs(eb)
    else:
        raise ValueError(f"Неизвестная функция потерь: {loss}")

    n = d.size
    if n < 3:
        raise ValueError("Для теста нужно не меньше 3 прогнозов")
    d_mean = d.mean()
    # Долгосрочная дисперсия с автоковариациями до лага h−1
    gamma = [np.sum((d[k:] - d_mean) * (d[: n - k] - d_mean)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return DMResult(float("nan"), float("nan"), float(d_mean))
    dm = d_mean / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * hln)
    p = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return DMResult(stat, p, float(d_mean))
