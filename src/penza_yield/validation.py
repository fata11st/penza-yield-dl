"""Схемы проверки прогнозов без заглядывания в будущее.

Основная схема — расширяющееся окно (expanding window): на шаге k модель обучается на годах
[train_start, t_k] и прогнозирует год t_k + 1. Для честной оценки моделей с подбором гиперпараметров —
вложенная схема: подбор только на годах ``tune_years``, итоговая оценка — на ``eval_years``,
которые в подборе не участвуют.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

# Протокол по умолчанию (см. docs/PLAN.md): первый прогноз — 2011 г. (засуха 2010 г. уже в обучении),
# подбор на 2011–2018, итоговая оценка на 2019–2025.
FIRST_TEST_YEAR = 2011
TUNE_YEARS = tuple(range(2011, 2019))
EVAL_YEARS = tuple(range(2019, 2026))


@dataclass(frozen=True)
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_year: int


def expanding_window(
    years: Sequence[int],
    test_years: Sequence[int],
    train_start: int | None = None,
    min_train: int = 1,
) -> Iterator[Fold]:
    """Одношаговые прогнозы: для каждого года из test_years — обучение на всех годах строго раньше него.

    years: годы наблюдений (порядок строк данных);
    train_start: первый год, включаемый в обучение (например, 2000 для моделей с NDVI);
    min_train: минимальное число лет в обучении, иначе ValueError.
    """
    years = np.asarray(years)
    for ty in test_years:
        test_idx = np.flatnonzero(years == ty)
        if test_idx.size != 1:
            raise ValueError(f"Год {ty} встречается в данных {test_idx.size} раз(а)")
        train = years < ty
        if train_start is not None:
            train &= years >= train_start
        train_idx = np.flatnonzero(train)
        if train_idx.size < min_train:
            raise ValueError(f"Для {ty} в обучении {train_idx.size} лет < min_train={min_train}")
        yield Fold(train_idx, test_idx, int(ty))


def exponential_weights(train_years: Sequence[int], decay: float) -> np.ndarray:
    """Веса наблюдений w_i = decay ** (t_last − t_i), как в Ridge-методике (λ ∈ [0.70, 1.00])."""
    if not 0 < decay <= 1:
        raise ValueError("decay должен быть в (0, 1]")
    t = np.asarray(train_years, dtype=float)
    return decay ** (t.max() - t)
