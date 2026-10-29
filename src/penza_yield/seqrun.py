"""Общий порядок запуска моделей сезона по вложенной схеме (E03, E04).

1) для каждого месяца отсечки — подбор конфигурации по объединённому MAPE пяти культур на 2011–2018;
2) прогнозы выбранной конфигурацией на 2011–2025 ансамблем из других seed.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from penza_yield import data, metrics, validation
from penza_yield.evaluation import period_of
from penza_yield.seqmodel import GRUConfig, SeqDataset, fit_predict


def _task(ds: SeqDataset, year: int, cutoff: int, cfg: GRUConfig, seeds, threads: int):
    import torch

    torch.set_num_threads(threads)
    idx = np.flatnonzero(ds.years == year)
    train = np.flatnonzero(ds.years < year)
    return year, cutoff, cfg.key(), fit_predict(ds, train, idx, cutoff, cfg, seeds)[0]


def run(ds: SeqDataset, name: str, grid: list[GRUConfig], cutoffs, seeds_tune, seeds_final,
        n_jobs: int, threads: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Возвращает (predictions, tuning, выбранные конфигурации по отсечкам)."""
    tune, evaluate = list(validation.TUNE_YEARS), list(validation.EVAL_YEARS)
    year_pos = {int(y): i for i, y in enumerate(ds.years)}
    par = Parallel(n_jobs=n_jobs)

    t0 = time.time()
    jobs = [(y, c, cfg, seeds_tune) for c in cutoffs for cfg in grid for y in tune]
    print(f"[{name}] подбор: {len(jobs)} задач × {len(seeds_tune)} seed…", flush=True)
    rows = []
    for year, c, key, pred in par(delayed(_task)(ds, *j, threads) for j in jobs):
        i = year_pos[year]
        rows += [{"cutoff": c, "config": key, "ape": metrics.ape(ds.y[i, k], pred[k])} for k in range(len(data.TARGETS))]
    tuning = (pd.DataFrame(rows).groupby(["cutoff", "config"], as_index=False).ape.mean()
              .rename(columns={"ape": "tune_mape_pooled"}).sort_values(["cutoff", "tune_mape_pooled"]))
    best = {int(c): k for c, k in tuning.groupby("cutoff").first()["config"].items()}
    by_key = {cfg.key(): cfg for cfg in grid}
    print(f"[{name}] подбор за {time.time() - t0:.0f} с; выбрано: {best}", flush=True)

    jobs = [(y, c, by_key[best[c]], seeds_final) for c in cutoffs for y in tune + evaluate]
    print(f"[{name}] итоговые прогнозы: {len(jobs)} задач × {len(seeds_final)} seed…", flush=True)
    rows = []
    for year, c, key, pred in par(delayed(_task)(ds, *j, threads) for j in jobs):
        i = year_pos[year]
        rows += [{"model": f"{name}_m{c:02d}", "cutoff": c, "config": key, "crop": t, "year": year,
                  "period": period_of(year), "y": ds.y[i, k], "y_pred": pred[k]}
                 for k, t in enumerate(data.TARGETS)]
    print(f"[{name}] всего {time.time() - t0:.0f} с", flush=True)
    return pd.DataFrame(rows).sort_values(["cutoff", "crop", "year"]), tuning, best
