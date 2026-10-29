"""E05 — многослойный перцептрон на годовых признаках (набор ar_climate), отдельно по культурам.

Сетка: скрытый слой {4, 16} × L2 {0.1, 1, 10}; одна конфигурация на все культуры — по объединённому MAPE 2011–2018
(10 seed), итог — ансамбль 20 seed на 2011–2025. Дизайн — D10.
Запуск: python scripts/e05_mlp.py   (--quick — пробный прогон)
"""
from penza_yield import resources

LIMITS = resources.apply_limits()

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from penza_yield import data, evaluation, metrics, validation  # noqa: E402
from penza_yield.meta import collect  # noqa: E402
from penza_yield.tabular import FEATURE_SETS, crop_frames, fit_predict_mlp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEATURES = FEATURE_SETS["ar_climate"]
GRID = {"hidden": [4, 16], "alpha": [0.1, 1.0, 10.0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = ROOT / "experiments" / ("E05-mlp" + ("-quick" if args.quick else ""))
    out.mkdir(parents=True, exist_ok=True)

    frames = crop_frames()
    tune, evaluate = list(validation.TUNE_YEARS), list(validation.EVAL_YEARS)
    grid = list(itertools.product(GRID["hidden"], GRID["alpha"]))
    seeds_tune, seeds_final = list(range(10)), list(range(100, 120))
    if args.quick:
        grid, seeds_tune, seeds_final = grid[:1], [0], [100]
    par = Parallel(n_jobs=int(LIMITS["cpu"]["n_jobs"]))
    t0 = time.time()

    def job(crop, year, h, a, seeds):
        return crop, year, (h, a), fit_predict_mlp(frames[crop], year, FEATURES, h, a, seeds)

    res = par(delayed(job)(c, y, h, a, seeds_tune) for (h, a) in grid for c in data.TARGETS for y in tune)
    tuning = pd.DataFrame([{"config": f"h={cfg[0]}|alpha={cfg[1]}", "cfg": cfg,
                            "ape": metrics.ape(frames[c].loc[y, "y"], p)} for c, y, cfg, p in res])
    tuning = tuning.groupby(["config"], as_index=False).agg(tune_mape_pooled=("ape", "mean"), cfg=("cfg", "first"))
    tuning = tuning.sort_values("tune_mape_pooled")
    best = tuning.iloc[0]
    print(f"[mlp] подбор за {time.time() - t0:.0f} с; выбрано: {best.config}", flush=True)

    h, a = best.cfg
    res = par(delayed(job)(c, y, h, a, seeds_final) for c in data.TARGETS for y in tune + evaluate)
    pred = pd.DataFrame([{"model": "mlp", "config": best.config, "crop": c, "year": y,
                          "period": evaluation.period_of(y), "y": frames[c].loc[y, "y"], "y_pred": p}
                         for c, y, _, p in res])
    pred.to_csv(out / "predictions.csv", index=False)
    tuning.drop(columns="cfg").to_csv(out / "tuning.csv", index=False)
    evaluation.report(pred, out, "E05-mlp")
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(collect(experiment=out.name, features=FEATURES, grid=GRID, selected=best.config,
                          seeds_tune=seeds_tune, seeds_final=seeds_final, limits=LIMITS,
                          seconds=round(time.time() - t0)), f, ensure_ascii=False, indent=2)
    print(f"\nФайлы: {out}")


if __name__ == "__main__":
    main()
