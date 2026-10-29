"""E06 — TabPFN (предобученная модель для малых таблиц) на годовых признаках, без подбора гиперпараметров.

Признаки — набор ar_climate_ndvi (пропуски NDVI до 2000 г. TabPFN обрабатывает сам), отдельно по культурам,
среднее по 3 seed. Подбора нет, поэтому годы 2011–2018 и 2019–2025 — просто два периода оценки. Дизайн — D10.

Нужен ключ Prior Labs: учётная запись, принятая лицензия модели и переменная TABPFN_TOKEN
(хранится в ~/.config/penza-yield/env, не в git). Без ключа скрипт завершается с кодом 3, ночной запуск идёт дальше.
Запуск: python scripts/e06_tabpfn.py   (--quick — пробный прогон)
"""
import os
import sys

if not os.environ.get("TABPFN_TOKEN"):
    print("✗ TABPFN_TOKEN не задан — E06 пропущен (см. docs/ENVIRONMENT.md, раздел «Ключи моделей»)")
    sys.exit(3)
os.environ.setdefault("TABPFN_NO_BROWSER", "1")

from penza_yield import resources  # noqa: E402

LIMITS = resources.apply_limits()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from penza_yield import data, evaluation, validation  # noqa: E402
from penza_yield.meta import collect  # noqa: E402
from penza_yield.tabular import FEATURE_SETS, crop_frames, fit_predict_tabpfn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEATURES = FEATURE_SETS["ar_climate_ndvi"]
SEEDS = [0, 1, 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = ROOT / "experiments" / ("E06-tabpfn" + ("-quick" if args.quick else ""))
    out.mkdir(parents=True, exist_ok=True)

    frames = crop_frames()
    years = [*validation.TUNE_YEARS, *validation.EVAL_YEARS]
    crops, seeds = data.TARGETS, SEEDS
    if args.quick:
        crops, years, seeds = crops[:1], years[:2], [0]
    t0 = time.time()
    rows = []
    for c in crops:
        for y in years:
            p = fit_predict_tabpfn(frames[c], y, FEATURES, seeds)
            rows.append({"model": "tabpfn", "crop": c, "year": y, "period": evaluation.period_of(y),
                         "y": frames[c].loc[y, "y"], "y_pred": p})
        print(f"[tabpfn] {c}: готово ({time.time() - t0:.0f} с)", flush=True)
    pred = pd.DataFrame(rows)
    pred.to_csv(out / "predictions.csv", index=False)
    evaluation.report(pred, out, "E06-tabpfn")
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(collect(experiment=out.name, features=FEATURES, seeds=seeds, tabpfn=metadata.version("tabpfn"),
                          limits=LIMITS, seconds=round(time.time() - t0)), f, ensure_ascii=False, indent=2)
    print(f"\nФайлы: {out}")


if __name__ == "__main__":
    main()
