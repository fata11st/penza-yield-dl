"""E03 (компактная GRU) и E04 (компактная 1D-CNN): внутрисезонный прогноз на 4 агрономических переменных.

Дизайн и критерий успеха зафиксированы в D10 до запуска: вход — температура, осадки, влажность почвы 7–28 см, NDVI;
сетка смещена к сильной регуляризации; модель полезна, только если на годах подбора лучше своего якоря ma3.

Запуск:  python scripts/e03_e04_compact.py --arch gru     (E03)
         python scripts/e03_e04_compact.py --arch cnn     (E04)
         добавьте --quick для пробного прогона
"""
from penza_yield import resources

LIMITS = resources.apply_limits()

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from penza_yield import data, evaluation, validation  # noqa: E402
from penza_yield.meta import collect  # noqa: E402
from penza_yield.seqmodel import COMPACT_VARIABLES, GRUConfig, build_dataset  # noqa: E402
from penza_yield.seqrun import run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {"gru": ("E03-gru-compact", "gru_c"), "cnn": ("E04-cnn-compact", "cnn_c")}
GRID = {"hidden": [4, 8], "weight_decay": [1e-2, 1e-1, 1.0], "epochs": [30, 100]}
CUTOFFS = [6, 7, 8, 9]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(EXPERIMENTS), required=True)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    exp, name = EXPERIMENTS[args.arch]
    out = ROOT / "experiments" / (exp + ("-quick" if args.quick else ""))
    out.mkdir(parents=True, exist_ok=True)

    grid = [GRUConfig(hidden=h, weight_decay=wd, epochs=ep, arch=args.arch, variables=COMPACT_VARIABLES)
            for h, wd, ep in itertools.product(GRID["hidden"], GRID["weight_decay"], GRID["epochs"])]
    cutoffs, seeds_tune, seeds_final = CUTOFFS, list(range(10)), list(range(100, 120))
    if args.quick:
        grid, cutoffs, seeds_tune, seeds_final = grid[:2], [8], [0], [100]

    ds = build_dataset(data.load_annual(), data.load_monthly(check=False))
    pred, tuning, best = run(ds, name, grid, cutoffs, seeds_tune, seeds_final,
                             int(LIMITS["cpu"]["n_jobs"]), int(LIMITS["cpu"]["threads_per_process"]))
    pred.to_csv(out / "predictions.csv", index=False)
    tuning.to_csv(out / "tuning.csv", index=False)
    evaluation.report(pred, out, exp)

    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(collect(experiment=out.name, arch=args.arch, variables=list(COMPACT_VARIABLES), grid=GRID,
                          cutoffs=cutoffs, selected=best, seeds_tune=seeds_tune, seeds_final=seeds_final,
                          tune_years=list(validation.TUNE_YEARS), eval_years=list(validation.EVAL_YEARS),
                          limits=LIMITS), f, ensure_ascii=False, indent=2)
    print(f"\nФайлы: {out}")


if __name__ == "__main__":
    main()
