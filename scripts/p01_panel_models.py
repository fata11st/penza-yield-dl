"""P01 — модели на панели 10 регионов (D11): подбор на 2011–2018, прогнозы 2006–2025.

Запуск:  python scripts/p01_panel_models.py --model {baselines,ridge,gbm,mlp,gru,chronos}  [--quick]
Результаты: experiments/P01-<model>/predictions.csv (+ tuning.csv, meta.json)
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

from penza_yield import metrics, panel_models as pm, validation  # noqa: E402
from penza_yield.meta import collect  # noqa: E402
from penza_yield.panel import load_panel, period_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
YEARS = list(range(2006, 2026))
TUNE = list(validation.TUNE_YEARS)
GRIDS = {
    "ridge": [{"features": f, "alpha": a} for f, a in itertools.product(["lags", "lags_eng"], [0.1, 1, 10, 100, 1000])],
    "gbm": [{"num_leaves": nl, "n_estimators": n} for nl, n in itertools.product([4, 8], [100, 300])],
    "mlp": [{"hidden": h, "wd": wd, "epochs": ep} for h, wd, ep in itertools.product([16, 64], [1e-3, 1e-2, 1e-1],
                                                                                      [200, 500])],
    "gru": [{"hidden": h, "wd": wd, "epochs": ep} for h, wd, ep in itertools.product([8, 16, 32], [1e-3, 1e-2, 1e-1],
                                                                                      [100, 300])],
}
FUNCS = {"ridge": pm.predict_ridge, "gbm": pm.predict_gbm, "mlp": pm.predict_mlp, "gru": pm.predict_gru}
SEEDS = {"ridge": ([0], [0]), "gbm": (list(range(5)), list(range(5))), "mlp": (list(range(10)), list(range(100, 120))),
         "gru": (list(range(10)), list(range(100, 120)))}


def key(cfg: dict) -> str:
    return "|".join(f"{k}={v}" for k, v in cfg.items())


def attach_truth(P, pred: pd.DataFrame, t: int) -> pd.DataFrame:
    T = P.targets[(P.targets.year == t)][["region", "crop", "y"]]
    out = pred.merge(T, on=["region", "crop"], how="left")
    out["year"], out["period"] = t, period_of(t)
    return out


def task(P, model, cfg, t, seeds, threads):
    import torch

    torch.set_num_threads(threads)
    return t, key(cfg), attach_truth(P, FUNCS[model](P, t, cfg, seeds), t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["baselines", *FUNCS, "chronos"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = ROOT / "experiments" / (f"P01-{args.model}" + ("-quick" if args.quick else ""))
    out.mkdir(parents=True, exist_ok=True)
    P = load_panel()
    years, tune = YEARS, TUNE
    t0 = time.time()
    info = {}

    if args.model == "baselines":
        if args.quick:
            years = [2015, 2020]
        pred = pd.concat([attach_truth(P, pm.predict_baselines(P, t), t) for t in years], ignore_index=True)

    elif args.model == "chronos":
        import torch
        from chronos import Chronos2Pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
        if args.quick:
            years = [2015, 2020]
        parts, fallbacks = [], {}
        for name, cov in (("chronos_uni_p", False), ("chronos_cov_p", True)):
            n_fb = 0
            for t in years:
                p, nf = pm.predict_chronos(P, t, pipe, cov, name)
                parts.append(attach_truth(P, p, t))
                n_fb += nf
            fallbacks[name] = n_fb
            print(f"[chronos] {name}: готово ({time.time() - t0:.0f} с), прогнозов якорем: {n_fb}", flush=True)
        pred = pd.concat(parts, ignore_index=True)
        info = {"device": device, "fallback_to_anchor": fallbacks}

    else:
        grid = GRIDS[args.model]
        seeds_tune, seeds_final = SEEDS[args.model]
        if args.quick:
            grid, tune, years = grid[:2], [2015, 2016], [2015, 2020]
            seeds_tune, seeds_final = seeds_tune[:1], seeds_final[:1]
        n_jobs, threads = int(LIMITS["cpu"]["n_jobs"]), int(LIMITS["cpu"]["threads_per_process"])
        par = Parallel(n_jobs=n_jobs)
        print(f"[{args.model}] подбор: {len(grid)} конфигураций × {len(tune)} лет…", flush=True)
        res = par(delayed(task)(P, args.model, cfg, t, seeds_tune, threads) for cfg in grid for t in tune)
        tuning = (pd.concat([r.assign(config=k) for _, k, r in res])
                  .assign(ape=lambda d: metrics.ape(d.y, d.y_pred))
                  .groupby("config", as_index=False).ape.mean().rename(columns={"ape": "tune_mape_pooled"})
                  .sort_values("tune_mape_pooled"))
        tuning.to_csv(out / "tuning.csv", index=False)
        best = next(c for c in grid if key(c) == tuning.config.iloc[0])
        print(f"[{args.model}] подбор за {time.time() - t0:.0f} с; выбрано: {key(best)} "
              f"(MAPE подбора {tuning.tune_mape_pooled.iloc[0]:.2f}%)", flush=True)
        res = par(delayed(task)(P, args.model, best, t, seeds_final, threads) for t in years)
        pred = pd.concat([r.assign(config=k) for _, k, r in res], ignore_index=True)
        info = {"grid": grid, "selected": key(best), "seeds_tune": seeds_tune, "seeds_final": seeds_final}

    pred = pred.sort_values(["model", "region", "crop", "year"])
    pred.to_csv(out / "predictions.csv", index=False)
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(collect(experiment=out.name, years=years, tune_years=tune, limits=LIMITS,
                          seconds=round(time.time() - t0), **info), f, ensure_ascii=False, indent=2, default=str)
    ev = pred[pred.period == "eval"].assign(ape=lambda d: metrics.ape(d.y, d.y_pred))
    print(f"[{args.model}] готово за {time.time() - t0:.0f} с: {len(pred)} прогнозов → {out}")
    if not ev.empty:
        print(ev.groupby("model").ape.agg(["count"]).to_string())


if __name__ == "__main__":
    main()
