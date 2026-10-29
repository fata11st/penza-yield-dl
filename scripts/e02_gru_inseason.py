"""E02 — многозадачная GRU на помесячных данных, внутрисезонный прогноз.

Для каждого месяца отсечки (июнь…сентябрь) отдельно:
  1) подбор конфигурации по объединённому MAPE пяти культур на одношаговых прогнозах 2011–2018 (ансамбль 10 seed);
  2) прогнозы выбранной конфигурацией на 2011–2025 (ансамбль 20 seed); годы 2019–2025 в подборе не участвуют.
Сравнение с эталонами E01 — по объединённой выборке 5 культур × 7 лет (решение D8).

Запуск (Ubuntu penza-dl, корень проекта, лучше в tmux):  python scripts/e02_gru_inseason.py
Пробный быстрый прогон:                                  python scripts/e02_gru_inseason.py --quick
Расчёт на CPU (модели крошечные, GPU не нужен). Результаты: experiments/E02-gru-inseason/
"""
from penza_yield import resources

LIMITS = resources.apply_limits()

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from penza_yield import data, metrics, validation  # noqa: E402
from penza_yield.meta import collect  # noqa: E402
from penza_yield.seqmodel import GRUConfig, build_dataset, fit_predict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CUTOFFS = {6: "июнь", 7: "июль", 8: "август", 9: "сентябрь"}
GRID = {"hidden": [8, 16], "weight_decay": [1e-3, 1e-2, 1e-1], "epochs": [100, 300]}


def task(ds, year, cutoff, cfg, seeds, threads):
    import torch

    torch.set_num_threads(threads)
    idx = np.flatnonzero(ds.years == year)
    train = np.flatnonzero(ds.years < year)
    return year, cutoff, cfg.key(), fit_predict(ds, train, idx, cutoff, cfg, seeds)[0]


def run(ds, jobs, n_jobs, threads):
    out = Parallel(n_jobs=n_jobs, verbose=0)(delayed(task)(ds, *j, threads) for j in jobs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="пробный прогон: 2 seed, 1 конфигурация, 2 отсечки")
    args = ap.parse_args()

    out_dir = ROOT / "experiments" / ("E02-gru-inseason" + ("-quick" if args.quick else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    n_jobs, threads = int(LIMITS["cpu"]["n_jobs"]), int(LIMITS["cpu"]["threads_per_process"])

    ds = build_dataset(data.load_annual(), data.load_monthly(check=False))
    tune, evaluate = list(validation.TUNE_YEARS), list(validation.EVAL_YEARS)
    grid = [GRUConfig(hidden=h, weight_decay=wd, epochs=ep)
            for h, wd, ep in itertools.product(GRID["hidden"], GRID["weight_decay"], GRID["epochs"])]
    cutoffs = dict(CUTOFFS)
    seeds_tune, seeds_final = list(range(10)), list(range(100, 120))
    if args.quick:
        grid, cutoffs = grid[:1], {6: "июнь", 9: "сентябрь"}
        seeds_tune, seeds_final = [0, 1], [100, 101]

    t0 = time.time()
    # 1. Подбор на 2011–2018
    jobs = [(y, c, cfg, seeds_tune) for c in cutoffs for cfg in grid for y in tune]
    print(f"Подбор: {len(jobs)} задач × {len(seeds_tune)} seed, {n_jobs} процессов…", flush=True)
    res = run(ds, jobs, n_jobs, threads)
    rows = []
    for year, c, key, pred in res:
        i = list(ds.years).index(year)
        rows += [{"cutoff": c, "config": key, "year": year, "crop": t, "ape": metrics.ape(ds.y[i, k], pred[k])}
                 for k, t in enumerate(data.TARGETS)]
    tuning = pd.DataFrame(rows).groupby(["cutoff", "config"], as_index=False).ape.mean()
    tuning = tuning.rename(columns={"ape": "tune_mape_pooled"}).sort_values(["cutoff", "tune_mape_pooled"])
    tuning.to_csv(out_dir / "tuning.csv", index=False)
    best = tuning.groupby("cutoff").first()["config"].to_dict()
    cfg_by_key = {cfg.key(): cfg for cfg in grid}
    print(f"  готово за {time.time() - t0:.0f} с; выбрано: {best}", flush=True)

    # 2. Прогнозы выбранных конфигураций на 2011–2025
    jobs = [(y, c, cfg_by_key[best[c]], seeds_final) for c in cutoffs for y in tune + evaluate]
    print(f"Итоговые прогнозы: {len(jobs)} задач × {len(seeds_final)} seed…", flush=True)
    res = run(ds, jobs, n_jobs, threads)
    rows = []
    for year, c, key, pred in res:
        i = list(ds.years).index(year)
        rows += [{"model": f"gru_m{c:02d}", "cutoff": c, "config": key, "crop": t, "year": year,
                  "period": "eval" if year in evaluate else "tune", "y": ds.y[i, k], "y_pred": pred[k]}
                 for k, t in enumerate(data.TARGETS)]
    pred = pd.DataFrame(rows).sort_values(["cutoff", "crop", "year"])
    pred["ape"] = metrics.ape(pred.y, pred.y_pred)
    pred.to_csv(out_dir / "predictions.csv", index=False)

    # 3. Эталоны E01 и выбранный по годам подбора эталон для каждой культуры
    base = pd.read_csv(ROOT / "experiments" / "E01-baselines" / "predictions.csv")
    base["ape"] = metrics.ape(base.y, base.y_pred)
    sel = (base[base.period == "tune"].groupby(["crop", "model"]).ape.mean().reset_index()
           .sort_values("ape").groupby("crop").first()["model"].to_dict())
    chosen = base[base.apply(lambda r: sel[r.crop] == r.model, axis=1)].assign(model="baseline_sel")
    base = pd.concat([base, chosen])

    # 4. Сводка: по культурам и объединённо, с интервалами (переотбор лет)
    allp = pd.concat([pred[["model", "crop", "year", "period", "y", "y_pred", "ape"]],
                      base[["model", "crop", "year", "period", "y", "y_pred", "ape"]]])
    summary = []
    for (model, period), g in allp.groupby(["model", "period"]):
        lo, hi = metrics.bootstrap_mape_ci({y: gg.ape.to_numpy() for y, gg in g.groupby("year")})
        summary.append({"model": model, "crop": "pooled", "period": period, "mape": g.ape.mean(),
                        "median_ape": g.ape.median(), "max_ape": g.ape.max(), "ci95_lo": lo, "ci95_hi": hi, "n": len(g)})
        for crop, gc in g.groupby("crop"):
            lo, hi = metrics.bootstrap_mape_ci({y: gg.ape.to_numpy() for y, gg in gc.groupby("year")})
            summary.append({"model": model, "crop": crop, "period": period, "mape": gc.ape.mean(),
                            "median_ape": gc.ape.median(), "max_ape": gc.ape.max(), "ci95_lo": lo, "ci95_hi": hi,
                            "n": len(gc)})
    summary = pd.DataFrame(summary)
    summary.to_csv(out_dir / "results.csv", index=False)

    # 5. Объединённый тест Диболда–Мариано на годах оценки: GRU против каждого эталона
    ev = allp[allp.period == "eval"].pivot_table(index=["crop", "year"], columns="model", values="ape")
    comp = []
    for c in cutoffs:
        m = f"gru_m{c:02d}"
        for b in ("naive", "trend", "ridge", "baseline_sel"):
            r = metrics.paired_dm(ev[m], ev[b])
            comp.append({"model": m, "vs": b, "n": len(ev), "mape_model": ev[m].mean(), "mape_vs": ev[b].mean(),
                         "dm_stat": r.statistic, "p_value": r.p_value})
    comp = pd.DataFrame(comp)
    comp.to_csv(out_dir / "comparison.csv", index=False)

    # 6. Рисунок: объединённый MAPE от месяца отсечки против эталонов
    pooled = summary[(summary.crop == "pooled") & (summary.period == "eval")].set_index("model")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    xs = list(cutoffs)
    gm = pooled.loc[[f"gru_m{c:02d}" for c in xs]]
    ax.errorbar(xs, gm.mape, yerr=[gm.mape - gm.ci95_lo, gm.ci95_hi - gm.mape], marker="o", capsize=4,
                color="#1f4e79", label="GRU (многозадачная)")
    styles = {"naive": ":", "trend": "--", "ridge": "-.", "baseline_sel": "-"}
    names = {"naive": "наивный", "trend": "тренд", "ridge": "Ridge", "baseline_sel": "эталон, выбранный на 2011–2018"}
    for b, ls in styles.items():
        ax.axhline(pooled.loc[b, "mape"], ls=ls, color="#7f7f7f", lw=1.2, label=names[b])
    ax.set_xticks(xs, [CUTOFFS[c] for c in xs])
    ax.set_xlabel("Данные по состоянию на конец месяца")
    ax.set_ylabel("MAPE, % (5 культур, 2019–2025)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_mape_by_cutoff.png", dpi=200)

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(collect(experiment=out_dir.name, tune_years=tune, eval_years=evaluate, grid=GRID,
                          selected=best, seeds_tune=seeds_tune, seeds_final=seeds_final, cutoffs=list(cutoffs),
                          baseline_selected=sel, limits=LIMITS, seconds=round(time.time() - t0)),
                  f, ensure_ascii=False, indent=2)

    # Печать
    print(f"\nГотово за {time.time() - t0:.0f} с. Файлы: {out_dir}\n")
    tab = summary[summary.period == "eval"].pivot(index="model", columns="crop", values="mape")
    order = [f"gru_m{c:02d}" for c in cutoffs] + ["naive", "trend", "ridge", "baseline_sel"]
    tab = tab.loc[order, [*data.TARGETS, "pooled"]].rename(columns={**{k: v.split()[0] for k, v in
                                                                         data.CROP_NAMES_RU.items()},
                                                                      "pooled": "ВСЕ"})
    print("MAPE, %, годы оценки 2019–2025:")
    print(tab.round(2).to_string())
    print("\nОбъединённый тест Диболда–Мариано (35 прогнозов), GRU против эталонов:")
    print(comp[["model", "vs", "mape_model", "mape_vs", "dm_stat", "p_value"]].round(3).to_string(index=False))
    print(f"\nЭталон, выбранный на 2011–2018: {sel}")


if __name__ == "__main__":
    main()
