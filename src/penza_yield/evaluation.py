"""Единая сводка экспериментов по правилам D8.

Все эксперименты сохраняют predictions.csv со столбцами model, crop, year, period ('tune' | 'eval'), y, y_pred.
Здесь они сравниваются с эталонами E01, «эталоном, выбранным на 2011–2018» и якорем ma3 (D10).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from penza_yield import data, metrics, validation
from penza_yield.features import build_features

ROOT = Path(__file__).resolve().parents[2]
E01 = ROOT / "experiments" / "E01-baselines" / "predictions.csv"
COLS = ["model", "crop", "year", "period", "y", "y_pred"]
REFERENCES = ("baseline_sel", "anchor_ma3", "naive", "trend", "ridge")


def anchor_predictions() -> pd.DataFrame:
    """Якорь ma3 — среднее урожайности трёх предыдущих лет — как самостоятельный прогноз."""
    annual, monthly = data.load_annual(), data.load_monthly(check=False)
    rows = []
    for t in data.TARGETS:
        f = build_features(annual, monthly, t)
        for y in (*validation.TUNE_YEARS, *validation.EVAL_YEARS):
            rows.append({"model": "anchor_ma3", "crop": t, "year": y,
                         "period": "eval" if y in validation.EVAL_YEARS else "tune",
                         "y": f.loc[y, "y"], "y_pred": f.loc[y, "ma3"]})
    return pd.DataFrame(rows)


def reference_predictions() -> pd.DataFrame:
    """Эталоны E01 + эталон, выбранный на годах подбора (для каждой культуры) + якорь ma3."""
    base = pd.read_csv(E01)[COLS]
    tune_mape = (base[base.period == "tune"].assign(ape=lambda d: metrics.ape(d.y, d.y_pred))
                 .groupby(["crop", "model"]).ape.mean().reset_index())
    sel = tune_mape.sort_values("ape").groupby("crop").first()["model"].to_dict()
    chosen = base[[sel[c] == m for c, m in zip(base.crop, base.model)]].assign(model="baseline_sel")
    return pd.concat([base, chosen, anchor_predictions()], ignore_index=True)


def _row(model, crop, period, g):
    lo, hi = metrics.bootstrap_mape_ci({y: gg.ape.to_numpy() for y, gg in g.groupby("year")})
    return {"model": model, "crop": crop, "period": period, "mape": g.ape.mean(), "median_ape": g.ape.median(),
            "max_ape": g.ape.max(), "ci95_lo": lo, "ci95_hi": hi, "n": len(g)}


def summarize(pred: pd.DataFrame) -> pd.DataFrame:
    """MAPE, медиана, максимум APE и бутстреп-интервал (переотбор лет) — объединённо и по культурам."""
    pred = pred.assign(ape=metrics.ape(pred.y, pred.y_pred))
    rows = []
    for (model, period), g in pred.groupby(["model", "period"]):
        rows.append(_row(model, "pooled", period, g))
        rows += [_row(model, crop, period, gc) for crop, gc in g.groupby("crop")]
    return pd.DataFrame(rows)


def compare(pred: pd.DataFrame, models, references=REFERENCES, period: str = "eval") -> pd.DataFrame:
    """Парный тест DM по объединённой выборке (5 культур × годы периода) — каждая модель против каждого эталона."""
    p = pred[pred.period == period].assign(ape=lambda d: metrics.ape(d.y, d.y_pred))
    wide = p.pivot_table(index=["crop", "year"], columns="model", values="ape")
    rows = []
    for m in models:
        for r in references:
            if r == m or r not in wide or m not in wide:
                continue
            ok = wide[[m, r]].dropna()
            res = metrics.paired_dm(ok[m], ok[r])
            rows.append({"model": m, "vs": r, "period": period, "n": len(ok), "mape_model": ok[m].mean(),
                         "mape_vs": ok[r].mean(), "dm_stat": res.statistic, "p_value": res.p_value})
    return pd.DataFrame(rows)


def report(pred: pd.DataFrame, out_dir: Path, title: str) -> None:
    """Сохраняет results.csv и comparison.csv эксперимента и печатает сводку."""
    models = sorted(pred.model.unique())
    allp = pd.concat([pred[COLS], reference_predictions()], ignore_index=True)
    summary = summarize(allp)
    summary.to_csv(out_dir / "results.csv", index=False)
    comp = pd.concat([compare(allp, models, period="eval"), compare(allp, models, period="tune")])
    comp.to_csv(out_dir / "comparison.csv", index=False)

    short = {k: v.split()[0] for k, v in data.CROP_NAMES_RU.items()} | {"pooled": "ВСЕ"}
    order = [*models, "baseline_sel", "anchor_ma3", "naive", "trend", "ridge"]
    for period, name in (("tune", "годы подбора 2011–2018"), ("eval", "годы оценки 2019–2025")):
        tab = summary[summary.period == period].pivot(index="model", columns="crop", values="mape")
        tab = tab.reindex(index=order, columns=[*data.TARGETS, "pooled"]).rename(columns=short)
        print(f"\n{title} — MAPE, %, {name}:")
        print(tab.round(2).to_string())
    ev = comp[(comp.period == "eval") & comp.vs.isin(["baseline_sel", "anchor_ma3"])]
    print("\nОбъединённый тест DM, годы оценки (35 прогнозов):")
    print(ev[["model", "vs", "mape_model", "mape_vs", "dm_stat", "p_value"]].round(3).to_string(index=False))
    tune_anchor = summary.query("model == 'anchor_ma3' and crop == 'pooled' and period == 'tune'").mape.iloc[0]
    for m in models:
        tm = summary.query("model == @m and crop == 'pooled' and period == 'tune'").mape.iloc[0]
        verdict = "лучше" if tm < tune_anchor else "НЕ лучше"
        print(f"Критерий D10, {m}: на годах подбора {tm:.2f}% против якоря {tune_anchor:.2f}% — {verdict} якоря")


def period_of(year: int) -> str:
    return "eval" if year in validation.EVAL_YEARS else "tune"
