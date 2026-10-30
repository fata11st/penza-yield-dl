"""P03 — сводка по панели (D11): основная оценка — 9 новых регионов, 2019–2025; вторичная — Пенза.

Эталон — «выбранный на 2011–2018» для каждой культуры среди наивного, тренда и якоря (по всем регионам), и якорь.
Тест DM — по всем парным прогнозам и консервативный вариант по годовым средним (n = 7), так как ошибки соседних
регионов одного года связаны погодой.
Запуск: python scripts/p03_summary.py → experiments/P03-summary/
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from penza_yield import metrics  # noqa: E402
from penza_yield.panel import CROPS, NEW_REGIONS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "P03-summary"
BASES = ["naive", "trend", "anchor"]
NAMES = {"naive": "наивный", "trend": "тренд", "anchor": "якорь (среднее 3 лет)", "baseline_sel": "эталон, выбранный на 2011–2018",
         "ridge_p": "Ridge (панель)", "gbm_p": "LightGBM (панель)", "mlp_p": "MLP (панель)", "gru_p": "GRU (панель)",
         "chronos_uni_p": "Chronos-2, ряд", "chronos_cov_p": "Chronos-2 + агропризнаки",
         "ens_mean": "ансамбль: среднее", "ens_median": "ансамбль: медиана", "ens_nn": "ансамбль нейросетей",
         "ens_invmape": "ансамбль: веса 1/MAPE", "ens_stack": "ансамбль: стекинг", "ens_select": "ансамбль: выбор лучшей"}


def load() -> pd.DataFrame:
    dirs = [d for d in sorted((ROOT / "experiments").glob("P0[12]-*")) if not d.name.endswith("-quick")]
    p = pd.concat([pd.read_csv(d / "predictions.csv")[["model", "region", "crop", "year", "period", "y", "y_pred"]]
                   for d in dirs], ignore_index=True)
    p = p[p.year >= 2011]
    tune = p[(p.period == "tune") & p.model.isin(BASES)].assign(ape=lambda d: metrics.ape(d.y, d.y_pred))
    sel = tune.groupby(["crop", "model"]).ape.mean().reset_index().sort_values("ape").groupby("crop").first()["model"]
    chosen = p[[sel[c] == m for c, m in zip(p.crop, p.model)]].assign(model="baseline_sel")
    p = pd.concat([p, chosen], ignore_index=True)
    # общий набор рядов: только те (регион, культура, год), где есть прогнозы всех моделей
    n_models = p.model.nunique()
    full = p.groupby(["region", "crop", "year"]).model.transform("nunique") == n_models
    p = p[full].assign(ape=lambda d: metrics.ape(d.y, d.y_pred))
    p["group"] = np.where(p.region == "penza", "penza", "new")
    return p, sel.to_dict()


def dm_yearly(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Консервативный DM: разности средних APE по годам (n = число лет)."""
    ya, yb = a.groupby("year").ape.mean(), b.groupby("year").ape.mean()
    ya, yb = ya.align(yb, join="inner")
    if len(ya) < 3:
        return float("nan")
    return metrics.paired_dm(ya.to_numpy(), yb.to_numpy()).p_value


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p, sel = load()
    models = list(p.model.unique())
    rows = []
    for m in models:
        r = {"model": m, "name": NAMES.get(m, m)}
        for grp, per in (("new", "tune"), ("new", "eval"), ("penza", "eval")):
            g = p[(p.model == m) & (p.group == grp) & (p.period == per)]
            r[f"mape_{grp}_{per}"] = g.ape.mean()
            r[f"n_{grp}_{per}"] = len(g)
            if (grp, per) == ("new", "eval"):
                r["ci95_lo"], r["ci95_hi"] = metrics.bootstrap_mape_ci({y: gg.ape.to_numpy() for y, gg in g.groupby("year")})
                for ref in ("baseline_sel", "anchor"):
                    if m == ref:
                        continue
                    h = p[(p.model == ref) & (p.group == grp) & (p.period == per)]
                    j = g.merge(h, on=["region", "crop", "year"], suffixes=("", "_r"))
                    r[f"p_{ref}"] = metrics.paired_dm(j.ape, j.ape_r).p_value
                    r[f"p_{ref}_yearly"] = dm_yearly(g, h)
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("mape_new_eval")
    t.to_csv(OUT / "summary.csv", index=False)

    ev = p[(p.group == "new") & (p.period == "eval")]
    by_crop = ev.pivot_table(index="model", columns="crop", values="ape").reindex(index=t.model, columns=CROPS)
    by_crop.to_csv(OUT / "mape_eval_new_by_crop.csv")
    by_region = ev.pivot_table(index="model", columns="region", values="ape").reindex(index=t.model,
                                                                                      columns=NEW_REGIONS)
    by_region.to_csv(OUT / "mape_eval_new_by_region.csv")

    # Рисунок: основная оценка с интервалами
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(t) + 1.3))
    for i, row in enumerate(t[::-1].itertuples()):
        col = "#7f7f7f" if row.model in (*BASES, "baseline_sel") else ("#b5651d" if row.model.startswith("ens")
                                                                      else "#1f4e79")
        ax.errorbar(row.mape_new_eval, i, xerr=[[row.mape_new_eval - row.ci95_lo], [row.ci95_hi - row.mape_new_eval]],
                    fmt="o", color=col, ecolor=col, capsize=3, lw=1)
    ax.axvline(t.set_index("model").loc["baseline_sel", "mape_new_eval"], ls="--", color="#7f7f7f", lw=1)
    ax.set_yticks(range(len(t)), [NAMES.get(m, m) for m in t.model[::-1]], fontsize=8)
    ax.set_xlabel("MAPE, %, 9 регионов × 5 культур × 2019–2025 (95% бутстреп по годам)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_panel_summary.png", dpi=200)

    pd.set_option("display.width", 220)
    cols = ["model", "mape_new_tune", "mape_new_eval", "ci95_lo", "ci95_hi", "p_baseline_sel", "p_baseline_sel_yearly",
            "p_anchor", "mape_penza_eval", "n_new_eval"]
    print("ПАНЕЛЬ — основная оценка: 9 новых регионов, 2019–2025; MAPE, %")
    print(t[cols].round(3).to_string(index=False))
    print("\nMAPE по культурам (новые регионы, 2019–2025):")
    print(by_crop.round(2).to_string())
    print(f"\nЭталон, выбранный на 2011–2018: {sel}\nФайлы: {OUT}")


if __name__ == "__main__":
    main()
