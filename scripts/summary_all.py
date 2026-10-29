"""Сводка всех экспериментов E01–E07 по правилам D8/D10: одна таблица и один рисунок.

Собирает experiments/E0*/predictions.csv (кроме пробных -quick), сравнивает с эталонами и якорем ma3.
Запуск: python scripts/summary_all.py → experiments/SUMMARY/
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from penza_yield import data, evaluation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "SUMMARY"
REF_ORDER = ["baseline_sel", "anchor_ma3", "naive", "trend", "ridge"]
NAMES = {"baseline_sel": "эталон, выбранный на 2011–2018", "anchor_ma3": "якорь: среднее 3 прошлых лет",
         "naive": "наивный", "trend": "тренд", "ridge": "Ridge"}
# Модели без подбора (zero-shot / без гиперпараметров): их MAPE на 2011–2018 — честная оценка, а не подгонка
NO_TUNING = ("tabpfn", "chronos2")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preds, origin = [], {}
    for d in sorted((ROOT / "experiments").glob("E0*")):
        if d.name.endswith("-quick") or d.name.startswith("E01") or not (d / "predictions.csv").exists():
            continue
        p = pd.read_csv(d / "predictions.csv")[evaluation.COLS]
        preds.append(p)
        origin |= {m: d.name for m in p.model.unique()}
    models = sorted(origin)
    allp = pd.concat([*preds, evaluation.reference_predictions()], ignore_index=True)
    origin |= {m: "E01-baselines" for m in REF_ORDER}

    s = evaluation.summarize(allp)
    comp = evaluation.compare(allp, [*models, *[r for r in REF_ORDER if r != "baseline_sel"]],
                              references=("baseline_sel", "anchor_ma3"), period="eval")
    pooled = s[s.crop == "pooled"].pivot(index="model", columns="period", values="mape")
    ev = s[(s.crop == "pooled") & (s.period == "eval")].set_index("model")
    anchor_tune = pooled.loc["anchor_ma3", "tune"]

    rows = []
    for m in [*models, *REF_ORDER]:
        r = {"model": m, "experiment": origin[m], "mape_tune": pooled.loc[m, "tune"], "mape_eval": ev.loc[m, "mape"],
             "eval_ci95_lo": ev.loc[m, "ci95_lo"], "eval_ci95_hi": ev.loc[m, "ci95_hi"],
             "eval_max_ape": ev.loc[m, "max_ape"]}
        for ref in ("baseline_sel", "anchor_ma3"):
            c = comp[(comp.model == m) & (comp.vs == ref)]
            r[f"p_vs_{ref}"] = c.p_value.iloc[0] if len(c) else float("nan")
        r["beats_anchor_on_tune"] = bool(r["mape_tune"] < anchor_tune) if m not in REF_ORDER else None
        r["tune_is_honest"] = m.startswith(NO_TUNING) or m in REF_ORDER and m not in ("ridge", "baseline_sel")
        rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "summary.csv", index=False)

    by_crop = s[(s.period == "eval")].pivot(index="model", columns="crop", values="mape")
    by_crop = by_crop.reindex(index=[*models, *REF_ORDER], columns=[*data.TARGETS, "pooled"])
    by_crop.to_csv(OUT / "mape_eval_by_crop.csv")

    # Рисунок: MAPE 2019–2025 по всем культурам с 95% интервалами
    t = table.sort_values("mape_eval", ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(t) + 1.2))
    colors = ["#7f7f7f" if m in REF_ORDER else "#1f4e79" for m in t.model]
    for i, (row, col) in enumerate(zip(t.itertuples(), colors)):
        ax.errorbar(row.mape_eval, i, xerr=[[row.mape_eval - row.eval_ci95_lo], [row.eval_ci95_hi - row.mape_eval]],
                    fmt="o", color=col, ecolor=col, capsize=3, lw=1)
    ax.axvline(table.set_index("model").loc["baseline_sel", "mape_eval"], ls="--", color="#7f7f7f", lw=1)
    ax.set_yticks(range(len(t)), [NAMES.get(m, m) for m in t.model], fontsize=8)
    ax.set_xlabel("MAPE, %, 5 культур × 2019–2025 (95% бутстреп-интервал по годам)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_summary.png", dpi=200)

    with open(OUT / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Сводка экспериментов\n\nMAPE по 35 прогнозам (5 культур × 2019–2025), правила D8/D10.\n\n")
        try:
            f.write(table.round(3).to_markdown(index=False))
        except ImportError:  # нет пакета tabulate
            f.write("```\n" + table.round(3).to_string(index=False) + "\n```")
        f.write("\n")

    pd.set_option("display.width", 200)
    print("СВОДКА E01–E07 — объединённый MAPE, %:")
    show = table[["model", "mape_tune", "mape_eval", "eval_ci95_lo", "eval_ci95_hi", "p_vs_baseline_sel",
                  "p_vs_anchor_ma3", "beats_anchor_on_tune"]].sort_values("mape_eval")
    print(show.round(3).to_string(index=False))
    print("\nMAPE по культурам, годы оценки 2019–2025:")
    print(by_crop.rename(columns={k: v.split()[0] for k, v in data.CROP_NAMES_RU.items()}).round(2).to_string())
    print(f"\nФайлы: {OUT}")


if __name__ == "__main__":
    main()
