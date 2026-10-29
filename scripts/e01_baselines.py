"""E01 — эталонные модели по вложенной схеме.

Модели: наивный прогноз (урожайность прошлого года), линейный тренд, Ridge (сетка 3 набора признаков × 4 α × 3 λ).
Ridge подбирается по MAPE одношаговых прогнозов 2011–2018; выбранная конфигурация оценивается на 2019–2025,
которые в подборе не участвуют. Окно обучения расширяется: на каждый год — все предшествующие годы.

Запуск (Ubuntu penza-dl, корень проекта):  python scripts/e01_baselines.py
Результаты: experiments/E01-baselines/
"""
from penza_yield import resources

resources.apply_limits()

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from penza_yield import baselines, data, metrics, validation  # noqa: E402
from penza_yield.features import FEATURE_SETS, build_features  # noqa: E402
from penza_yield.meta import collect  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "experiments" / "E01-baselines"
MIN_TRAIN = 8  # минимум лет в окне обучения (для набора с NDVI окно начинается с 2000 г.)


def one_step(frame: pd.DataFrame, years, predict, features: list[str]) -> list[dict]:
    """Одношаговые прогнозы по расширяющемуся окну: для года t обучение на всех годах < t."""
    rows = []
    for t in years:
        train = baselines.usable_rows(frame[frame.index < t], features)
        test = frame.loc[[t]]
        if len(train) < MIN_TRAIN:
            raise ValueError(f"{t}: в окне обучения {len(train)} лет < {MIN_TRAIN}")
        rows.append({"year": t, "y": float(test["y"].iloc[0]), "y_pred": predict(train, test)})
    return rows


def scores(df: pd.DataFrame) -> dict:
    return {"mape": metrics.mape(df.y, df.y_pred), "rmse": metrics.rmse(df.y, df.y_pred),
            "mae": metrics.mae(df.y, df.y_pred), "n": len(df)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    annual, monthly = data.load_annual(), data.load_monthly(check=False)
    tune, evaluate = list(validation.TUNE_YEARS), list(validation.EVAL_YEARS)

    preds, selection = [], []
    for target in data.TARGETS:
        frame = build_features(annual, monthly, target)

        for name, fn, feats in [("naive", baselines.predict_naive, ["lag1"]),
                                ("trend", baselines.predict_trend, [])]:
            for r in one_step(frame, tune + evaluate, fn, feats):
                preds.append({"crop": target, "model": name, "config": "", **r})

        # Подбор Ridge только на годах подбора
        grid = []
        for cfg in baselines.ridge_configs(FEATURE_SETS):
            feats = FEATURE_SETS[cfg["features"]]
            fn = lambda tr, te, f=feats, c=cfg: baselines.predict_ridge(tr, te, f, c["alpha"], c["decay"])  # noqa: E731
            rows = one_step(frame, tune, fn, feats)
            grid.append((metrics.mape([r["y"] for r in rows], [r["y_pred"] for r in rows]), cfg, fn, feats))
        grid.sort(key=lambda g: g[0])
        best_mape, best_cfg, best_fn, best_feats = grid[0]
        selection.append({"crop": target, "config": baselines.config_key(best_cfg),
                          "tune_mape": best_mape, "n_configs": len(grid)})

        # Выбранная конфигурация: прогнозы на годы подбора (для справки) и на годы оценки
        for r in one_step(frame, tune + evaluate, best_fn, best_feats):
            preds.append({"crop": target, "model": "ridge", "config": baselines.config_key(best_cfg), **r})

    p = pd.DataFrame(preds)
    p["period"] = p.year.map(lambda y: "eval" if y in evaluate else "tune")
    p.to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame(selection).to_csv(OUT / "ridge_selection.csv", index=False)

    res = [{"crop": c, "model": m, "period": per, **scores(g)} for (c, m, per), g in p.groupby(["crop", "model", "period"])]
    res = pd.DataFrame(res)
    res.to_csv(OUT / "results.csv", index=False)

    # Тест Диболда–Мариано на годах оценки: Ridge против наивного и тренда (n = 7 — мощность теста мала)
    dm = []
    ev = p[p.period == "eval"]
    for c in data.TARGETS:
        g = ev[ev.crop == c].pivot(index="year", columns="model", values="y_pred")
        y = ev[(ev.crop == c) & (ev.model == "ridge")].set_index("year").y.loc[g.index]
        for other in ("naive", "trend"):
            r = metrics.diebold_mariano(y, g["ridge"], g[other])
            dm.append({"crop": c, "vs": other, "dm_stat": r.statistic, "p_value": r.p_value,
                       "mean_ape_diff": r.mean_loss_diff})
    pd.DataFrame(dm).to_csv(OUT / "dm_tests.csv", index=False)

    with open(OUT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(collect(experiment="E01-baselines", tune_years=tune, eval_years=evaluate,
                          ridge_grid=baselines.RIDGE_GRID, feature_sets=FEATURE_SETS, min_train=MIN_TRAIN),
                  f, ensure_ascii=False, indent=2)

    # Сводка: MAPE на годах оценки
    table = res[res.period == "eval"].pivot(index="crop", columns="model", values="mape").loc[data.TARGETS]
    table.index = [data.CROP_NAMES_RU[c] for c in table.index]
    print("MAPE, %, годы оценки 2019–2025 (подбор Ridge — только 2011–2018):")
    print(table[["naive", "trend", "ridge"]].round(2).to_string())
    print("\nВыбранные конфигурации Ridge:")
    print(pd.DataFrame(selection)[["crop", "config", "tune_mape"]].round(2).to_string(index=False))
    print(f"\nФайлы: {OUT}")


if __name__ == "__main__":
    main()
