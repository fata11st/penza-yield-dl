"""E07 — Chronos-2 (предобученная модель временных рядов), прогноз без дообучения (zero-shot).

Для каждого года t ряд урожайности каждой культуры за 1990…t−1 подаётся как контекст, прогноз — на один шаг.
Варианты (все заранее объявлены в D10, основной — chronos2_cov):
  chronos2_uni    — только ряд урожайности;
  chronos2_cov    — + погода сезона t как известные ковариаты (те же 6 признаков, что у эталонов, без NDVI);
  chronos2_cov_xl — то же + совместный прогноз пяти культур (cross_learning).
Точечный прогноз — медиана (квантиль 0,5). Подбора нет. Веса amazon/chronos-2 (Apache-2.0) скачиваются с Hugging Face.
Запуск: python scripts/e07_chronos.py   (--quick — пробный прогон)
"""
from penza_yield import resources

LIMITS = resources.apply_limits()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from penza_yield import data, evaluation, validation  # noqa: E402
from penza_yield.meta import collect  # noqa: E402
from penza_yield.tabular import crop_frames  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "amazon/chronos-2"
COVARIATES = ["t_summer", "p_summer", "p_veget", "t_may", "swc_spring", "pet_summer"]
VARIANTS = {"chronos2_uni": (False, False), "chronos2_cov": (True, False), "chronos2_cov_xl": (True, True)}


def frames_for_year(frames: dict, year: int, with_cov: bool):
    """Контекст — годы < year; будущее — год year (только ковариаты)."""
    ctx, fut = [], []
    for crop, f in frames.items():
        hist = f[f.index < year]
        c = pd.DataFrame({"item_id": crop, "timestamp": pd.to_datetime(hist.index.astype(str)), "target": hist.y})
        u = pd.DataFrame({"item_id": [crop], "timestamp": [pd.Timestamp(f"{year}-01-01")]})
        if with_cov:
            for v in COVARIATES:
                c[v] = hist[v].to_numpy()
                u[v] = [f.loc[year, v]]
        ctx.append(c.reset_index(drop=True))
        fut.append(u)
    return pd.concat(ctx, ignore_index=True), pd.concat(fut, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = ROOT / "experiments" / ("E07-chronos2" + ("-quick" if args.quick else ""))
    out.mkdir(parents=True, exist_ok=True)

    from chronos import Chronos2Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=device)
    frames = crop_frames()
    years = [*validation.TUNE_YEARS, *validation.EVAL_YEARS]
    variants = VARIANTS
    if args.quick:
        years, variants = years[:2], {"chronos2_cov": VARIANTS["chronos2_cov"]}

    t0 = time.time()
    rows = []
    for name, (with_cov, cross) in variants.items():
        for y in years:
            ctx, fut = frames_for_year(frames, y, with_cov)
            fc = pipe.predict_df(ctx, future_df=fut if with_cov else None, prediction_length=1,
                                 quantile_levels=[0.1, 0.5, 0.9], id_column="item_id", timestamp_column="timestamp",
                                 target="target", cross_learning=cross, freq="YS")
            for _, r in fc.iterrows():
                if pd.Timestamp(r["timestamp"]).year != y:
                    raise ValueError(f"Chronos вернул прогноз не на тот год: {r['timestamp']}")
                rows.append({"model": name, "crop": r["item_id"], "year": y, "period": evaluation.period_of(y),
                             "y": frames[r["item_id"]].loc[y, "y"], "y_pred": float(r["0.5"]),
                             "q10": float(r["0.1"]), "q90": float(r["0.9"])})
        print(f"[chronos] {name}: готово ({time.time() - t0:.0f} с)", flush=True)

    pred = pd.DataFrame(rows).sort_values(["model", "crop", "year"])
    pred.to_csv(out / "predictions.csv", index=False)
    evaluation.report(pred, out, "E07-chronos2")
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(collect(experiment=out.name, model_id=MODEL_ID, covariates=COVARIATES,
                          variants={k: {"covariates": v[0], "cross_learning": v[1]} for k, v in variants.items()},
                          chronos=metadata.version("chronos-forecasting"), device=device, limits=LIMITS,
                          seconds=round(time.time() - t0)), f, ensure_ascii=False, indent=2)
    print(f"\nФайлы: {out}")


if __name__ == "__main__":
    main()
