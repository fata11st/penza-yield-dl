"""P02 — ансамбли прогнозов на панели (D11). Веса для года t — только по прогнозам лет 2006…t−1.

Для каждой культуры отдельно, состав компонентов фиксирован:
  ens_mean      — среднее всех компонентов;
  ens_median    — медиана всех компонентов;
  ens_nn        — среднее нейросетевых (MLP, GRU, Chronos-2 с ковариатами);
  ens_invmape   — веса ∝ 1 / MAPE компонента по прошлым годам;
  ens_stack     — неотрицательные веса с суммой 1, МНК в лог-шкале по прошлым годам;
  ens_select    — компонент с наименьшим MAPE по прошлым годам.
Запуск: python scripts/p02_ensembles.py  → experiments/P02-ensembles/
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from penza_yield import metrics
from penza_yield.meta import collect
from penza_yield.panel import CROPS, period_of

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "P02-ensembles"
COMPONENTS = ["naive", "trend", "anchor", "ridge_p", "gbm_p", "mlp_p", "gru_p", "chronos_uni_p", "chronos_cov_p"]
NN = ["mlp_p", "gru_p", "chronos_cov_p"]
FIRST = 2011


def load_components() -> pd.DataFrame:
    parts = [pd.read_csv(d / "predictions.csv") for d in sorted((ROOT / "experiments").glob("P01-*"))
             if not d.name.endswith("-quick")]
    p = pd.concat(parts, ignore_index=True)
    wide = p.pivot_table(index=["region", "crop", "year"], columns="model", values="y_pred")
    y = p.groupby(["region", "crop", "year"]).y.first()
    missing = [c for c in COMPONENTS if c not in wide]
    if missing:
        raise SystemExit(f"Нет прогнозов компонентов: {missing} — сначала запустите P01")
    wide = wide[COMPONENTS].dropna()
    wide["y"] = y.loc[wide.index]
    return wide.reset_index()


def stack_weights(past: pd.DataFrame) -> np.ndarray:
    X = np.log(past[COMPONENTS].to_numpy())
    y = np.log(past.y.to_numpy())
    k = len(COMPONENTS)
    res = minimize(lambda w: np.sum((y - X @ w) ** 2), np.full(k, 1 / k), method="SLSQP",
                   bounds=[(0, 1)] * k, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return res.x


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    W = load_components()
    rows, weights = [], []
    for crop in CROPS:
        wc = W[W.crop == crop]
        for t in sorted(y for y in wc.year.unique() if y >= FIRST):
            cur, past = wc[wc.year == t], wc[wc.year < t]
            if past.empty:  # нет прошлых прогнозов — равные веса (в полном запуске не случается: разгон с 2006 г.)
                w_inv = w_st = np.full(len(COMPONENTS), 1 / len(COMPONENTS))
                best = "anchor"
            else:
                mape_past = {c: metrics.mape(past.y, past[c]) for c in COMPONENTS}
                inv = np.array([1 / mape_past[c] for c in COMPONENTS])
                w_inv = inv / inv.sum()
                w_st = stack_weights(past)
                best = min(mape_past, key=mape_past.get)
            comp = cur[COMPONENTS].to_numpy()
            ens = {
                "ens_mean": comp.mean(1),
                "ens_median": np.median(comp, axis=1),
                "ens_nn": cur[NN].to_numpy().mean(1),
                "ens_invmape": comp @ w_inv,
                "ens_stack": np.exp(np.log(comp) @ w_st),
                "ens_select": cur[best].to_numpy(),
            }
            for name, v in ens.items():
                rows += [{"model": name, "region": r, "crop": crop, "year": t, "period": period_of(t), "y": y_, "y_pred": p_}
                         for r, y_, p_ in zip(cur.region, cur.y, v)]
            weights.append({"crop": crop, "year": t, "selected": best,
                            **{f"inv_{c}": w for c, w in zip(COMPONENTS, w_inv)},
                            **{f"stack_{c}": w for c, w in zip(COMPONENTS, w_st)}})
    pd.DataFrame(rows).to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame(weights).to_csv(OUT / "weights.csv", index=False)
    with open(OUT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(collect(experiment=OUT.name, components=COMPONENTS, nn=NN, first_year=FIRST), f,
                  ensure_ascii=False, indent=2)
    print(f"Ансамбли готовы: {len(rows)} прогнозов → {OUT}")


if __name__ == "__main__":
    main()
