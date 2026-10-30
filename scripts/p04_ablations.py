"""P04 — абляции на панели (D13): вклад агропризнаков и каждой модели в ансамбль.

1) Признаки: LightGBM и MLP с конфигурациями, выбранными в P01 (без нового подбора), обучаются
   (а) только на лагах и (б) без каждой из 7 групп агропризнаков; годы 2011–2025.
2) Ансамбли (пересчёт из прогнозов P01, без обучения): ансамбль нейросетей без каждой из трёх моделей;
   среднее всех без каждого компонента; ансамбль классических моделей (Ridge + LightGBM).
Сравнение — как в P03: 9 новых регионов, 2019–2025, парный DM по прогнозам и по годовым средним.
Запуск: python scripts/p04_ablations.py  (--quick — пробный прогон) → experiments/P04-ablations/
"""
from penza_yield import resources

LIMITS = resources.apply_limits()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from penza_yield import metrics, panel_models as pm  # noqa: E402
from penza_yield.meta import collect  # noqa: E402
from penza_yield.panel import FEATURE_GROUPS, load_panel, period_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
YEARS = list(range(2011, 2026))
COMPONENTS = ["naive", "trend", "anchor", "ridge_p", "gbm_p", "mlp_p", "gru_p", "chronos_uni_p", "chronos_cov_p"]
NN = ["mlp_p", "gru_p", "chronos_cov_p"]


def selected_cfg(model: str) -> dict:
    sel = json.loads((EXP / f"P01-{model}" / "meta.json").read_text())["selected"]
    cfg = {}
    for part in sel.split("|"):
        k, v = part.split("=")
        cfg[k] = int(v) if v.isdigit() else float(v) if v.replace(".", "", 1).replace("e-", "", 1).isdigit() else v
    return cfg


def run_variant(P, model, cfg, name, years, seeds, threads):
    import torch

    torch.set_num_threads(threads)
    fn = {"gbm": pm.predict_gbm, "mlp": pm.predict_mlp}[model]
    out = []
    for t in years:
        p = fn(P, t, cfg, seeds)
        T = P.targets[P.targets.year == t][["region", "crop", "y"]]
        out.append(p.merge(T, on=["region", "crop"]).assign(model=name, year=t, period=period_of(t)))
    return pd.concat(out, ignore_index=True)


def ensemble_variants() -> pd.DataFrame:
    p = pd.concat([pd.read_csv(d / "predictions.csv") for d in sorted(EXP.glob("P01-*")) if not d.name.endswith("-quick")])
    p = p[p.year >= 2011]
    w = p.pivot_table(index=["region", "crop", "year"], columns="model", values="y_pred")[COMPONENTS].dropna()
    y = p.groupby(["region", "crop", "year"]).y.first().loc[w.index]
    ens = {"ens_nn": w[NN].mean(axis=1), "ens_mean": w[COMPONENTS].mean(axis=1), "ens_classic": w[["ridge_p", "gbm_p"]].mean(axis=1),
           "ens_classic_anchor": w[["ridge_p", "gbm_p", "anchor"]].mean(axis=1)}
    for m in NN:
        ens[f"ens_nn_minus_{m}"] = w[[c for c in NN if c != m]].mean(axis=1)
    for m in COMPONENTS:
        ens[f"ens_mean_minus_{m}"] = w[[c for c in COMPONENTS if c != m]].mean(axis=1)
    rows = []
    for name, v in ens.items():
        d = v.rename("y_pred").reset_index().assign(model=name)
        d["y"] = y.to_numpy()
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out["period"] = out.year.map(period_of)
    return out


def compare(pred: pd.DataFrame, reference: str) -> pd.DataFrame:
    ev = pred[(pred.period == "eval") & (pred.region != "penza")].assign(ape=lambda d: metrics.ape(d.y, d.y_pred))
    ref = ev[ev.model == reference]
    rows = []
    for m, g in ev.groupby("model"):
        j = g.merge(ref, on=["region", "crop", "year"], suffixes=("", "_r"))
        r = {"model": m, "reference": reference, "mape": g.ape.mean(), "n": len(g), "delta_vs_ref": g.ape.mean() - ref.ape.mean()}
        if m != reference:
            r["p_dm"] = metrics.paired_dm(j.ape, j.ape_r).p_value
            ya, yb = g.groupby("year").ape.mean(), ref.groupby("year").ape.mean()
            r["p_dm_yearly"] = metrics.paired_dm(ya.to_numpy(), yb.loc[ya.index].to_numpy()).p_value if len(ya) >= 3 else np.nan
        lo, hi = metrics.bootstrap_mape_ci({y: gg.ape.to_numpy() for y, gg in g.groupby("year")})
        r["ci95_lo"], r["ci95_hi"] = lo, hi
        rows.append(r)
    return pd.DataFrame(rows).sort_values("mape") if rows else pd.DataFrame(columns=["model", "mape"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = EXP / ("P04-ablations" + ("-quick" if args.quick else ""))
    out.mkdir(parents=True, exist_ok=True)
    P = load_panel()
    t0 = time.time()
    years = [2015, 2020, 2021] if args.quick else YEARS
    seeds = {"gbm": [0] if args.quick else list(range(5)), "mlp": [100] if args.quick else list(range(100, 120))}

    variants = []
    for model in ("gbm", "mlp"):
        base = selected_cfg(model)
        variants.append((model, {**base}, f"{model}_full"))
        variants.append((model, {**base, "features": "lags"}, f"{model}_lags_only"))
        if model == "gbm":
            variants += [(model, {**base, "drop_group": g}, f"gbm_minus_{g}") for g in FEATURE_GROUPS]
    if args.quick:
        variants = [variants[0], variants[1], variants[2], variants[-2], variants[-1]]
    threads = int(LIMITS["cpu"]["threads_per_process"])
    print(f"[P04] признаки: {len(variants)} вариантов × {len(years)} лет…", flush=True)
    feats = Parallel(n_jobs=int(LIMITS["cpu"]["n_jobs"]))(
        delayed(run_variant)(P, m, cfg, name, years, seeds[m], threads) for m, cfg, name in variants)
    feats = pd.concat(feats, ignore_index=True)
    feats.to_csv(out / "predictions_features.csv", index=False)
    print(f"[P04] признаки готовы за {time.time() - t0:.0f} с", flush=True)

    ens = ensemble_variants()
    ens.to_csv(out / "predictions_ensembles.csv", index=False)

    tab_g = compare(feats[feats.model.str.startswith("gbm")], "gbm_full")
    tab_m = compare(feats[feats.model.str.startswith("mlp")], "mlp_full")
    tab_e = compare(ens, "ens_nn")
    tab = pd.concat([tab_g.assign(block="features_gbm"), tab_m.assign(block="features_mlp"),
                     tab_e.assign(block="ensembles")], ignore_index=True)
    tab.to_csv(out / "summary.csv", index=False)
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(collect(experiment=out.name, years=years, variants=[v[2] for v in variants],
                          gbm_cfg=selected_cfg("gbm"), mlp_cfg=selected_cfg("mlp"), feature_groups=FEATURE_GROUPS,
                          limits=LIMITS, seconds=round(time.time() - t0)), f, ensure_ascii=False, indent=2)

    pd.set_option("display.width", 200)
    cols = ["model", "mape", "delta_vs_ref", "ci95_lo", "ci95_hi", "p_dm", "p_dm_yearly", "n"]
    for name, t in (("LightGBM: вклад признаков (опора — gbm_full)", tab_g), ("MLP (опора — mlp_full)", tab_m),
                    ("Ансамбли (опора — ens_nn)", tab_e)):
        print(f"\n{name}; 9 новых регионов, 2019–2025:")
        print(t[cols].round(3).to_string(index=False))
    print(f"\nГотово за {time.time() - t0:.0f} с → {out}")


if __name__ == "__main__":
    main()
