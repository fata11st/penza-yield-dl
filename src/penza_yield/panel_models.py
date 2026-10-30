"""Модели на панели регионов (D11). Каждая функция прогнозирует все ряды одного года t по данным лет < t.

Общее: цель r = log(y/ma3), прогноз ŷ = ma3 · exp(r̂); признаки — аномалии внутри региона по окну обучения;
обучение — на прогнозируемых строках (известны y и три предыдущих года) всех регионов с годом < t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from penza_yield.panel import CROPS, ENG_FEATURES, SEQ_VARIABLES, Panel, region_anomalies

LAGS = ["lr1", "lr2"]


# ---------------------------------------------------------------- эталоны
def predict_baselines(P: Panel, t: int) -> pd.DataFrame:
    T = P.targets
    test = T[(T.year == t) & T.forecastable]
    rows = []
    for _, r in test.iterrows():
        hist = T[(T.region == r.region) & (T.crop == r.crop) & (T.year < t) & T.valid]
        trend = np.nan
        if len(hist) >= 5:
            b = np.polyfit(hist.year, hist.y, 1)
            trend = max(np.polyval(b, t), 0.05 * r.ma3)  # линейный тренд не должен уходить в ноль
        rows += [{"model": "naive", "region": r.region, "crop": r.crop, "y_pred": r.lag1},
                 {"model": "trend", "region": r.region, "crop": r.crop, "y_pred": trend if np.isfinite(trend) else r.ma3},
                 {"model": "anchor", "region": r.region, "crop": r.crop, "y_pred": r.ma3}]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- табличные данные
def tabular(P: Panel, t: int, crop: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """(train, test, столбцы признаков): строки (регион, культура, год) с аномалиями агропризнаков."""
    T = P.targets[P.targets.forecastable]
    if crop is not None:
        T = T[T.crop == crop]
    df = T.join(P.features, on=["region", "year"])
    train_mask = df.year < t
    # нормы признаков — по годам < t каждого региона (по таблице регион×год, а не по строкам культур)
    F = P.features.reset_index()
    z = region_anomalies(F[ENG_FEATURES], F.region, F.year < t)
    z["ndvi_missing"] = F.ndvi_peak.isna().astype(float).to_numpy()
    z = z.fillna(0.0)
    z.index = pd.MultiIndex.from_frame(F[["region", "year"]])
    df = T.join(z, on=["region", "year"])
    cols = [*ENG_FEATURES, "ndvi_missing"]
    return df[train_mask], df[df.year == t], cols


def _to_pred(test: pd.DataFrame, r_hat, model: str) -> pd.DataFrame:
    return pd.DataFrame({"model": model, "region": test.region.to_numpy(), "crop": test.crop.to_numpy(),
                         "y_pred": test.ma3.to_numpy() * np.exp(np.asarray(r_hat, float))})


def predict_ridge(P: Panel, t: int, cfg: dict, seeds=None) -> pd.DataFrame:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    out = []
    for c in CROPS:
        tr, te, eng = tabular(P, t, c)
        if te.empty:
            continue
        cols = LAGS + (eng if cfg["features"] == "lags_eng" else [])
        Xtr = pd.concat([tr[cols], pd.get_dummies(tr.region, dtype=float)], axis=1)
        Xte = pd.concat([te[cols], pd.get_dummies(te.region, dtype=float)], axis=1).reindex(columns=Xtr.columns,
                                                                                           fill_value=0.0)
        sx = StandardScaler().fit(Xtr)
        m = Ridge(alpha=cfg["alpha"]).fit(sx.transform(Xtr), tr.r)
        out.append(_to_pred(te, m.predict(sx.transform(Xte)), "ridge_p"))
    return pd.concat(out, ignore_index=True)


def predict_gbm(P: Panel, t: int, cfg: dict, seeds) -> pd.DataFrame:
    import lightgbm as lgb

    out = []
    for c in CROPS:
        tr, te, eng = tabular(P, t, c)
        if te.empty:
            continue
        cols = LAGS + eng
        Xtr, Xte = tr[cols].copy(), te[cols].copy()
        regs = sorted(P.regions)
        Xtr["region_code"], Xte["region_code"] = tr.region.map(regs.index), te.region.map(regs.index)
        preds = []
        for s in seeds:
            m = lgb.LGBMRegressor(n_estimators=cfg["n_estimators"], num_leaves=cfg["num_leaves"], learning_rate=0.05,
                                  min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  reg_lambda=1.0, random_state=s, verbose=-1, n_jobs=1)
            m.fit(Xtr, tr.r, categorical_feature=["region_code"])
            preds.append(m.predict(Xte))
        out.append(_to_pred(te, np.mean(preds, axis=0), "gbm_p"))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- нейросети
def _train(model, inputs, target, mask, cfg, seed):
    import torch

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-2), weight_decay=cfg["wd"])
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(cfg["epochs"]):
        noisy = [x + cfg.get("noise", 0.1) * torch.randn(x.shape, generator=gen) if i == 0 else x
                 for i, x in enumerate(inputs)]
        err = (model(*noisy) - target) ** 2
        loss = (err * mask).sum() / mask.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


def predict_mlp(P: Panel, t: int, cfg: dict, seeds) -> pd.DataFrame:
    import torch
    from torch import nn

    tr, te, eng = tabular(P, t)
    cols = LAGS + eng
    crop_d = pd.get_dummies(pd.Categorical(tr.crop, categories=CROPS), dtype=float)
    reg_d = pd.get_dummies(pd.Categorical(tr.region, categories=P.regions), dtype=float)
    mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)

    def X(d):
        cd = pd.get_dummies(pd.Categorical(d.crop, categories=CROPS), dtype=float)
        rd = pd.get_dummies(pd.Categorical(d.region, categories=P.regions), dtype=float)
        return torch.tensor(np.hstack([((d[cols] - mu) / sd).fillna(0).to_numpy(), cd.to_numpy(), rd.to_numpy()]),
                            dtype=torch.float32)

    # цель стандартизуется по культуре
    stats = tr.groupby("crop").r.agg(["mean", "std"])
    tgt = torch.tensor(((tr.r - tr.crop.map(stats["mean"])) / tr.crop.map(stats["std"])).to_numpy(),
                       dtype=torch.float32)[:, None]
    xtr, xte = X(tr), X(te)
    n_in = xtr.shape[1]
    _ = crop_d, reg_d
    preds = []
    for s in seeds:
        torch.manual_seed(s)
        net = nn.Sequential(nn.Linear(n_in, cfg["hidden"]), nn.ReLU(), nn.Dropout(0.2), nn.Linear(cfg["hidden"], 1))
        net = _train(net, [xtr], tgt, torch.ones_like(tgt), {**cfg, "noise": 0.0}, s)
        with torch.no_grad():
            preds.append(net(xte).numpy()[:, 0])
    r_hat = np.mean(preds, axis=0) * te.crop.map(stats["std"]).to_numpy() + te.crop.map(stats["mean"]).to_numpy()
    return _to_pred(te, r_hat, "mlp_p")


def _seq_tensors(P: Panel, t: int, cutoff: int = 9):
    """Последовательности (регион, год): аномалии месяца в регионе по годам < t, январь…cutoff."""
    vi = [P.variables.index(v) for v in SEQ_VARIABLES]
    cube = P.monthly[:, :, :cutoff, :][:, :, :, vi]                  # [R, Y, M, V]
    years = np.asarray(P.years)
    train_y = years < t
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(cube[:, train_y], axis=1, keepdims=True)    # норма месяца в регионе
        sd = np.nanstd(cube[:, train_y], axis=1, keepdims=True)
    sd = np.where(np.isnan(sd) | (sd < 1e-8), 1.0, sd)
    z = (cube - np.nan_to_num(mu)) / sd
    avail = (~np.isnan(z[..., [SEQ_VARIABLES.index("ndvi")]])).astype(float)
    z = np.nan_to_num(z)
    return np.concatenate([z, avail], axis=-1)                         # [R, Y, M, V+1]


def predict_gru(P: Panel, t: int, cfg: dict, seeds) -> pd.DataFrame:
    import torch
    from torch import nn

    T = P.targets
    seq = _seq_tensors(P, t)
    R, Y = len(P.regions), len(P.years)
    piv = {k: T.pivot_table(index=["region", "year"], columns="crop", values=k, dropna=False)
           .reindex(pd.MultiIndex.from_product([P.regions, P.years]), columns=CROPS) for k in ("r", "lr1", "lr2")}
    fc = T.pivot_table(index=["region", "year"], columns="crop", values="forecastable", dropna=False) \
        .reindex(pd.MultiIndex.from_product([P.regions, P.years]), columns=CROPS).fillna(False).astype(bool)
    lags = np.concatenate([piv["lr1"].to_numpy(), piv["lr2"].to_numpy()], axis=1).reshape(R, Y, -1)
    lag_mask = (~np.isnan(lags)).astype(float)
    reg_onehot = np.repeat(np.eye(R)[:, None, :], Y, axis=1)
    years = np.asarray(P.years)
    r = piv["r"].to_numpy().reshape(R, Y, len(CROPS))
    m = fc.to_numpy().reshape(R, Y, len(CROPS))

    tr = m[:, years < t].any(-1)
    te_y = list(P.years).index(t)
    r_tr = np.where(m[:, years < t], r[:, years < t], np.nan)
    mu, sd = np.nanmean(r_tr, axis=(0, 1)), np.nanstd(r_tr, axis=(0, 1)) + 1e-8
    l_tr = lags[:, years < t]
    l_mu, l_sd = np.nanmean(l_tr, axis=(0, 1)), np.nanstd(l_tr, axis=(0, 1)) + 1e-8
    static = np.concatenate([np.nan_to_num((lags - l_mu) / l_sd), lag_mask, reg_onehot], axis=-1)

    def sel(a, idx_years, rows):
        return a[:, idx_years][rows]

    ytr = np.flatnonzero(years < t)
    rows_tr = tr
    s_tr = torch.tensor(sel(seq, ytr, rows_tr), dtype=torch.float32)
    st_tr = torch.tensor(sel(static, ytr, rows_tr), dtype=torch.float32)
    tg = sel(np.nan_to_num((r - mu) / sd), ytr, rows_tr)
    mk = sel(m.astype(float), ytr, rows_tr)
    tg, mk = torch.tensor(tg, dtype=torch.float32), torch.tensor(mk, dtype=torch.float32)
    s_te = torch.tensor(seq[:, te_y], dtype=torch.float32)
    st_te = torch.tensor(static[:, te_y], dtype=torch.float32)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(s_tr.shape[-1], cfg["hidden"], batch_first=True)
            self.drop = nn.Dropout(0.2)
            self.head = nn.Linear(cfg["hidden"] + st_tr.shape[-1], len(CROPS))

        def forward(self, s, st):
            _, h = self.gru(s)
            return self.head(self.drop(torch.cat([h[-1], st], dim=1)))

    preds = []
    for s in seeds:
        torch.manual_seed(s)
        net = _train(Net(), [s_tr, st_tr], tg, mk, cfg, s)
        with torch.no_grad():
            preds.append(net(s_te, st_te).numpy())
    r_hat = np.mean(preds, axis=0) * sd + mu                          # [R, crops]
    rows = []
    test = T[(T.year == t) & T.forecastable]
    for _, x in test.iterrows():
        rows.append({"model": "gru_p", "region": x.region, "crop": x.crop,
                     "y_pred": x.ma3 * np.exp(r_hat[P.regions.index(x.region), CROPS.index(x.crop)])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Chronos-2
CHRONOS_COVARIATES = [f for f in ENG_FEATURES if not f.startswith("ndvi")]
MIN_CONTEXT = 8


def chronos_frames(P: Panel, t: int, with_cov: bool):
    """Контекст — непрерывный участок известной урожайности до t−1 (≥ MIN_CONTEXT лет) для каждого ряда года t."""
    T = P.targets
    test = T[(T.year == t) & T.forecastable]
    ctx, fut, fallback = [], [], []
    for _, x in test.iterrows():
        h = T[(T.region == x.region) & (T.crop == x.crop) & (T.year < t)].set_index("year").y
        run = []
        for yr in range(t - 1, min(P.years) - 1, -1):
            if np.isnan(h.get(yr, np.nan)):
                break
            run.append(yr)
        if len(run) < MIN_CONTEXT:
            fallback.append((x.region, x.crop))
            continue
        run = sorted(run)
        sid = f"{x.region}|{x.crop}"
        c = pd.DataFrame({"item_id": sid, "timestamp": pd.to_datetime([f"{y}-01-01" for y in run]), "target": h.loc[run].to_numpy()})
        u = pd.DataFrame({"item_id": [sid], "timestamp": [pd.Timestamp(f"{t}-01-01")]})
        if with_cov:
            f = P.features.loc[x.region]
            for v in CHRONOS_COVARIATES:
                c[v] = f.loc[run, v].to_numpy()
                u[v] = [f.loc[t, v]]
        ctx.append(c)
        fut.append(u)
    return (pd.concat(ctx, ignore_index=True) if ctx else None,
            pd.concat(fut, ignore_index=True) if fut else None, fallback, test)


def predict_chronos(P: Panel, t: int, pipe, with_cov: bool, name: str) -> tuple[pd.DataFrame, int]:
    ctx, fut, fallback, test = chronos_frames(P, t, with_cov)
    rows = []
    if ctx is not None:
        fc = pipe.predict_df(ctx, future_df=fut if with_cov else None, prediction_length=1, quantile_levels=[0.5],
                             id_column="item_id", timestamp_column="timestamp", target="target",
                             cross_learning=True, freq="YS")
        for _, r in fc.iterrows():
            if pd.Timestamp(r["timestamp"]).year != t:
                raise ValueError("Chronos вернул прогноз не на тот год")
            region, crop = r["item_id"].split("|")
            rows.append({"model": name, "region": region, "crop": crop, "y_pred": float(r["0.5"])})
    for region, crop in fallback:   # слишком короткий непрерывный ряд → якорь
        ma3 = test[(test.region == region) & (test.crop == crop)].ma3.iloc[0]
        rows.append({"model": name, "region": region, "crop": crop, "y_pred": ma3})
    return pd.DataFrame(rows), len(fallback)
