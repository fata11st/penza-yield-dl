"""Панель (D11): правила очистки и отсутствие утечки из будущего."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from penza_yield import panel_models as pm
from penza_yield.panel import ENG_FEATURES, load_panel


@pytest.fixture(scope="module")
def P():
    return load_panel()


def test_all_regions_loaded_with_same_layout(P):
    assert len(P.regions) == 10 and P.monthly.shape == (10, 36, 12, 16)
    assert P.features.shape == (360, len(ENG_FEATURES))


def test_sentinel_and_small_area_excluded(P):
    t = P.targets.set_index(["region", "crop", "year"])
    assert not t.loc[("chuvashia", "p_sugar_beet", 2021), "valid"]
    assert t.loc[("chuvashia", "p_sugar_beet", 2021), "excluded"] == "sentinel"
    assert (P.targets.loc[P.targets.excluded == "small_area", "area"] < 1.0).all()
    assert P.targets.loc[P.targets.valid, "y"].gt(0).all()


def test_ndvi_only_modis_and_pet_positive(P):
    nd, pet = P.variables.index("ndvi"), P.variables.index("pet_mm")
    assert np.isnan(P.monthly[:, :10, :, nd]).all()          # 1990–1999 — нет
    assert np.isnan(P.monthly[:, 10, 0, nd]).all()           # январь 2000 — нет
    assert np.nanmin(P.monthly[:, :, :, pet]) >= 0


def test_forecastable_needs_three_known_previous_years(P):
    t = P.targets
    g = t[(t.region == "samara") & (t.crop == "p_sugar_beet")].set_index("year")
    for yr in g.index[g.forecastable]:
        assert g.loc[yr - 3 : yr - 1, "valid"].all()
        assert np.isclose(g.loc[yr, "ma3"], g.loc[yr - 3 : yr - 1, "y"].mean())


def _perturb_future(P, t):
    feats = P.features.copy()
    future = feats.index.get_level_values("year") > t
    feats.loc[future, :] = feats.loc[future, :] + 1000.0
    cube = P.monthly.copy()
    cube[:, np.asarray(P.years) > t] += 1000.0
    tg = P.targets.copy()
    tg.loc[tg.year > t, ["y", "r", "lr1", "lr2"]] = 99.0
    return dataclasses.replace(P, features=feats, monthly=cube, targets=tg)


def test_tabular_anomalies_ignore_future(P):
    tr, te, cols = pm.tabular(P, 2015)
    tr2, te2, _ = pm.tabular(_perturb_future(P, 2015), 2015)
    assert tr.year.max() == 2014 and (te.year == 2015).all()
    pd.testing.assert_frame_equal(te[cols].reset_index(drop=True), te2[cols].reset_index(drop=True))
    pd.testing.assert_frame_equal(tr[cols].reset_index(drop=True), tr2[cols].reset_index(drop=True))


@pytest.mark.parametrize("fn,cfg", [
    (pm.predict_ridge, {"features": "lags_eng", "alpha": 10}),
    (pm.predict_mlp, {"hidden": 8, "wd": 1e-2, "epochs": 20}),
    (pm.predict_gru, {"hidden": 4, "wd": 1e-2, "epochs": 10}),
])
def test_models_ignore_future(P, fn, cfg):
    a = fn(P, 2015, cfg, [0]).sort_values(["region", "crop"]).y_pred.to_numpy()
    b = fn(_perturb_future(P, 2015), 2015, cfg, [0]).sort_values(["region", "crop"]).y_pred.to_numpy()
    assert np.allclose(a, b)


def test_baselines_use_past_only(P):
    a = pm.predict_baselines(P, 2015).sort_values(["model", "region", "crop"]).y_pred.to_numpy()
    b = pm.predict_baselines(_perturb_future(P, 2015), 2015).sort_values(["model", "region", "crop"]).y_pred.to_numpy()
    assert np.allclose(a, b)


def test_chronos_context_ends_before_target_year(P):
    ctx, fut, fb, test = pm.chronos_frames(P, 2015, with_cov=True)
    assert ctx.timestamp.dt.year.max() == 2014 and (fut.timestamp.dt.year == 2015).all()
    assert ctx.groupby("item_id").size().min() >= pm.MIN_CONTEXT
    assert len(fb) + ctx.item_id.nunique() == len(test)
