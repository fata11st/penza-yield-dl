import numpy as np
import pytest

from penza_yield import evaluation
from penza_yield.tabular import FEATURE_SETS, crop_frames, fit_predict_mlp, split


@pytest.fixture(scope="module")
def frames():
    return crop_frames()


def test_split_has_no_future_years(frames):
    f = frames["p_sugar_beet"]
    for year in (2011, 2019, 2025):
        train, test = split(f, year, FEATURE_SETS["ar_climate"])
        assert train.index.max() == year - 1 and list(test.index) == [year]
        assert train.r.notna().all()


def test_ratio_target_matches_anchor(frames):
    f = frames["p_potato"]
    assert np.allclose(f.loc[2015, "y"], f.loc[2015, "ma3"] * np.exp(f.loc[2015, "r"]))


def test_mlp_forecast_ignores_future(frames):
    f = frames["p_grain_crops"]
    feats = FEATURE_SETS["ar_climate"]
    base = fit_predict_mlp(f, 2015, feats, 4, 1.0, [0])
    g = f.copy()
    g.loc[g.index > 2015, ["y", "r", *feats]] = 999.0
    assert base == pytest.approx(fit_predict_mlp(g, 2015, feats, 4, 1.0, [0]))


def test_anchor_reference_is_three_year_mean(frames):
    a = evaluation.anchor_predictions().set_index(["crop", "year"])
    f = frames["p_vegetables"]
    assert a.loc[("p_vegetables", 2020), "y_pred"] == pytest.approx(f.loc[2017:2019, "y"].mean())
