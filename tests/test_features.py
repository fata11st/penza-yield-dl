import numpy as np

from penza_yield import data
from penza_yield.features import FEATURE_SETS, build_features


def _frame(target="p_grain_crops"):
    return build_features(data.load_annual(), data.load_monthly(check=False), target)


def test_lags_use_only_past_years():
    f = _frame()
    y = f["y"]
    for t in range(1993, 2026):
        assert f.loc[t, "lag1"] == y.loc[t - 1]
        assert f.loc[t, "lag2"] == y.loc[t - 2]
        assert np.isclose(f.loc[t, "ma3"], y.loc[t - 3 : t - 1].mean())


def test_current_year_target_not_in_features():
    f = _frame()
    for fs, cols in FEATURE_SETS.items():
        assert "y" not in cols, fs
        # ни один признак не совпадает с целевой переменной текущего года
        for c in cols:
            assert not np.allclose(f[c].dropna(), f["y"].loc[f[c].dropna().index]), (fs, c)


def test_climate_features_complete_and_ndvi_from_2000():
    f = _frame()
    for c in ("t_summer", "p_summer", "p_veget", "t_may", "swc_spring", "pet_summer"):
        assert f[c].notna().all(), c
    assert f.loc[:1999, "ndvi_peak"].isna().all()
    assert f.loc[2000:, "ndvi_peak"].notna().all()
