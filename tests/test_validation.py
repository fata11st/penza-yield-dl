import numpy as np
import pytest

from penza_yield import metrics, validation

YEARS = list(range(1990, 2026))


def test_expanding_window_never_sees_future():
    folds = list(validation.expanding_window(YEARS, range(2011, 2026)))
    assert [f.test_year for f in folds] == list(range(2011, 2026))
    for f in folds:
        train_years = np.asarray(YEARS)[f.train_idx]
        assert train_years.max() < f.test_year
        assert train_years.max() == f.test_year - 1  # окно действительно расширяется до года перед прогнозом


def test_train_start_and_min_train():
    f = next(validation.expanding_window(YEARS, [2011], train_start=2000))
    assert np.asarray(YEARS)[f.train_idx].tolist() == list(range(2000, 2011))
    with pytest.raises(ValueError):
        next(validation.expanding_window(YEARS, [2011], train_start=2000, min_train=12))


def test_tune_and_eval_years_disjoint():
    assert not set(validation.TUNE_YEARS) & set(validation.EVAL_YEARS)
    assert max(validation.TUNE_YEARS) < min(validation.EVAL_YEARS)


def test_exponential_weights():
    w = validation.exponential_weights([2008, 2009, 2010], 0.5)
    assert np.allclose(w, [0.25, 0.5, 1.0])
    assert np.allclose(validation.exponential_weights([1, 2, 3], 1.0), 1.0)


def test_mape_known_value():
    assert metrics.mape([100, 200], [110, 180]) == pytest.approx(10.0)


def test_dm_detects_better_forecast_and_is_antisymmetric():
    rng = np.random.default_rng(0)
    y = 10 + rng.normal(size=25)
    good = y + rng.normal(scale=0.1, size=25)
    bad = y + rng.normal(scale=2.0, size=25)
    ab = metrics.diebold_mariano(y, good, bad)
    ba = metrics.diebold_mariano(y, bad, good)
    assert ab.mean_loss_diff < 0 and ab.p_value < 0.01
    assert ab.statistic == pytest.approx(-ba.statistic)
    assert ab.p_value == pytest.approx(ba.p_value)
