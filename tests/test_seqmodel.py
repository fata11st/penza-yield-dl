"""Проверки отсутствия утечки во внутрисезонной модели E02."""
import dataclasses

import numpy as np
import pytest

from penza_yield import data
from penza_yield.seqmodel import GRUConfig, build_dataset, fit_predict

CFG = GRUConfig(hidden=4, epochs=15)
SEEDS = [0]


@pytest.fixture(scope="module")
def ds():
    return build_dataset(data.load_annual(), data.load_monthly(check=False))


def _idx(ds, test_year):
    years = ds.years
    return np.flatnonzero(years < test_year), np.flatnonzero(years == test_year)


def _perturb(ds, years_mask, months):
    seq = ds.seq.copy()
    seq[np.ix_(years_mask, months)] += 100.0
    return dataclasses.replace(ds, seq=seq)


def test_target_and_anchor_use_past_only(ds):
    t = list(ds.years).index(2015)
    assert np.allclose(ds.anchor[t], ds.y[t - 3 : t].mean(0))
    assert np.allclose(ds.r[t], np.log(ds.y[t] / ds.anchor[t]))
    assert not ds.trainable()[:3].any() and ds.trainable()[3:].all()


def test_months_after_cutoff_do_not_affect_forecast(ds):
    tr, te = _idx(ds, 2020)
    base = fit_predict(ds, tr, te, 6, CFG, SEEDS)
    moved = fit_predict(_perturb(ds, np.ones(len(ds.years), bool), list(range(6, 12))), tr, te, 6, CFG, SEEDS)
    assert np.allclose(base, moved)


def test_future_years_do_not_affect_forecast(ds):
    tr, te = _idx(ds, 2015)
    base = fit_predict(ds, tr, te, 8, CFG, SEEDS)
    future = ds.years > 2015
    ds2 = _perturb(ds, future, list(range(12)))
    ds2 = dataclasses.replace(ds2, r=np.where(future[:, None], 0.0, ds.r))
    assert np.allclose(base, fit_predict(ds2, tr, te, 8, CFG, SEEDS))


def test_in_season_months_are_used(ds):
    tr, te = _idx(ds, 2020)
    base = fit_predict(ds, tr, te, 8, CFG, SEEDS)
    moved = fit_predict(_perturb(ds, ds.years == 2020, [5, 6]), tr, te, 8, CFG, SEEDS)
    assert not np.allclose(base, moved), "модель должна реагировать на погоду июня–июля прогнозируемого года"


def test_leaky_split_rejected(ds):
    with pytest.raises(ValueError):
        fit_predict(ds, np.arange(len(ds.years)), [len(ds.years) - 1], 8, CFG, SEEDS)


def test_forecast_is_positive_and_deterministic(ds):
    tr, te = _idx(ds, 2019)
    a = fit_predict(ds, tr, te, 9, CFG, [1, 2])
    b = fit_predict(ds, tr, te, 9, CFG, [1, 2])
    assert a.shape == (1, 5) and (a > 0).all() and np.allclose(a, b)
