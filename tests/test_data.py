import numpy as np
import pytest

from penza_yield import data


def test_dataset_hash_matches():
    data.verify()


def test_annual_targets_complete():
    df = data.load_annual()
    assert df.index.min() == 1990 and df.index.max() == 2025 and len(df) == 36
    assert df[data.TARGETS].notna().all().all(), "урожайность не должна иметь пропусков"
    assert (df[data.TARGETS] > 0).all().all()


def test_monthly_shape_and_variables():
    m = data.load_monthly()
    assert m.values.shape == (36, 12, len(m.variables))
    assert list(m.years) == list(range(1990, 2026))
    assert len(m.variables) == 16
    # Температура и осадки метеостанции — без пропусков за все годы
    for v in ("mean_temp", "rf_sum", "swc_7_28cm"):
        assert m.mask[:, :, m.variables.index(v)].all(), v


def test_ndvi_missing_only_before_feb_2000():
    m = data.load_monthly()
    nd = m.mask[:, :, m.variables.index("ndvi")]
    years = list(m.years)
    assert not nd[: years.index(2000)].any(), "NDVI до 2000 г. быть не должно"
    assert not nd[years.index(2000), 0], "январь 2000 г. — до начала съёмки MODIS"
    assert nd[years.index(2000) + 1 :].all()


def test_up_to_month_hides_future_months():
    m = data.load_monthly().up_to_month(6)
    assert not m.mask[:, 6:, :].any()
    assert np.isnan(m.values[:, 6:, :]).all()
    with pytest.raises(ValueError):
        m.up_to_month(13)


def test_changed_file_is_rejected(tmp_path):
    bad = tmp_path / "bad.xlsx"
    bad.write_bytes(data.DEFAULT_PATH.read_bytes() + b"x")
    with pytest.raises(data.DatasetHashError):
        data.verify(bad)
