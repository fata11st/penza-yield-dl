"""Панель 10 регионов Поволжья и Черноземья (1990–2025): загрузка, очистка, агрономические признаки (решение D11).

Правила очистки (зафиксированы в D11 до запуска моделей):
  * целевые значения не заполняются; урожайность исключается, если площадь < 1 тыс. га или в площади/сборе
    служебное значение (≥ 1e6, например 999999999 у Чувашии, свёкла 2021);
  * NDVI — только MODIS (с февраля 2000 г.): ряды GIMMS 1990-х систематически выше на 0,04–0,07 (смена прибора);
  * потенциальная испаряемость ERA5 приводится к положительному знаку (у Пензы она в соглашении ERA5 отрицательна);
  * показатели листа «Вегетация_GTK» не используются: в Пензе и в остальных регионах они посчитаны разными методами;
    агрономические признаки строятся заново одинаково для всех регионов из помесячных рядов;
  * модели используют аномалии внутри региона (норма и разброс — по окну обучения), поэтому разные источники
    станционного климата (метеостанции, NASA POWER) не смешиваются по уровню.
"""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from penza_yield import data

ROOT = Path(__file__).resolve().parents[2]
REGIONS_DIR = ROOT / "data" / "raw" / "regions"

# Пенза — основной регион прошлых экспериментов (её годы оценки уже просмотрены); остальные — новые данные.
REGIONS = {
    "penza": ("Пензенская обл.", data.DEFAULT_PATH),
    "chuvashia": ("Чувашская Респ.", REGIONS_DIR / "chuvashia_dataset_2025.xlsx"),
    "mordovia": ("Респ. Мордовия", REGIONS_DIR / "mordovia_dataset_2025.xlsx"),
    "nizhny": ("Нижегородская обл.", REGIONS_DIR / "nizhny_dataset_2025.xlsx"),
    "ryazan": ("Рязанская обл.", REGIONS_DIR / "ryazan_dataset_2025.xlsx"),
    "samara": ("Самарская обл.", REGIONS_DIR / "samara_dataset_2025.xlsx"),
    "saratov": ("Саратовская обл.", REGIONS_DIR / "saratov_dataset_2025.xlsx"),
    "tambov": ("Тамбовская обл.", REGIONS_DIR / "tambov_dataset_2025.xlsx"),
    "tatarstan": ("Респ. Татарстан", REGIONS_DIR / "tatarstan_dataset_2025.xlsx"),
    "ulyanovsk": ("Ульяновская обл.", REGIONS_DIR / "ulyanovsk_dataset_2025.xlsx"),
}
NEW_REGIONS = [r for r in REGIONS if r != "penza"]
SHA256 = {
    "penza": data.DATASET_SHA256,
    "chuvashia": "4081c6cdeb464176a2c64f0c9e88083869dbeda242b9356c9a183de4dd5da758",
    "mordovia": "61c967d2f3d0c2a946ef5630b6f7617488bce3d18e1f84a38b42dfc80d7c03e5",
    "nizhny": "bcd8eab3826a153a9a343ee501635e8d2d129433be2c90eb052d49ff04b837c9",
    "ryazan": "cd857c11f34c1a6727f535ff8440691cfa1a6d1a2b419b6a5e844fc3b618b61d",
    "samara": "a42d3cf0e82696277e15cfa3c2fdd1c6ea76a0d0610003ce1c45881ee5b3a7db",
    "saratov": "33df7c8ec0fdcd1291587450c8355d0c75fe519e6fa65abc1fc0e32d853d5e4c",
    "tambov": "4467d5c5c5acbc8a3c04ad3690950989f95f4db414d94de660b9036be0a2424b",
    "tatarstan": "c402a73f8612969ebb4e5357ab3075a7104692d214545c34f21bf2905b82bbee",
    "ulyanovsk": "4e4b586f7846161ab3e02c81bf2880d85c10d04df8151a4a7d76cbed4c66d26c",
}
CROPS = data.TARGETS
MIN_AREA = 1.0          # тыс. га
SENTINEL = 1e6
YEARS = list(range(1990, 2026))
DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# Агрономические признаки (D11): одинаковы для всех регионов, только из помесячных рядов до сентября включительно
ENG_FEATURES = [
    "t_may_jun", "t_jul_aug", "tmax_jul_aug",          # тепло; жара в налив
    "p_apr_may", "p_jun", "p_jul_aug",                 # осадки по фазам
    "htc_may_aug",                                     # ГТК Селянинова по месяцам
    "wb_may_aug",                                      # водный баланс: осадки ERA5 − потенциальная испаряемость
    "swc_apr", "swc_deep_apr", "swc_jul",              # влагозапас весной (0,07–0,28 и 0,28–1 м) и в июле
    "swe_winter", "frost_min",                         # запас воды в снеге; минимум температуры почвы Я–Ф
    "rad_jun_aug",                                     # солнечная радиация
    "ndvi_may", "ndvi_peak",                           # состояние посевов (с 2000 г.)
]
# Группы агропризнаков для абляций (D13)
FEATURE_GROUPS = {
    "heat": ["t_may_jun", "t_jul_aug", "tmax_jul_aug"],
    "precip": ["p_apr_may", "p_jun", "p_jul_aug"],
    "water_balance": ["htc_may_aug", "wb_may_aug"],
    "soil_moisture": ["swc_apr", "swc_deep_apr", "swc_jul"],
    "winter": ["swe_winter", "frost_min"],
    "radiation": ["rad_jun_aug"],
    "ndvi": ["ndvi_may", "ndvi_peak"],
}
SEQ_VARIABLES = ["mean_temp", "max_temp", "rf_sum", "swc_7_28cm", "swc_28_100cm", "pet_mm", "solar_rad_jm2", "ndvi"]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_all() -> None:
    for r, (_, path) in REGIONS.items():
        if _sha(path) != SHA256[r]:
            raise data.DatasetHashError(f"{r}: файл {path.name} отличается от зафиксированной версии")


@dataclass(frozen=True)
class Panel:
    targets: pd.DataFrame        # region, crop, year, y, area, valid
    monthly: np.ndarray          # [n_regions, n_years, 12, n_vars], NaN — пропуск
    variables: list[str]
    regions: list[str]
    years: list[int]
    features: pd.DataFrame       # индекс (region, year), столбцы ENG_FEATURES

    def region_index(self, r: str) -> int:
        return self.regions.index(r)


def _clean_targets(region: str, path: Path) -> pd.DataFrame:
    y = pd.read_excel(path, sheet_name="Урожайность и площади").dropna(subset=["Год"])
    y["Год"] = y["Год"].astype(int)
    y = y.set_index("Год").reindex(YEARS)
    rows = []
    for c in CROPS:
        s_col, w_col = c.replace("p_", "s_"), c.replace("p_", "w_")
        p = pd.to_numeric(y[c], errors="coerce")
        s = pd.to_numeric(y[s_col], errors="coerce")
        w = pd.to_numeric(y.get(w_col), errors="coerce") if w_col in y else pd.Series(np.nan, index=y.index)
        sentinel = (s >= SENTINEL) | (w >= SENTINEL)
        small = s < MIN_AREA
        valid = p.notna() & (p > 0) & ~sentinel & ~small
        for yr in YEARS:
            rows.append({"region": region, "crop": c, "year": yr, "y": p[yr] if valid[yr] else np.nan,
                         "area": np.nan if sentinel[yr] else s[yr], "valid": bool(valid[yr]),
                         "excluded": "sentinel" if sentinel[yr] else ("small_area" if (small[yr] and p.notna()[yr])
                                                                      else "")})
    return pd.DataFrame(rows)


def _monthly(path: Path) -> tuple[np.ndarray, list[str]]:
    m = data.load_monthly(path, check=False)
    vals = m.values.astype(float).copy()
    vals[~m.mask] = np.nan
    years = list(m.years)
    if years != YEARS:
        vals = pd.DataFrame(vals.reshape(len(years), -1), index=years).reindex(YEARS).to_numpy().reshape(
            len(YEARS), 12, -1)
    v = list(m.variables)
    vals[:, :, v.index("pet_mm")] = np.abs(vals[:, :, v.index("pet_mm")])
    nd = v.index("ndvi")
    vals[: YEARS.index(2000), :, nd] = np.nan        # GIMMS 1990-х не используем (ступенька прибора)
    vals[YEARS.index(2000), 0, nd] = np.nan          # январь 2000 г. — до начала MODIS
    return vals, v


def _engineer(cube: np.ndarray, variables: list[str], region: str) -> pd.DataFrame:
    V = {name: cube[:, :, i] for i, name in enumerate(variables)}  # [years, 12]

    def mon(name, months, how="mean"):
        x = V[name][:, [k - 1 for k in months]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = {"mean": np.nanmean, "sum": np.nansum, "max": np.nanmax, "min": np.nanmin}[how](x, axis=1)
        return np.where(np.isnan(x).all(axis=1), np.nan, out)

    f = pd.DataFrame(index=pd.MultiIndex.from_product([[region], YEARS], names=["region", "year"]))
    f["t_may_jun"] = mon("mean_temp", [5, 6])
    f["t_jul_aug"] = mon("mean_temp", [7, 8])
    f["tmax_jul_aug"] = mon("max_temp", [7, 8])
    f["p_apr_may"] = mon("rf_sum", [4, 5], "sum")
    f["p_jun"] = mon("rf_sum", [6], "sum")
    f["p_jul_aug"] = mon("rf_sum", [7, 8], "sum")
    t = V["mean_temp"][:, 4:8]
    heat = np.where(t > 10, t, 0.0) * DAYS[4:8]
    with np.errstate(invalid="ignore", divide="ignore"):
        f["htc_may_aug"] = np.where(heat.sum(1) > 0, 10 * np.nansum(V["rf_sum"][:, 4:8], 1) / heat.sum(1), np.nan)
    f["wb_may_aug"] = mon("precip_mm", [5, 6, 7, 8], "sum") - mon("pet_mm", [5, 6, 7, 8], "sum")
    f["swc_apr"] = mon("swc_7_28cm", [4])
    f["swc_deep_apr"] = mon("swc_28_100cm", [4])
    f["swc_jul"] = mon("swc_7_28cm", [7])
    swe = V["snow_water_eq_mm"]
    prev_dec = np.concatenate([[np.nan], swe[:-1, 11]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f["swe_winter"] = np.nanmax(np.column_stack([prev_dec, swe[:, 0:3]]), axis=1)
    f["frost_min"] = mon("soil_temp_0_7cm", [1, 2], "min")
    f["rad_jun_aug"] = mon("solar_rad_jm2", [6, 7, 8])
    f["ndvi_may"] = mon("ndvi", [5])
    f["ndvi_peak"] = mon("ndvi", [6, 7, 8], "max")
    return f


def load_panel(check: bool = True) -> Panel:
    if check:
        verify_all()
    targets, cubes, feats, variables = [], [], [], None
    for r, (_, path) in REGIONS.items():
        targets.append(_clean_targets(r, path))
        cube, v = _monthly(path)
        variables = variables or v
        if v != variables:
            raise ValueError(f"{r}: другой набор помесячных переменных")
        cubes.append(cube)
        feats.append(_engineer(cube, v, r))
    t = pd.concat(targets, ignore_index=True).sort_values(["region", "crop", "year"]).reset_index(drop=True)
    g = t.groupby(["region", "crop"]).y
    t["lag1"], t["lag2"] = g.shift(1), g.shift(2)
    t["ma3"] = g.transform(lambda s: s.shift(1).rolling(3).mean())  # три предыдущих года, все должны быть известны
    with np.errstate(invalid="ignore", divide="ignore"):
        t["r"] = np.log(t.y / t.ma3)
        t["lr1"], t["lr2"] = np.log(t.lag1 / t.ma3), np.log(t.lag2 / t.ma3)
    t["forecastable"] = t.valid & t.ma3.notna() & t.lag1.notna() & t.lag2.notna()
    return Panel(t, np.stack(cubes), variables, list(REGIONS), YEARS, pd.concat(feats))


def period_of(year: int) -> str:
    from penza_yield import validation

    if year in validation.EVAL_YEARS:
        return "eval"
    if year in validation.TUNE_YEARS:
        return "tune"
    return "warmup"


def region_anomalies(values: pd.DataFrame, regions: pd.Series, train_mask: pd.Series) -> pd.DataFrame:
    """Стандартизация признаков внутри каждого региона по строкам обучения (без утечки)."""
    out = values.copy().astype(float)
    for r in regions.unique():
        m = regions == r
        tr = values[m & train_mask]
        mu, sd = tr.mean(), tr.std().replace(0, np.nan)
        out.loc[m] = ((values[m] - mu) / sd).to_numpy()
    return out
