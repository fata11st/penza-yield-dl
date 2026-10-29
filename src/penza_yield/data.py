"""Загрузка датасета Пензенской области (1990–2025).

Два представления одних и тех же данных:

* ``load_annual`` — годовая таблица (лист «Сводный_ML»), как в работах на основе Ridge-регрессии;
* ``load_monthly`` — помесячный тензор ``[год, месяц, переменная]`` для последовательностных моделей,
  с маской наличия значений. Пропуски не заполняются: решение о заполнении принимает модель.

Файл датасета сверяется по SHA-256, чтобы все эксперименты гарантированно шли на одной версии данных.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "data" / "raw" / "penza_dataset_2025.xlsx"
DATASET_SHA256 = "4b916c038d760c6d069078bd6b1a2d667c25547a7f4d7576eab98d522aad4b68"

YEAR_COL = "Год"

# Целевые переменные: урожайность, т/га
TARGETS = ["p_grain_crops", "p_sugar_beet", "p_sunflower", "p_potato", "p_vegetables"]
CROP_NAMES_RU = {
    "p_grain_crops": "Зерновые и зернобобовые",
    "p_sugar_beet": "Сахарная свёкла",
    "p_sunflower": "Подсолнечник",
    "p_potato": "Картофель",
    "p_vegetables": "Овощи",
}

# Листы с помесячными рядами и переменные в них
MONTHLY_SHEETS = {
    "Климат_температура": ["mean_temp", "max_temp", "min_temp"],
    "Климат_осадки": ["rf_sum"],
    "ERA5_влажность_почвы": ["swc_0_7cm", "swc_7_28cm", "swc_28_100cm"],
    "ERA5_снег": ["snow_depth_cm", "snow_water_eq_mm"],
    "ERA5_темп_почвы": ["soil_temp_0_7cm", "soil_temp_7_28cm"],
    "ERA5_атмосфера": ["temp_2m_c", "precip_mm", "pet_mm", "solar_rad_jm2"],
    "NDVI": ["ndvi"],
}

_MONTH_SUFFIX = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_RE = re.compile(r"^(?P<var>.+)_(?:m(?P<num>\d\d)|(?P<name>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec))$")


class DatasetHashError(RuntimeError):
    """Файл датасета отличается от зафиксированной версии."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path = DEFAULT_PATH) -> None:
    """Проверяет, что датасет — та самая версия. Иначе DatasetHashError."""
    actual = sha256(path)
    if actual != DATASET_SHA256:
        raise DatasetHashError(
            f"Датасет {path} изменился: SHA-256 {actual}, ожидалось {DATASET_SHA256}. "
            "Если обновление намеренное — обновите DATASET_SHA256 и запишите решение в docs/DECISIONS.md."
        )


def _read(path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    df = df.dropna(subset=[YEAR_COL])
    df[YEAR_COL] = df[YEAR_COL].astype(int)
    return df.set_index(YEAR_COL).sort_index()


def load_annual(path: Path = DEFAULT_PATH, check: bool = True) -> pd.DataFrame:
    """Годовая таблица: лист «Сводный_ML», индекс — год."""
    if check:
        verify(path)
    df = _read(path, "Сводный_ML")
    return df.apply(pd.to_numeric, errors="coerce")


@dataclass(frozen=True)
class MonthlyData:
    """Помесячные ряды.

    values: float32 [n_years, 12, n_vars], пропуски — NaN;
    mask:   bool    [n_years, 12, n_vars], True — значение есть;
    years:  int     [n_years];
    variables: имена переменных в порядке последней оси.
    """

    values: np.ndarray
    mask: np.ndarray
    years: np.ndarray
    variables: list[str]

    def up_to_month(self, cutoff: int) -> "MonthlyData":
        """Внутрисезонный срез: оставить только месяцы 1..cutoff (прогноз «по состоянию на» конец месяца)."""
        if not 1 <= cutoff <= 12:
            raise ValueError("cutoff — номер месяца 1..12")
        values = self.values.copy()
        mask = self.mask.copy()
        values[:, cutoff:, :] = np.nan
        mask[:, cutoff:, :] = False
        return MonthlyData(values, mask, self.years, self.variables)


def load_monthly(path: Path = DEFAULT_PATH, check: bool = True) -> MonthlyData:
    """Собирает помесячный тензор из всех листов MONTHLY_SHEETS."""
    if check:
        verify(path)
    frames: dict[str, pd.DataFrame] = {}
    for sheet, variables in MONTHLY_SHEETS.items():
        df = _read(path, sheet)
        for col in df.columns:
            m = _MONTH_RE.match(str(col))
            if not m or m["var"] not in variables:
                continue
            month = int(m["num"]) if m["num"] else _MONTH_SUFFIX[m["name"]]
            frames.setdefault(m["var"], pd.DataFrame(index=df.index, columns=range(1, 13), dtype=float))
            frames[m["var"]][month] = pd.to_numeric(df[col], errors="coerce")

    variables = [v for vs in MONTHLY_SHEETS.values() for v in vs]
    missing = [v for v in variables if v not in frames]
    if missing:
        raise ValueError(f"В датасете не найдены помесячные переменные: {missing}")

    years = sorted(set().union(*(f.index for f in frames.values())))
    cube = np.stack(
        [frames[v].reindex(years)[list(range(1, 13))].to_numpy(dtype=np.float32) for v in variables],
        axis=-1,
    )
    return MonthlyData(cube, ~np.isnan(cube), np.asarray(years, dtype=int), variables)
