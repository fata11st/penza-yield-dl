"""Сводка по датасету: версия, годы, целевые переменные, пропуски в помесячных рядах.

Запуск: python scripts/check_data.py
"""
from penza_yield import data

data.verify()
print(f"✓ Датасет {data.DEFAULT_PATH.name}, SHA-256 совпадает")

annual = data.load_annual(check=False)
print(f"\nГодовая таблица: {annual.shape[0]} лет ({annual.index.min()}–{annual.index.max()}), "
      f"{annual.shape[1]} столбцов")
print("\nУрожайность, т/га (мин / медиана / макс / последний год):")
for t in data.TARGETS:
    s = annual[t]
    print(f"  {data.CROP_NAMES_RU[t]:<26} {s.min():6.2f} {s.median():6.2f} {s.max():6.2f} {s.iloc[-1]:6.2f}")

m = data.load_monthly(check=False)
print(f"\nПомесячный тензор: {m.values.shape} (годы × месяцы × переменные)")
print("Переменные с пропусками (годы без единого значения):")
for i, v in enumerate(m.variables):
    empty = [int(y) for y, row in zip(m.years, m.mask[:, :, i]) if not row.any()]
    partial = int((~m.mask[:, :, i]).sum())
    if partial:
        span = f"{empty[0]}–{empty[-1]}" if empty else "—"
        print(f"  {v:<18} пустых месяцев {partial:3d}, пустые годы: {span}")
