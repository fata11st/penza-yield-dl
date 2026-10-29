"""Проверка перед ночным запуском — выполнять, пока вы у компьютера.

Скачивает веса Chronos-2 и TabPFN (если задан ключ) и делает по одному пробному прогнозу, чтобы ночью ничего не
упёрлось в сеть или авторизацию. Запуск: python scripts/preflight.py
"""
import os
import sys
import time
from pathlib import Path

ok = True

# Ключи моделей — тот же файл, что подхватывает run_all.sh
_env = Path.home() / ".config" / "penza-yield" / "env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def step(name, fn):
    global ok
    t0 = time.time()
    try:
        msg = fn()
        print(f"✓ {name} ({time.time() - t0:.0f} с){': ' + msg if msg else ''}", flush=True)
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"✗ {name}: {type(e).__name__}: {e}", flush=True)


def gpu():
    import torch

    free, total = torch.cuda.mem_get_info(0)
    return f"{torch.cuda.get_device_name(0)}, свободно {free / 2**30:.1f} ГБ"


def chronos():
    import importlib.util

    import torch
    from chronos import Chronos2Pipeline

    spec = importlib.util.spec_from_file_location("e07", Path(__file__).with_name("e07_chronos.py"))
    e07 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e07)
    from penza_yield.tabular import crop_frames

    pipe = Chronos2Pipeline.from_pretrained(e07.MODEL_ID, device_map="cuda" if torch.cuda.is_available() else "cpu")
    ctx, fut = e07.frames_for_year(crop_frames(), 2015, True)
    fc = pipe.predict_df(ctx, future_df=fut, prediction_length=1, quantile_levels=[0.5], id_column="item_id",
                         timestamp_column="timestamp", target="target", freq="YS")
    return f"прогноз на 2015 г. получен для {len(fc)} культур"


def tabpfn():
    if not os.environ.get("TABPFN_TOKEN"):
        raise RuntimeError("TABPFN_TOKEN не задан — E06 ночью будет пропущен (остальное пойдёт)")
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    from penza_yield.tabular import FEATURE_SETS, crop_frames, fit_predict_tabpfn

    fr = crop_frames()["p_grain_crops"]
    p = fit_predict_tabpfn(fr, 2015, FEATURE_SETS["ar_climate_ndvi"], [0])
    return f"пробный прогноз {p:.2f} т/га"


step("GPU", gpu)
step("Chronos-2: загрузка весов и пробный прогноз", chronos)
step("TabPFN: ключ, загрузка весов и пробный прогноз", tabpfn)
print("\nВсё готово к ночному запуску." if ok else "\nЕсть проблемы (см. выше). Неготовые шаги ночью пропустятся.")
sys.exit(0 if ok else 1)
