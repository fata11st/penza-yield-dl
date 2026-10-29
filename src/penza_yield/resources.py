"""Ограничение ресурсов, чтобы не мешать соседним машинам WSL (сервер toraX и `ml`).

Все три машины WSL — в одной виртуальной машине: общие 40 ГБ ОЗУ, 24 потока CPU и одна RTX 5070 Ti,
на которой постоянно работает воркер toraX (~1,7 ГБ). Лимиты — в configs/resources.yaml.
Вызывайте ``apply_limits()`` в начале каждого скрипта ДО импорта тяжёлых библиотек.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "resources.yaml"


def load_limits(path: Path = CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_limits(path: Path = CONFIG) -> dict:
    """Ставит лимиты потоков, приоритет процесса и долю видеопамяти. Возвращает словарь лимитов."""
    cfg = load_limits(path)
    threads = str(cfg["cpu"]["threads_per_process"])
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, threads)

    nice = int(cfg["cpu"].get("nice", 0))
    if nice:
        try:
            current = os.nice(0)
            if current < nice:
                os.nice(nice - current)
        except OSError:
            pass

    try:
        import torch
    except ImportError:
        return cfg
    torch.set_num_threads(int(threads))
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(cfg["gpu"]["memory_fraction"]), 0)
    return cfg


def n_jobs() -> int:
    """Число параллельных процессов joblib."""
    return int(load_limits()["cpu"]["n_jobs"])
