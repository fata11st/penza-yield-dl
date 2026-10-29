"""Метаданные запуска для воспроизводимости: коммит, версии пакетов, GPU, хеш датасета."""
from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ["torch", "numpy", "pandas", "scikit-learn", "scipy", "statsmodels"]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return ""


def collect(**extra) -> dict:
    from penza_yield import data

    meta = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "packages": {},
        "dataset_sha256": data.sha256(data.DEFAULT_PATH),
    }
    for p in PACKAGES:
        try:
            meta["packages"][p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
            meta["cuda"] = torch.version.cuda
    except ImportError:
        pass
    meta.update(extra)
    return meta

