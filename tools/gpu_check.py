"""Проверка GPU и того, кто ещё им пользуется. Запуск: python tools/gpu_check.py"""
import subprocess

import torch

print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("✗ CUDA недоступна")
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
free, total = torch.cuda.mem_get_info(0)
print(f"✓ {name}, compute capability {cap}, свободно {free / 2**30:.1f} из {total / 2**30:.1f} ГБ")
x = torch.randn(2048, 2048, device="cuda")
torch.cuda.synchronize()
print(f"✓ умножение матриц на GPU: {(x @ x).float().abs().mean().item():.3f}")
try:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    print("Процессы на GPU (воркер toraX — норма):\n" + (out or "  нет"))
except FileNotFoundError:
    print("nvidia-smi не найден: проверьте /usr/lib/wsl/lib в PATH (/etc/environment)")
