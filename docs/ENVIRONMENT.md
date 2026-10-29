# Окружение

Настроено 29.10.2026. Все шаги — на ПК `Zver`.

## Машина

Windows 11, AMD Ryzen 9 9900X (12 ядер / 24 потока), 64 ГБ, RTX 5070 Ti 16 ГБ (Blackwell, sm_120),
драйвер Windows 617.42 (CUDA драйвера 13.4). WSL 3.0.3.

## WSL

Три дистрибутива в одной виртуальной машине:

| Дистрибутив | Назначение | Пользователь |
|---|---|---|
| `Ubuntu-24.04` | сервер toraX — не трогать | root |
| `ml` | эксперименты ml-lab | max |
| `penza-dl` | этот проект | pnz |

`C:\Users\guliv\.wslconfig` — **общий**, не менять:
```ini
[wsl2]
memory=40GB
swap=8GB
networkingMode=mirrored
vmIdleTimeout=-1
[experimental]
autoMemoryReclaim=gradual
```

## Как создавался `penza-dl`

PowerShell:
```powershell
wsl --install Ubuntu-24.04 --name penza-dl
```

Ubuntu `penza-dl`:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl wget unzip htop
sudo sh -c 'printf "[boot]\nsystemd=true\n\n[network]\ngenerateResolvConf=false\n\n[user]\ndefault=pnz\n\n[interop]\nenabled=false\nappendWindowsPath=false\n" > /etc/wsl.conf'
sudo sed -i 's#^PATH="\(.*\)"#PATH="\1:/usr/lib/wsl/lib"#' /etc/environment
```

PowerShell (перезапуск только своей машины):
```powershell
wsl --terminate penza-dl
```

Ubuntu `penza-dl` — DNS (иначе WSL перезаписывает `resolv.conf` и `pip` не находит серверы):
```bash
sudo rm -f /etc/resolv.conf
sudo sh -c 'printf "nameserver 77.88.8.8\nnameserver 1.1.1.1\n" > /etc/resolv.conf'
```

Python и PyTorch:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
mkdir -p ~/projects/penza-yield-dl && cd ~/projects/penza-yield-dl
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
```
Результат: torch 2.14.1+cu130, CUDA 13.0, `(12, 0)`, GPU работает.

Доступ к GitHub — отдельный ключ развёртывания (с правом записи), хост `github-penza`:
```bash
ssh-keygen -t ed25519 -C "penza-dl@Zver" -f ~/.ssh/github_penza -N ""
printf "Host github-penza\n  HostName github.com\n  User git\n  IdentityFile ~/.ssh/github_penza\n  IdentitiesOnly yes\n" >> ~/.ssh/config && chmod 600 ~/.ssh/config
```
Отпечаток GitHub (ED25519): `SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU`.

## Соседи

- `wsl --shutdown` — **запрещено**: остановит сервер toraX.
- Порты сервера 22, 80, 443, 5432, 8000, 8080, 8443 и 2222 (`ml`) не занимать.
- Воркер toraX постоянно держит ~1,7 ГБ видеопамяти.
- Источник сведений о соседях — `fata11st/toraX`, `docs/14_WORKSTATIONS.md`.
