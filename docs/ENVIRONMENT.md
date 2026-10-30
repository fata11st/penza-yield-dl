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

## Доступ по SSH с Mac

Через Tailscale, порт **2223** (22 — сервер toraX, 2222 — `ml`), только по ключу. Настроено 30.10.2026.

Ubuntu `penza-dl` — конфигурация до установки пакета, иначе он займёт порт 22:
```bash
sudo mkdir -p /etc/ssh/sshd_config.d
printf 'Port 2223\nPasswordAuthentication no\nKbdInteractiveAuthentication no\nPermitRootLogin no\nAllowUsers pnz\n' | sudo tee /etc/ssh/sshd_config.d/10-penza.conf
sudo apt install -y openssh-server
sudo systemctl daemon-reload && sudo systemctl restart ssh.socket && sudo systemctl enable ssh
```
Отпечаток ключа машины (ED25519): `SHA256:k07WyL/i7icJN+YM8+afCy7x7GTpXWrLqWWJFihTjmU`.
В `~/.ssh/authorized_keys` — ключ Mac `macbook` (ed25519).

PowerShell от администратора — брандмауэр (только домашняя сеть и Tailscale) и задача, которая держит машину запущенной:
```powershell
New-NetFirewallHyperVRule -Name "penza-ssh" -DisplayName "penza-dl SSH (WSL)" -Direction Inbound -VMCreatorId "{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}" -Protocol TCP -LocalPorts 2223 -RemoteAddresses 192.168.0.0/24,100.64.0.0/10 -Action Allow
New-NetFirewallRule -Name "penza-ssh" -DisplayName "penza-dl SSH (WSL)" -Direction Inbound -Protocol TCP -LocalPort 2223 -RemoteAddress 192.168.0.0/24,100.64.0.0/10 -Action Allow -Profile Any
$a = New-ScheduledTaskAction -Execute "conhost.exe" -Argument '--headless wsl.exe -d penza-dl -u root -- bash -lc "exec sleep infinity"'
Register-ScheduledTask -TaskName "penza-dl" -Action $a -Trigger (New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME) -Settings (New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Seconds 0) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)) -Force
```

Mac, `~/.ssh/config`:
```
Host penza
  HostName 100.115.96.127
  Port 2223
  User pnz
  IdentityFile ~/.ssh/id_ed25519
  UseKeychain yes
  AddKeysToAgent yes
```
Перезапуск только этой машины — `wsl --terminate penza-dl`, затем `Start-ScheduledTask -TaskName penza-dl`.

## Ключи моделей

TabPFN (E06): учётная запись Prior Labs, принятая лицензия модели, ключ API. Ключ хранится вне git:
```bash
mkdir -p ~/.config/penza-yield && chmod 700 ~/.config/penza-yield
read -rs -p "TABPFN_TOKEN: " T && printf 'TABPFN_TOKEN=%s\n' "$T" > ~/.config/penza-yield/env && chmod 600 ~/.config/penza-yield/env && unset T
```
`scripts/run_all.sh` подхватывает этот файл. Chronos-2 (E07) ключа не требует.

## Ночной запуск

```bash
python scripts/preflight.py      # при себе: скачивает веса, пробные прогнозы
tmux new -s night
bash scripts/run_all.sh          # E03–E07 и сводка; журналы — experiments/logs/
```

## Соседи

- `wsl --shutdown` — **запрещено**: остановит сервер toraX.
- Порты сервера 22, 80, 443, 5432, 8000, 8080, 8443 и 2222 (`ml`) не занимать; 2223 — SSH этой машины.
- Воркер toraX постоянно держит ~1,7 ГБ видеопамяти.
- Источник сведений о соседях — `fata11st/toraX`, `docs/14_WORKSTATIONS.md`.
