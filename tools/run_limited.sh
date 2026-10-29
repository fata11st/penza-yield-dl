#!/usr/bin/env bash
# Запуск долгой задачи с жёстким лимитом памяти и пониженным приоритетом, чтобы не мешать соседним машинам WSL.
#   tools/run_limited.sh python scripts/<эксперимент>.py
# Лимит памяти берётся из configs/resources.yaml (memory.max_gb). Долгие запуски — внутри tmux.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_GB="$(python -c "import yaml;print(yaml.safe_load(open('$ROOT/configs/resources.yaml'))['memory']['max_gb'])")"
exec systemd-run --user --scope --quiet -p MemoryMax="${MAX_GB}G" -p MemorySwapMax=0 nice -n 10 "$@"
