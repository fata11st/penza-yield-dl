#!/usr/bin/env bash
# Ночной запуск E03–E07 и сводки — по очереди; сбой одного шага не останавливает остальные.
#   tmux new -s night
#   bash scripts/run_all.sh
# Журналы: experiments/logs/, итог — experiments/SUMMARY/.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
# Ключи моделей (TABPFN_TOKEN) — вне git
if [ -f "$HOME/.config/penza-yield/env" ]; then set -a; . "$HOME/.config/penza-yield/env"; set +a; fi

LOGS="$ROOT/experiments/logs"
mkdir -p "$LOGS"
STATUS=()
run() {
  local name="$1"; shift
  echo -e "\n===== $name: $(date '+%H:%M:%S') =====" | tee -a "$LOGS/run_all.log"
  "$@" 2>&1 | tee "$LOGS/$name.log"
  local code=${PIPESTATUS[0]}
  STATUS+=("$name: код $code")
  echo "----- $name завершён с кодом $code ($(date '+%H:%M:%S'))" | tee -a "$LOGS/run_all.log"
}

START=$(date +%s)
run E03-gru-compact  python scripts/e03_e04_compact.py --arch gru
run E04-cnn-compact  python scripts/e03_e04_compact.py --arch cnn
run E05-mlp          python scripts/e05_mlp.py
run E06-tabpfn       python scripts/e06_tabpfn.py
run E07-chronos2     python scripts/e07_chronos.py
run SUMMARY          python scripts/summary_all.py

echo -e "\n===== ИТОГ ($(( ($(date +%s) - START) / 60 )) мин) =====" | tee -a "$LOGS/run_all.log"
printf '%s\n' "${STATUS[@]}" | tee -a "$LOGS/run_all.log"
echo "Код 3 у E06 — нет ключа TabPFN (шаг пропущен намеренно)." | tee -a "$LOGS/run_all.log"
