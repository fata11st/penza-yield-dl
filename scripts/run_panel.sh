#!/usr/bin/env bash
# Панель 10 регионов (D11): все модели P01 по очереди, затем ансамбли P02 и сводка P03.
#   tmux new -s panel
#   bash scripts/run_panel.sh
# Сбой одной модели не останавливает остальные; журналы — experiments/logs/panel_*.log.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
LOGS="$ROOT/experiments/logs"
mkdir -p "$LOGS"
STATUS=()
run() {
  local name="$1"; shift
  echo -e "\n===== $name: $(date '+%H:%M:%S') =====" | tee -a "$LOGS/run_panel.log"
  "$@" 2>&1 | tee "$LOGS/panel_$name.log"
  local code=${PIPESTATUS[0]}
  STATUS+=("$name: код $code")
  echo "----- $name завершён с кодом $code ($(date '+%H:%M:%S'))" | tee -a "$LOGS/run_panel.log"
}

START=$(date +%s)
python -m pytest -q tests/test_panel.py || { echo "Тесты панели не прошли — запуск остановлен"; exit 1; }
for m in baselines ridge gbm mlp gru chronos; do
  run "$m" python scripts/p01_panel_models.py --model "$m"
done
run ensembles python scripts/p02_ensembles.py
run summary   python scripts/p03_summary.py

echo -e "\n===== ИТОГ ($(( ($(date +%s) - START) / 60 )) мин) =====" | tee -a "$LOGS/run_panel.log"
printf '%s\n' "${STATUS[@]}" | tee -a "$LOGS/run_panel.log"
