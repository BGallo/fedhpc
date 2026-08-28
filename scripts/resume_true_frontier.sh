#!/usr/bin/env bash
# Resume the exact true-frontier exploration for fedhpc_known_runtime_offline_10min.json
# from both checkpoints, bisecting the remaining open boxes.
#   arg1: time budget per checkpoint (seconds, default 3600)
set -euo pipefail
cd "$(dirname "$0")/.."

BUDGET="${1:-3600}"
INST=data/fedhpc_known_runtime_offline_10min.json
STAMP=$(date +%Y%m%d_%H%M%S)

echo "=== MAIN region  ($(date))  budget=${BUDGET}s ==="
uv run python pareto_runs/resume_map.py "$INST" \
    pareto_runs/fedhpc_10min_map_checkpoint.json "$BUDGET" 200 3 \
    2>&1 | tee "pareto_runs/resume_main_${STAMP}.log"

echo
echo "=== GAP region  ($(date))  budget=${BUDGET}s ==="
uv run python pareto_runs/resume_map.py "$INST" \
    pareto_runs/fedhpc_10min_gap_checkpoint.json "$BUDGET" 600 3 \
    2>&1 | tee "pareto_runs/resume_gap_${STAMP}.log"

echo
echo "=== DONE  ($(date)) ==="
