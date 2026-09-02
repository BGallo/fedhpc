#!/usr/bin/env bash
# Drive the pos_congestion MAIN Pareto-front bisection to completion: repeatedly
# resume map_pareto_frontier() on the same checkpoint until it has zero open
# boxes (provably complete front) or a round cap is hit.
#
# Every round does at least one exact solve, and each solve always makes
# progress (adds a proven point, shrinks a box, or drops a box as
# inconclusive), so this cannot spin — but MAX_ROUNDS is a safety stop.
#
# Safe to start while another resume_map.py for this checkpoint is still
# running: it waits for that process to exit first (never two writers on one
# checkpoint). Safe to kill/restart: every round resumes from the checkpoint.
#
#   arg1: per-round time budget (s, default 14400 = 4h)
#   arg2: per-solve time limit (s, default 3600)
#   arg3: max rounds (default 40)
set -euo pipefail
cd "$(dirname "$0")/.."

ROUND_BUDGET="${1:-14400}"
PER_SOLVE="${2:-3600}"
MAX_ROUNDS="${3:-40}"
INST=data/pos_congestion_known_runtime_offline_10min.json
CKPT=pareto_runs/pos_congestion_10min_map_checkpoint.json

while pgrep -f "resume_map.py .*pos_congestion_10min_map_checkpoint" >/dev/null; do
    echo "[loop] waiting for in-flight resume_map.py to exit ... $(date)"
    sleep 60
done

ckpt_stat () {  # prints "<n_points> <n_boxes> <n_unresolved>", or "- - -" if no checkpoint yet
    [[ -f "$CKPT" ]] || { echo "- - -"; return; }
    uv run python -c "import json; d=json.load(open('$CKPT')); print(len(d['solutions']), len(d['boxes']), d['n_boxes_unresolved'])"
}

for ((round = 0; round < MAX_ROUNDS; round++)); do
    read -r pts boxes unres <<<"$(ckpt_stat)"
    echo "[loop] round $round start $(date) — $pts points, $boxes box(es) open, $unres dropped"
    # boxes == "0" only after a real checkpoint exists and every box has closed;
    # "-" means the checkpoint hasn't been created yet (still need the anchor solves)
    if [[ "$boxes" == "0" ]]; then
        echo "[loop] FRONT COMPLETE — 0 open boxes."
        break
    fi
    STAMP=$(date +%Y%m%d_%H%M%S)
    uv run python pareto_runs/resume_map.py "$INST" "$CKPT" "$ROUND_BUDGET" "$PER_SOLVE" 2 \
        2>&1 | tee "pareto_runs/resume_pc_main_${STAMP}.log"
done

read -r pts boxes unres <<<"$(ckpt_stat)"
echo "[loop] DONE $(date) — $pts points, $boxes box(es) open, $unres dropped"
