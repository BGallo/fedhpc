#!/usr/bin/env bash
# Exact true-Pareto-front mapping for pos_congestion_known_runtime_offline_10min.json
# (3340 window jobs + 2617 running jobs, 29 types — pos/CSR1 congestion event).
#
# Bisection exploration via map_pareto_frontier(), same machinery as
# scripts/resume_true_frontier.sh (the fedhpc 964-job 10min front) but with a
# COMPLETELY SEPARATE set of checkpoints and logs:
#
#     pareto_runs/pos_congestion_10min_map_checkpoint.json   <- MAIN region restart file
#     pareto_runs/pos_congestion_10min_gap_checkpoint.json   <- GAP region restart file (seeded later)
#     pareto_runs/resume_pc_main_*.log  /  resume_pc_gap_*.log
#
# so this front and the fedhpc front can be worked on at the same time without
# any file contention. See pareto_runs/STATUS_pos_congestion.md before resuming.
#
#   arg1: time budget per checkpoint (seconds, default 3600)
#   arg2: region — "main" | "gap" | "both"  (default "both")
#         "gap" is skipped until pareto_runs/seed_gap_checkpoint.py has carved it
#         off the MAIN checkpoint (needs anchor B + a mid-front point to exist).
set -euo pipefail
cd "$(dirname "$0")/.."

BUDGET="${1:-3600}"
REGION="${2:-both}"
INST=data/pos_congestion_known_runtime_offline_10min.json
MAIN_CKPT=pareto_runs/pos_congestion_10min_map_checkpoint.json
GAP_CKPT=pareto_runs/pos_congestion_10min_gap_checkpoint.json
STAMP=$(date +%Y%m%d_%H%M%S)

run_region () {
    local ckpt="$1" per_solve="$2" tag="$3"
    echo "=== pos_congestion ${tag}  ($(date))  budget=${BUDGET}s  per_solve=${per_solve}s ==="
    uv run python pareto_runs/resume_map.py "$INST" "$ckpt" "$BUDGET" "$per_solve" 3 \
        2>&1 | tee "pareto_runs/resume_pc_${tag}_${STAMP}.log"
    echo
}

# per-solve limits are large: this MIP is ~25.6M binary vars / 98M nonzeros
# (SpaceTimeFormulation, 3340+2617 jobs x 29 types x 288 slots). Presolve alone
# is ~70s and a 90s probe solve found no feasible point — see STATUS_pos_congestion.md.
if [[ "$REGION" == "main" || "$REGION" == "both" ]]; then
    run_region "$MAIN_CKPT" "${PC_MAIN_PER_SOLVE:-3600}" main
fi

if [[ "$REGION" == "gap" || "$REGION" == "both" ]]; then
    if [[ -f "$GAP_CKPT" ]]; then
        run_region "$GAP_CKPT" "${PC_GAP_PER_SOLVE:-3600}" gap
    else
        echo "=== pos_congestion GAP — $GAP_CKPT not seeded yet, skipping ==="
        echo "    seed it once the MAIN checkpoint has a sparse tail:"
        echo "    uv run python pareto_runs/seed_gap_checkpoint.py \\"
        echo "        $MAIN_CKPT $GAP_CKPT <f1_split>"
    fi
fi

echo "=== DONE  ($(date)) ==="
