#!/usr/bin/env bash
# Lifecycle-scenario parts on the LOCAL machine (gourmetdesk, RTX 4080)
# — companion to dgy_v6w_lifecycle.sh for agents whose dgy queues don't
# cover the lifecycle scenario (ft_* fine-tunes; trad MLPs are handled
# on dgy).  Same seed / flags / parts layout, so the resulting parts
# ship to dgy's reports/hvp_v6w_parts tree and merge idempotently.
#
# Usage: scripts/local_lifecycle_parts.sh ft_protect ft_protect_last ...
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export OMP_NUM_THREADS=2
Q=reports/local_lifecycle_parts.log
SEED=20260722
PARTS=reports/hvp_v6w_parts
MAXPAR=4
say() { echo "$(date -u +%FT%TZ) [local-lc] $*" | tee -a "$Q"; }
mkdir -p reports

[ $# -ge 1 ] || { say "no agents given"; exit 1; }
say "LOCAL LIFECYCLE PARTS ARMED (pid $$): $*"

part() {  # $1 agent key
  local A=$1
  if [ -s "$PARTS/$A/lifecycle/episodes.csv" ]; then
    say "part $A/lifecycle cached"
    return
  fi
  say "part $A/lifecycle start"
  nice -n 10 uv run python scripts/eval_human_vs_performance.py \
    --scenario lifecycle --record-every 200 --agents "$A" \
    --eval-seed "$SEED" --out-root "$PARTS/$A" \
    >> "reports/v6w_part_${A}_lifecycle.log" 2>&1
  say "part $A/lifecycle rc=$?"
}

NJOBS=0
for A in "$@"; do
  part "$A" &
  NJOBS=$((NJOBS + 1))
  if [ $((NJOBS % MAXPAR)) = 0 ]; then wait; fi
done
wait
say "LOCAL LIFECYCLE PARTS DONE: $*"
