#!/usr/bin/env bash
# HTT-RL retrained with the two defects the review surfaced corrected, on
# serval-paris.  Everything else is hc_v6's recipe verbatim (same reward,
# world, horizons, seed, budget, BC initialisation), so the run is
# comparable to hc_v6 row for row.
#
#   1. permissive action mask (env: mask_unavailable_technicians=false).
#      The strict mask offered only idle technicians, so "wait for the busy
#      expert" -- the trade section 5.3 motivates the action semantics with --
#      was not an available action.  Now only retired slots are masked.
#   2. permutation-invariant cross-slot attention (agent: set_positional=false).
#      _SetEncoder applied RoPE over the SLOT axis, so reordering the roster
#      changed the chosen technician in 53% of decisions on the trained hc_v6
#      checkpoint.  The cross-attention refiner already avoided RoPE for
#      exactly this reason; the set encoder now does too.
#
# Markers: DONE_TRAIN_HCFIX in reports/train_hc_fix.log, and the queue log.
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=2
Q=reports/hc_fix_queue.log
say() { echo "$(date -u +%FT%TZ) [hcfix] $*" | tee -a "$Q"; }
mkdir -p reports checkpoints
say "HC FIX TRAIN ARMED (pid $$, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

BC=checkpoints/bc_topsis_v6/set_transformer_bc.pt
if [ ! -f "$BC" ]; then
  say "ABORT: BC init $BC missing (ship it from the workstation)"
  exit 1
fi
say "BC init present: $(du -h "$BC" | cut -f1)"

say "training start (600 eps, parallel_envs=5, seed 42 -- hc_v6's recipe)"
uv run --no-sync python scripts/train_hydra.py \
  env=train_multiscale_v5_permfix agent=set_transformer_v6_permfix \
  episodes=600 parallel_envs=5 \
  sim_time=275000 sim_time_min=200000 sim_time_max=350000 \
  eval_interval=200 checkpoint_interval=50 seed=42 \
  init_checkpoint=$BC \
  checkpoint_dir=checkpoints/hc_fix \
  >> reports/train_hc_fix.log 2>&1
RC=$?
say "training rc=$RC"
echo "DONE_TRAIN_HCFIX rc=$RC $(date -u +%FT%TZ)" >> reports/train_hc_fix.log
[ "$RC" != "0" ] && { say "ABORT: training failed"; exit 1; }

pick_last() {
  local d=$1 p
  [ -f "$d/set_transformer_final.pt" ] && { echo "$d/set_transformer_final.pt"; return; }
  p=$(ls -1 "$d"/set_transformer_round*.pt 2>/dev/null | sort | tail -1)
  [ -z "$p" ] && p=$(ls -1 "$d"/set_transformer_ep*.pt 2>/dev/null | sort | tail -1)
  [ -z "$p" ] && p="$d/set_transformer_best.pt"
  echo "$p"
}
mkdir -p checkpoints/hc_fix_final
cp checkpoints/hc_fix/set_transformer_best.pt checkpoints/hc_fix_final/set_transformer_best.pt
LAST=$(pick_last checkpoints/hc_fix)
cp "$LAST" checkpoints/hc_fix_final/set_transformer_last.pt
say "final ckpts: best + last=$(basename "$LAST")"
say "HC FIX QUEUE DONE"
