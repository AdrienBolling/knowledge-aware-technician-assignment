#!/usr/bin/env bash
# HTT-RL retrained with ONLY the permutation-invariance fix, on serval-paris.
#
# hc_fix changed two things at once (set_positional=false AND the permissive
# action mask) and lost 28-64% against hc_v6, so the fix itself was never
# tested.  This run keeps hc_v6 recipe verbatim -- same reward, world,
# horizons, seed, budget, BC initialisation, and the strict technician mask
# (env=train_multiscale_v5) -- and changes one flag:
#   agent=set_transformer_v6_permfix  ->  set_positional=false
# (no RoPE over the slot axis, so reordering the roster cannot change the
# chosen technician).  Comparable to hc_v6 row for row, and to hc_fix with
# only the mask differing.
#
# Markers: DONE_TRAIN_HCPERM in reports/train_hc_perm.log, and the queue log.
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=2
Q=reports/hc_perm_queue.log
say() { echo "$(date -u +%FT%TZ) [hcperm] $*" | tee -a "$Q"; }
mkdir -p reports checkpoints
say "HC PERM TRAIN ARMED (pid $$, code $(git rev-parse --short HEAD), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

BC=checkpoints/bc_topsis_v6/set_transformer_bc.pt
[ -f "$BC" ] || { say "ABORT: BC init $BC missing"; exit 1; }
say "BC init present: $(du -h "$BC" | cut -f1)"

say "training start (600 eps, parallel_envs=5, seed 42 -- hc_v6 recipe, set_positional=false, strict mask)"
uv run --no-sync python scripts/train_hydra.py \
  env=train_multiscale_v5 agent=set_transformer_v6_permfix \
  episodes=600 parallel_envs=5 \
  sim_time=275000 sim_time_min=200000 sim_time_max=350000 \
  eval_interval=200 checkpoint_interval=50 seed=42 \
  init_checkpoint=$BC \
  checkpoint_dir=checkpoints/hc_perm \
  >> reports/train_hc_perm.log 2>&1
RC=$?
say "training rc=$RC"
echo "DONE_TRAIN_HCPERM rc=$RC $(date -u +%FT%TZ)" >> reports/train_hc_perm.log
[ "$RC" != "0" ] && { say "ABORT: training failed"; exit 1; }

pick_last() {
  local d=$1 p
  [ -f "$d/set_transformer_final.pt" ] && { echo "$d/set_transformer_final.pt"; return; }
  p=$(ls -1 "$d"/set_transformer_round*.pt 2>/dev/null | sort | tail -1)
  [ -z "$p" ] && p=$(ls -1 "$d"/set_transformer_ep*.pt 2>/dev/null | sort | tail -1)
  [ -z "$p" ] && p="$d/set_transformer_best.pt"
  echo "$p"
}
mkdir -p checkpoints/hc_perm_final
cp checkpoints/hc_perm/set_transformer_best.pt checkpoints/hc_perm_final/set_transformer_best.pt
LAST=$(pick_last checkpoints/hc_perm)
cp "$LAST" checkpoints/hc_perm_final/set_transformer_last.pt
say "final ckpts: best + last=$(basename "$LAST")"
say "HC PERM QUEUE DONE"
