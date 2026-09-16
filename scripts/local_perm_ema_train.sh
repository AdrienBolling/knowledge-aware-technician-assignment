#!/usr/bin/env bash
# HTT-RL with the permutation-invariance fix AND a frozen-rate memory, on
# the local 4080.  Companion of scripts/paris_perm_train.sh (fix only): the
# two runs share hc_v6's recipe verbatim (reward, world, horizons, seed,
# budget, BC initialisation, strict technician mask) and differ only in the
# memory, so comparing them isolates what memory adds.
#
# Memory (agent=set_transformer_v6_perm_ema, rnn_type="ema"): five
# exponential moving averages of the pooled context with half-lives of
# 60 / 360 / 1440 / 5760 / 20160 t.u. (1 h, 6 h, 1 day, 4 days, 2 weeks of
# plant time at 1 t.u. = 1 min), decayed by exp(-dt/tau) on simulated time.
# Rates are frozen; the read projection starts at zero, so the run starts
# from exactly the memoryless BC policy.
#
# OUT_ROOT (default: this checkout) receives checkpoints and logs, so the
# script can run from a separate worktree while the benchmark harness finds
# the checkpoints in the main checkout.  PY defaults to OUT_ROOT's venv.
#
# Markers: DONE_TRAIN_HCPERMEMA in reports/train_hc_perm_ema.log, and the
# queue log.
set -u
cd "$(dirname "$0")/.."
CODE=$(pwd)
OUT_ROOT="${OUT_ROOT:-$CODE}"
PY="${PY:-$OUT_ROOT/.venv/bin/python}"
export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=2
mkdir -p "$OUT_ROOT/reports" "$OUT_ROOT/checkpoints"
Q="$OUT_ROOT/reports/hc_perm_ema_queue.log"
LOG="$OUT_ROOT/reports/train_hc_perm_ema.log"
say() { echo "$(date -u +%FT%TZ) [hcpermema] $*" | tee -a "$Q"; }
say "HC PERM EMA TRAIN ARMED (pid $$, code $(git -C "$CODE" rev-parse --short HEAD), out $OUT_ROOT)"

BC="$OUT_ROOT/checkpoints/bc_topsis_v6/set_transformer_bc.pt"
[ -f "$BC" ] || { say "ABORT: BC init $BC missing"; exit 1; }
say "BC init present: $(md5sum "$BC" | cut -c1-12)"

say "training start (600 eps, parallel_envs=5, seed 42 -- hc_v6 recipe, set_positional=false, frozen-rate memory)"
"$PY" "$CODE/scripts/train_hydra.py" \
  env=train_multiscale_v5 agent=set_transformer_v6_perm_ema \
  episodes=600 parallel_envs=5 \
  sim_time=275000 sim_time_min=200000 sim_time_max=350000 \
  eval_interval=200 checkpoint_interval=50 seed=42 \
  init_checkpoint="$BC" \
  checkpoint_dir="$OUT_ROOT/checkpoints/hc_perm_ema" \
  hydra.run.dir="$OUT_ROOT/outputs/hc_perm_ema" \
  >> "$LOG" 2>&1
RC=$?
say "training rc=$RC"
echo "DONE_TRAIN_HCPERMEMA rc=$RC $(date -u +%FT%TZ)" >> "$LOG"
[ "$RC" != "0" ] && { say "ABORT: training failed"; exit 1; }

D="$OUT_ROOT/checkpoints/hc_perm_ema"
F="$OUT_ROOT/checkpoints/hc_perm_ema_final"
mkdir -p "$F"
cp "$D/set_transformer_best.pt" "$F/set_transformer_best.pt"
LAST="$D/set_transformer_final.pt"
[ -f "$LAST" ] || LAST=$(ls -1 "$D"/set_transformer_round*.pt 2>/dev/null | sort | tail -1)
cp "$LAST" "$F/set_transformer_last.pt"
say "final ckpts: best + last=$(basename "$LAST")"
say "HC PERM EMA QUEUE DONE"
