#!/usr/bin/env bash
# hc_v6_wr queue on dgy (GPU 0) — the CORRECTED warm-restart extension
# of hc_v6, moved off the local 4080 (user request 2026-09-03).
#
# hc_v6_ext (2026-08-31) was meant to warm-restart the LR at 1e-4 but
# the re-arm gate missed the same-size-extension case (ckpt last_epoch
# 832 vs total_updates ~837), so it trained all 600 eps at the floor
# 1.5e-5 — a valid "more budget at tail LR" data point, not the
# intended plasticity test.  With the gate fixed (re-arm on every
# restore; test_rearm_fires_on_same_size_extension), this queue runs
# the intended variant: same init (hc_v6 final.pt — md5-identical on
# dgy and desktop), same recipe/reward (env=train_multiscale_v5),
# seed 44 (fresh worlds, distinct from ext's 43 and hc_v6's 42),
# lr=1e-4 cosine actually applied.
#
# Then canonicalise best+last and run the 10 benchmark parts into the
# idempotent parts tree reports/hvp_v6w_parts/<agent>/<scenario>.
# NO merge here (parts are pulled to the desktop and merged into the
# canonical reports/hvp_eval_v6w there).
#
# uv --no-sync everywhere (deliberate torch downgrade on dgy).
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
Q=reports/hc_v6_wr_dgy_queue.log
INIT=checkpoints/hc_v6/set_transformer_final.pt
SEED=44
EVAL_SEED=20260722
PARTS=reports/hvp_v6w_parts
KEYS="hc_v6_wr hc_v6_wr_last"
SCENARIOS="very_long lifecycle massive_scale baseline small_scale"
say() { echo "$(date -u +%FT%TZ) [hc-v6-wr-dgy] $*" | tee -a "$Q"; }
mkdir -p reports checkpoints "$PARTS"

[ -f "$INIT" ] || { say "ABORT: init checkpoint $INIT missing"; exit 1; }
say "HC V6 WR DGY QUEUE ARMED (pid $$, GPU $CUDA_VISIBLE_DEVICES, seed $SEED, lr 1e-4 WARM RESTART, +600 eps from hc_v6 final)"

# ---------- 1. Train (hc_v6 recipe, 600 more eps) ----------
CKDIR=checkpoints/hc_v6_wr
if [ -f "$CKDIR/set_transformer_final.pt" ]; then
  say "train cached (final.pt exists)"
else
  say "hc_v6_wr training start (600 eps, parallel_envs=5)"
  uv run --no-sync python scripts/train_hydra.py \
    env=train_multiscale_v5 agent=set_transformer_v6 \
    episodes=600 parallel_envs=5 \
    sim_time=275000 sim_time_min=200000 sim_time_max=350000 \
    eval_interval=200 checkpoint_interval=50 seed=$SEED \
    init_checkpoint="$INIT" \
    checkpoint_dir="$CKDIR" \
    agent.params.lr=1e-4 \
    >> reports/train_hc_v6_wr.log 2>&1
  RC=$?
  say "training rc=$RC"
  echo "DONE_TRAIN_HC_V6_WR rc=$RC $(date -u +%FT%TZ)" >> reports/train_hc_v6_wr.log
fi

# ---------- 2. Canonicalise best + last ----------
pick_last() {  # $1 = checkpoint dir
  local d=$1 p
  if [ -f "$d/set_transformer_final.pt" ]; then
    echo "$d/set_transformer_final.pt"
    return
  fi
  p=$(ls -1 "$d"/set_transformer_round*.pt 2>/dev/null | sort | tail -1)
  [ -z "$p" ] && p=$(ls -1 "$d"/set_transformer_ep*.pt 2>/dev/null | sort | tail -1)
  [ -z "$p" ] && p="$d/set_transformer_best.pt"
  echo "$p"
}
if [ ! -f "$CKDIR/set_transformer_best.pt" ] && [ ! -f "$CKDIR/set_transformer_final.pt" ]; then
  say "ABORT: no checkpoints in $CKDIR (training failed — see reports/train_hc_v6_wr.log)"
  exit 1
fi
mkdir -p checkpoints/hc_v6_wr_final
cp "$CKDIR/set_transformer_best.pt" \
   checkpoints/hc_v6_wr_final/set_transformer_best.pt
LAST=$(pick_last "$CKDIR")
cp "$LAST" checkpoints/hc_v6_wr_final/set_transformer_last.pt
say "canonicalised: best + last=$(basename "$LAST")"
say "HC V6 WR TRAIN DONE"

# ---------- 3. Benchmark parts (idempotent) ----------
part() {  # $1 agent, $2 scenario
  local A=$1 S=$2 EXTRA=""
  if [ -s "$PARTS/$A/$S/episodes.csv" ]; then
    say "part $A/$S cached"
    return
  fi
  [ "$S" = massive_scale ] && EXTRA="--steps 25000 --n-eps 3"
  [ "$S" = very_long ] && EXTRA="--record-every 200"
  [ "$S" = lifecycle ] && EXTRA="--record-every 200"
  say "part $A/$S start (GPU $CUDA_VISIBLE_DEVICES)"
  nice -n 10 uv run --no-sync python scripts/eval_human_vs_performance.py \
    --scenario "$S" $EXTRA --agents "$A" --eval-seed "$EVAL_SEED" \
    --out-root "$PARTS/$A" \
    >> "reports/v6w_part_${A}_${S}.log" 2>&1
  say "part $A/$S rc=$?"
}

NJOBS=0
for S in $SCENARIOS; do
  for A in $KEYS; do
    part "$A" "$S" &
    NJOBS=$((NJOBS + 1))
    if [ $((NJOBS % 4)) = 0 ]; then wait; fi
  done
done
wait
for S in $SCENARIOS; do
  for A in $KEYS; do
    [ -s "$PARTS/$A/$S/episodes.csv" ] || say "MISSING part $A/$S (see reports/v6w_part_${A}_${S}.log)"
  done
done
say "HC V6 WR DGY QUEUE DONE"
