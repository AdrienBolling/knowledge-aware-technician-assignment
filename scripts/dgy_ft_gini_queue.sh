#!/usr/bin/env bash
# ft_gini fine-tune queue on dgy — one-lever knowledge-concentration
# variant, then benchmark best+last into the v6w generation.  Run
# inside zellij.
#
# Motivated by the 2026-08-14 lifecycle key-person analysis: the top-1
# technician holds 20-26% of total fleet knowledge under EVERY policy
# (rich-get-richer assignment + template hierarchy), which is exactly
# what select=highest_knowledge retirements punish.  This variant
# prices the distribution itself:
#
#   ft_gini : knowledge_gini enabled, coefficient 5.0
#             (raw = -Gini of per-tech knowledge volumes; running-
#             normalized like every component, so 5.0 is a ~40% share
#             of the enabled mix — deliberately heavy, per the ask)
#
# Same protocol as the other ft variants: init hc_v6 final.pt, 100 eps
# vec5, seed 4242 (same world sequence), ctor lr 3e-5 cosine.  The new
# knowledge_gini STEP metric is recorded by the (post-fix) harness, so
# concentration becomes observable over time in steps.csv.gz.
#
# Success criteria vs hc_v6: lower knowledge_gini trajectory, smaller
# implied-retiree knowledge at the lifecycle 0.8M/2.5M/4.2M events,
# knowledge level + products held.
#
# GPU plan: gates on `V6 FT QUEUE DONE` (GPU 1 frees then).
# uv --no-sync everywhere (dgy torch downgrade).
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export OMP_NUM_THREADS=2
Q=reports/ft_gini_queue.log
INIT=checkpoints/hc_v6/set_transformer_final.pt
SEED=4242
EVAL_SEED=20260722
OUTROOT=reports/hvp_eval_v6w
PARTS=reports/hvp_v6w_parts
KEYS="ft_gini ft_gini_last"
SCENARIOS="baseline small_scale massive_scale very_long lifecycle"
GPU="${FT_GINI_GPU:-1}"
say() { echo "$(date -u +%FT%TZ) [ft-gini] $*" | tee -a "$Q"; }
mkdir -p reports

[ -f "$INIT" ] || { say "ABORT: init checkpoint $INIT missing"; exit 1; }
say "FT GINI QUEUE ARMED (pid $$, GPU $GPU, seed $SEED, coeff 5.0)"

if [ "${FT_GINI_SKIP_GATE:-0}" = 1 ]; then
  say "gate SKIPPED (FT_GINI_SKIP_GATE=1)"
else
  until grep -q "V6 FT QUEUE DONE" reports/v6_ft_queue.log 2>/dev/null; do
    say "gate: waiting for V6 FT QUEUE DONE (GPU $GPU busy) — 15 min"
    sleep 900
  done
  say "gate passed: v6 ft queue done"
fi

CKDIR=checkpoints/ft_gini
if [ -f "$CKDIR/set_transformer_final.pt" ]; then
  say "train cached (final.pt exists)"
else
  say "train start (GPU $GPU)"
  CUDA_VISIBLE_DEVICES=$GPU nice -n 5 uv run --no-sync python scripts/train_hydra.py \
    env=train_multiscale_v5 agent=set_transformer_v6 \
    episodes=100 parallel_envs=5 \
    sim_time=275000 sim_time_min=200000 sim_time_max=350000 \
    eval_interval=50 eval_episodes=5 checkpoint_interval=50 \
    seed=$SEED \
    init_checkpoint="$INIT" \
    checkpoint_dir="$CKDIR" \
    agent.params.lr=3e-5 \
    '+env.gym.reward.knowledge_gini.enabled=true' \
    '+env.gym.reward.knowledge_gini.coefficient=5.0' \
    > reports/train_ft_gini.log 2>&1
  say "train rc=$?"
fi
if [ -f "$CKDIR/set_transformer_final.pt" ]; then
  cp "$CKDIR/set_transformer_final.pt" "$CKDIR/set_transformer_last.pt"
  say "canonicalised (last=final)"
else
  say "TRAIN FAILED — no final.pt; stopping before benchmarks"
  exit 1
fi
say "FT GINI TRAIN DONE"

export CUDA_VISIBLE_DEVICES=$GPU

part() {  # $1 agent, $2 scenario
  local A=$1 S=$2 EXTRA=""
  if [ -s "$PARTS/$A/$S/episodes.csv" ]; then
    say "part $A/$S cached"
    return
  fi
  [ "$S" = massive_scale ] && EXTRA="--steps 25000 --n-eps 3"
  [ "$S" = very_long ] && EXTRA="--record-every 200"
  [ "$S" = lifecycle ] && EXTRA="--record-every 200"
  say "part $A/$S start"
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
say "all ft_gini parts done"

for S in $SCENARIOS; do
  nice -n 10 uv run --no-sync python scripts/merge_hvp_parts.py \
    --parts "$PARTS" --dest "$OUTROOT/$S" --scenario "$S" \
    >> reports/ft_gini_merge.log 2>&1
  say "merge $S rc=$?"
done
nice -n 10 uv run --no-sync python scripts/analyze_hvp_results.py \
  --root "$OUTROOT" >> reports/ft_gini_merge.log 2>&1
say "analysis rc=$?"
say "FT GINI QUEUE DONE"
