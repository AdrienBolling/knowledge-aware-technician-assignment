#!/usr/bin/env bash
# Traditional-baseline queue on dgy: train A2C/GRPO/DQL MLP baselines
# (same corrected world + v5 reward as hc_v6), then benchmark best+last
# into the v6w generation.  Run inside zellij.
#
# The three agents consume the SAME set observation the HTT agent sees,
# flattened into a plain MLP with a padded Discrete(max_techs) masked
# head — the §7.4 "same information, traditional architecture" contrast.
# Training protocol mirrors v6 (600 eps, multiscale scenarios, seed 42)
# minus the HTT-specific pieces: from scratch (no BC — DQN cannot be
# BC-initialized, uniform protocol), per-decision gamma (no semi-MDP),
# no PopArt.  Deliberate deltas from the HTT world, both reviewed:
#   * env=train_multiscale_v5_mlp — RANGE-mode fleets U(4,30): the
#     per-slot MLP heads need every action index in training (ratio
#     mode reaches slots 27-29 with p<2%, leaving untrained head rows
#     that industrial/very_long argmax over).
#   * grpo_mlp trains at the FIXED 275k horizon (train_hydra pins it):
#     its episode-sum outcome z-scored within a group must not encode
#     the exogenous horizon draw.
# Wall-clock estimate: MLP forwards are ~30x cheaper than the set
# transformer — roughly 5-8 h per serial run, less for vec A2C.
#
# GPU plan: waits until GPUs 0/1/2 are all free (lifecycle + ft queues
# may still be running when this is armed), then a2c->GPU0, grpo->GPU1,
# dql->GPU2 in parallel.  TRAD_SKIP_GATE=1 skips the wait (explicit
# co-scheduling authorization); TRAD_GPU_A2C/GRPO/DQL reassign lanes
# when one of the default GPUs is occupied — the MLPs are small enough
# that two serial trainings share a V100 without contention.
#
# uv --no-sync everywhere (dgy torch downgrade).
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export OMP_NUM_THREADS=2
Q=reports/trad_baselines_queue.log
SEED=42
EVAL_SEED=20260722
OUTROOT=reports/hvp_eval_v6w
PARTS=reports/hvp_v6w_parts
KEYS="a2c_mlp a2c_mlp_last grpo_mlp grpo_mlp_last dql_mlp dql_mlp_last"
SCENARIOS="baseline small_scale massive_scale very_long lifecycle"
say() { echo "$(date -u +%FT%TZ) [trad] $*" | tee -a "$Q"; }
mkdir -p reports

say "TRAD BASELINES QUEUE ARMED (pid $$, seed $SEED)"

# ----- gate: wait for all three GPUs to be free ----------------------
if [ "${TRAD_SKIP_GATE:-0}" = 1 ]; then
  say "gate SKIPPED (TRAD_SKIP_GATE=1; lanes a2c=${TRAD_GPU_A2C:-0} grpo=${TRAD_GPU_GRPO:-1} dql=${TRAD_GPU_DQL:-2})"
else
  while true; do
    BUSY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
           | awk -F', ' '$1 <= 2 && $2 > 1000 {n++} END {print n+0}')
    [ "$BUSY" = 0 ] && break
    say "gate: $BUSY of GPUs 0-2 busy — waiting 10 min"
    sleep 600
  done
  say "gate passed: GPUs 0-2 free"
fi

canonicalise() {  # $1 agent_type — last=final; best falls back to final.
  local AT=$1 CKDIR=checkpoints/${1}_v1
  # final.pt is written only when the training loop completes — it is
  # the completion marker the benchmark stage gates on.  No round/ep
  # fallback: a crashed run gets re-run, not benchmarked.
  [ -f "$CKDIR/${AT}_final.pt" ] || return 0
  cp "$CKDIR/${AT}_final.pt" "$CKDIR/${AT}_last.pt"
  [ -f "$CKDIR/${AT}_best.pt" ] || cp "$CKDIR/${AT}_final.pt" "$CKDIR/${AT}_best.pt"
  say "canonicalised $AT (last=final$([ -f "$CKDIR/${AT}_best.pt" ] && echo ", best ok"))"
}

train() {  # $1 agent_type, $2 gpu, $3 parallel_envs
  local AT=$1 GPU=$2 PE=$3
  local CKDIR=checkpoints/${AT}_v1
  if [ -f "$CKDIR/${AT}_final.pt" ]; then
    say "train $AT cached (final.pt exists)"
    canonicalise "$AT"
    return 0
  fi
  say "train $AT start (GPU $GPU, vec$PE)"
  CUDA_VISIBLE_DEVICES=$GPU nice -n 5 uv run --no-sync python scripts/train_hydra.py \
    agent="$AT" env=train_multiscale_v5_mlp \
    episodes=600 parallel_envs="$PE" \
    sim_time=275000 sim_time_min=200000 sim_time_max=350000 \
    eval_interval=200 eval_episodes=5 checkpoint_interval=50 \
    seed=$SEED \
    checkpoint_dir="$CKDIR" \
    > "reports/train_${AT}.log" 2>&1
  local RC=$?
  say "train $AT rc=$RC"
  canonicalise "$AT"
  return $RC
}

train a2c_mlp  "${TRAD_GPU_A2C:-0}" 5 & P_A2C=$!
train grpo_mlp "${TRAD_GPU_GRPO:-1}" 1 & P_GRPO=$!
train dql_mlp  "${TRAD_GPU_DQL:-2}" 1 & P_DQL=$!
FAILED=0
wait "$P_A2C"  || { say "TRAIN FAILED a2c_mlp";  FAILED=1; }
wait "$P_GRPO" || { say "TRAIN FAILED grpo_mlp"; FAILED=1; }
wait "$P_DQL"  || { say "TRAIN FAILED dql_mlp";  FAILED=1; }
say "TRAD TRAIN DONE (failed=$FAILED)"

# ----- benchmark stage: best+last x 4 scenarios into v6w -------------
export CUDA_VISIBLE_DEVICES=0

part() {  # $1 agent key, $2 scenario
  local A=$1 S=$2 EXTRA=""
  local AT=${A%_last}
  local TAG=best; [ "$A" != "$AT" ] && TAG=last
  # Gate on completion (final.pt) AND the key's own checkpoint file —
  # a crashed training must not be benchmarked, and a missing best
  # must not silently drop the last key (or vice versa).
  if [ ! -f "checkpoints/${AT}_v1/${AT}_final.pt" ]; then
    say "part $A/$S SKIPPED (training incomplete — no final.pt)"
    return
  fi
  if [ ! -f "checkpoints/${AT}_v1/${AT}_${TAG}.pt" ]; then
    say "part $A/$S SKIPPED (missing ${AT}_${TAG}.pt)"
    return
  fi
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
    if [ $((NJOBS % 6)) = 0 ]; then wait; fi
  done
done
wait
say "all trad parts done"

for S in $SCENARIOS; do
  nice -n 10 uv run --no-sync python scripts/merge_hvp_parts.py \
    --parts "$PARTS" --dest "$OUTROOT/$S" --scenario "$S" \
    >> reports/trad_merge.log 2>&1
  say "merge $S rc=$?"
done
nice -n 10 uv run --no-sync python scripts/analyze_hvp_results.py \
  --root "$OUTROOT" >> reports/trad_merge.log 2>&1
say "analysis rc=$?"
say "TRAD BASELINES DONE"
