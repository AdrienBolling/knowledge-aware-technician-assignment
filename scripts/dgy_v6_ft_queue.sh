#!/usr/bin/env bash
# v6 reward fine-tune queue on dgy — train 3 one-lever reward variants,
# then benchmark them into the v6w generation.  Run inside zellij.
#
# Motivated by the v6w read: throughput learned (#1 at 5M) but fleet
# protection NOT (illness mid-pack, availability ~0.005 below best,
# quality weakest KPI 0.26-0.39).  One lever per variant, same seed =
# same world sequence, so behaviour deltas are attributable:
#
#   ft_protect : fleet_availability 0.5->2.0, workload_balance 0.5->1.0
#                (price the SYMPTOM: availability dips + imbalance)
#   ft_fatigue : fatigue_cost 1.0->2.5
#                (price the CAUSE: fatigue drives exhaustion->illness)
#   ft_quality : repair_quality 1.0->2.5
#                (price the weakest KPI: skill-match quality)
#
# Components are running-normalized, so coefficients are comparable
# shares of the reward mix — 2-2.5x is firm but not dominating.
#
# Fine-tune regime: init from hc_v6 final.pt, 100 episodes vec5, exact
# v5/v6 world + recipe otherwise.  LR: ctor 3e-5 (0.1x v6 peak),
# warmup 10 updates, cosine to the 0.05 floor (1.5e-6); the on-load
# re-arm uses the CONSTRUCTOR lr, so the checkpoint's 3e-4 base_lrs
# cannot leak in.
#
# Benchmark stage: best+last of each variant x 4 standard scenarios,
# same eval seed / parts tree / merge as the v6w generation, so the ft
# rows land beside hc_v6 in reports/hvp_eval_v6w.
#
# GPU plan: lifecycle benchmark owns GPU 0.  Lane A (GPU 1) trains
# protect then quality; lane B (GPU 2) trains fatigue.  The benchmark
# stage runs on GPU 1.  ~3.3 h/variant train + ~8-10 h benchmarks.
#
# uv --no-sync everywhere (dgy torch downgrade).
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export OMP_NUM_THREADS=2
Q=reports/v6_ft_queue.log
INIT=checkpoints/hc_v6/set_transformer_final.pt
SEED=4242
EVAL_SEED=20260722
OUTROOT=reports/hvp_eval_v6w
PARTS=reports/hvp_v6w_parts
FT_AGENTS="ft_protect ft_protect_last ft_fatigue ft_fatigue_last ft_quality ft_quality_last"
SCENARIOS="baseline small_scale massive_scale very_long"
say() { echo "$(date -u +%FT%TZ) [v6-ft] $*" | tee -a "$Q"; }
mkdir -p reports

if [ ! -f "$INIT" ]; then
  say "ABORT: init checkpoint $INIT missing"
  exit 1
fi
say "V6 FT QUEUE ARMED (pid $$, init $INIT, seed $SEED, 100 eps/variant)"

run_ft() {  # $1 name, $2 gpu, $3... hydra reward overrides
  local NAME=$1 GPU=$2; shift 2
  local CKDIR=checkpoints/ft_${NAME}
  if [ -f "$CKDIR/set_transformer_final.pt" ]; then
    say "ft_$NAME cached (final.pt exists)"
    return 0
  fi
  say "ft_$NAME start (GPU $GPU): $*"
  CUDA_VISIBLE_DEVICES=$GPU nice -n 5 uv run --no-sync python scripts/train_hydra.py \
    env=train_multiscale_v5 agent=set_transformer_v6 \
    episodes=100 parallel_envs=5 \
    sim_time=275000 sim_time_min=200000 sim_time_max=350000 \
    eval_interval=50 eval_episodes=5 checkpoint_interval=50 \
    seed=$SEED \
    init_checkpoint="$INIT" \
    checkpoint_dir="$CKDIR" \
    agent.params.lr=3e-5 \
    "$@" \
    > "reports/train_ft_${NAME}.log" 2>&1
  local RC=$?
  say "ft_$NAME rc=$RC"
  # canonicalise: last = final (vec loop saves final.pt since 255560a)
  if [ -f "$CKDIR/set_transformer_final.pt" ]; then
    cp "$CKDIR/set_transformer_final.pt" "$CKDIR/set_transformer_last.pt"
    say "ft_$NAME canonicalised (last=final)"
  fi
  return $RC
}

(  # lane A — GPU 1
  run_ft protect 1 \
    'env.gym.reward.fleet_availability.coefficient=2.0' \
    'env.gym.reward.workload_balance.coefficient=1.0'
  say "DONE_FT_PROTECT"
  run_ft quality 1 \
    'env.gym.reward.repair_quality.coefficient=2.5'
  say "DONE_FT_QUALITY"
) &
(  # lane B — GPU 2
  run_ft fatigue 2 \
    'env.gym.reward.fatigue_cost.coefficient=2.5'
  say "DONE_FT_FATIGUE"
) &
wait
say "V6 FT TRAIN DONE"

# ----- benchmark stage: ft best+last into the v6w generation ---------
export CUDA_VISIBLE_DEVICES=1

part() {  # $1 agent, $2 scenario
  local A=$1 S=$2 EXTRA=""
  local CK=${A%_last}
  if [ ! -f "checkpoints/${CK}/set_transformer_best.pt" ]; then
    say "part $A/$S SKIPPED (no checkpoint — training failed?)"
    return
  fi
  if [ -s "$PARTS/$A/$S/episodes.csv" ]; then
    say "part $A/$S cached"
    return
  fi
  [ "$S" = massive_scale ] && EXTRA="--steps 25000 --n-eps 3"
  [ "$S" = very_long ] && EXTRA="--record-every 200"
  say "part $A/$S start"
  nice -n 10 uv run --no-sync python scripts/eval_human_vs_performance.py \
    --scenario "$S" $EXTRA --agents "$A" --eval-seed "$EVAL_SEED" \
    --out-root "$PARTS/$A" \
    >> "reports/v6w_part_${A}_${S}.log" 2>&1
  say "part $A/$S rc=$?"
}

NJOBS=0
for S in $SCENARIOS; do
  for A in $FT_AGENTS; do
    part "$A" "$S" &
    NJOBS=$((NJOBS + 1))
    if [ $((NJOBS % 6)) = 0 ]; then wait; fi
  done
done
wait
say "all ft parts done"

for S in $SCENARIOS; do
  nice -n 10 uv run --no-sync python scripts/merge_hvp_parts.py \
    --parts "$PARTS" --dest "$OUTROOT/$S" --scenario "$S" \
    >> reports/v6_ft_merge.log 2>&1
  say "merge $S rc=$?"
done
nice -n 10 uv run --no-sync python scripts/analyze_hvp_results.py \
  --root "$OUTROOT" >> reports/v6_ft_merge.log 2>&1
say "analysis rc=$?"
say "V6 FT QUEUE DONE"
