#!/usr/bin/env bash
# v6 reward fine-tune queue on dgy — run inside a zellij session.
#
# Goal: small behavioural nudges on hc_v6 via reward re-weighting, one
# lever per variant (attribution stays clean; same seed = same world
# sequence for all three).  Motivated by the v6w read: throughput
# learned (#1 at 5M) but fleet protection NOT (illness mid-pack,
# availability ~0.005 below best, quality weakest KPI 0.26-0.39).
#
#   ft_protect : fleet_availability 0.5->2.0, workload_balance 0.5->1.0
#                (price the SYMPTOM: availability dips + imbalance)
#   ft_fatigue : fatigue_cost 1.0->2.5
#                (price the CAUSE: fatigue drives exhaustion->illness)
#   ft_quality : repair_quality 1.0->2.5
#                (price the weakest KPI: skill-match quality)
#
# Reward components are running-normalized (normalize_components=true),
# so coefficients are comparable shares of the reward mix — a 2-2.5x
# bump is a strong-but-not-dominating re-weight.
#
# Fine-tune regime: init from hc_v6 final.pt (the run's strongest
# stretch), 150 episodes vec5, exact v5/v6 world + recipe otherwise.
# LR schedule: ctor lr 3e-5 (0.1x the v6 peak), warmup 10 updates,
# cosine to the 0.05 floor (1.5e-6), total_updates auto-sized by the
# cadence-aware heuristic (~168 for 150 eps @ vec5).  The on-load
# re-arm uses the CONSTRUCTOR lr, so the checkpoint's 3e-4 base_lrs
# cannot leak into the fine-tune.
#
# GPU plan: lifecycle benchmark owns GPU 0 — lane A (GPU 1) runs
# ft_protect then ft_quality, lane B (GPU 2) runs ft_fatigue.
# ~5 h/variant -> lanes finish in ~10 h / ~5 h.
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
say() { echo "$(date -u +%FT%TZ) [v6-ft] $*" | tee -a "$Q"; }
mkdir -p reports

if [ ! -f "$INIT" ]; then
  say "ABORT: init checkpoint $INIT missing"
  exit 1
fi
say "V6 FT QUEUE ARMED (pid $$, init $INIT, seed $SEED)"

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
    episodes=150 parallel_envs=5 \
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
say "V6 FT QUEUE DONE"
