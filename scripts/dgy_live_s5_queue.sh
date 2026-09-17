#!/usr/bin/env bash
# Deployment-time adaptation study on S5, dgy GPUs 0-1 — run inside zellij.
#
# Question: can HTT-RL improve during deployment?  Base policy: HTT-RL qua.
# (checkpoints/ft_quality/set_transformer_best.pt).  Factory: the S5
# benchmark layout (eval seed 20260722), episode seed 2026072200.  Updates
# use the training reward of the quality fine-tune (train_multiscale_v5
# stack, repair_quality 2.5) and the fine-tune PPO recipe (lr 3e-5).
#
# Lane GPU 1 — scripts/live_finetune_s5.py, one S5 episode each, in parallel:
#   ft_quality_live_sto  PPO update every 2048 decisions, sampled actions
#   ft_quality_sto       control: sampled actions, no updates
#   ft_quality_live_det  PPO update every 2048 decisions, argmax actions
#   then deterministic benchmarks of the two live final checkpoints.
# Lane GPU 0:
#   ft_quality           benchmark rerun with this code (sanity: published
#                        S5 row = 30601 products)
#   ft_quality_s5ft      10-episode fine-tune on the fixed S5 factory
#                        (run_configs/benchmark_suite/lifecycle_fixed_s5.json,
#                        train_hydra, 5 envs), then its benchmark.
#
# Booking: GPUs 0-1 until 2026-09-17 23:59 local (21:59 UTC).  No stage
# starts after STOP_UTC.  Stages are cached: a rerun skips finished parts.
#
# Layout expected on dgy: this tree at ~/kata_live, with checkpoints/ft_quality
# linked to the main checkout; the main checkout's venv (torch 2.7.1+cu126,
# sm_70 kernels) runs everything.
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY="${PY:-$HOME/repositories/knowledge-aware-technician-assignment/.venv/bin/python}"
export PYTHONPATH="$ROOT/src:$ROOT/scripts"
export PYTHONHASHSEED=0
export WANDB_MODE=offline
export OMP_NUM_THREADS=2
Q=reports/live_s5_queue.log
OUT=reports/live_s5
CKPT=checkpoints/ft_quality/set_transformer_best.pt
STOP_UTC="${STOP_UTC:-2026-09-17 21:59:00}"
say() { echo "$(date -u +%FT%TZ) [live-s5] $*" | tee -a "$Q"; }
in_booking() { [ "$(date -u +%s)" -lt "$(date -u -d "$STOP_UTC" +%s)" ]; }
mkdir -p reports "$OUT"

say "LIVE S5 QUEUE ARMED (pid $$, code $(cat COMMIT 2>/dev/null || echo unknown))"

live() {  # $1 label  $2 updates on|off  $3 act
  local L=$1 U=$2 A=$3
  if [ -s "$OUT/$L/lifecycle/episodes.csv" ]; then say "$L cached"; return; fi
  in_booking || { say "$L skipped: past booking"; return; }
  say "$L start (GPU 1, updates=$U act=$A)"
  CUDA_VISIBLE_DEVICES=1 nice -n 5 "$PY" scripts/live_finetune_s5.py \
    --label "$L" --out-root "$OUT/$L" --checkpoint "$CKPT" \
    --updates "$U" --act "$A" > "reports/live_s5_$L.log" 2>&1
  say "$L rc=$?"
}

bench() {  # $1 harness key  $2 gpu
  local K=$1 G=$2
  if [ -s "$OUT/bench/$K/lifecycle/episodes.csv" ]; then say "bench $K cached"; return; fi
  in_booking || { say "bench $K skipped: past booking"; return; }
  say "bench $K start (GPU $G)"
  CUDA_VISIBLE_DEVICES=$G nice -n 5 "$PY" scripts/eval_human_vs_performance.py \
    --scenario lifecycle --agents "$K" --eval-seed 20260722 --record-every 200 \
    --out-root "$OUT/bench/$K" > "reports/live_s5_bench_$K.log" 2>&1
  say "bench $K rc=$?"
}

finetune() {
  local DIR=checkpoints/ft_quality_s5ft
  if [ -s "$DIR/set_transformer_final.pt" ]; then say "s5ft cached"; return; fi
  in_booking || { say "s5ft skipped: past booking"; return; }
  say "s5ft train start (GPU 0, 10 episodes x 5 envs)"
  CUDA_VISIBLE_DEVICES=0 nice -n 5 "$PY" scripts/train_hydra.py \
    env=lifecycle_fixed_s5 agent=set_transformer_v6 episodes=10 parallel_envs=5 \
    sim_time=5000000 sim_time_min=null sim_time_max=null max_steps=1500000 \
    tu_per_decision=8.9 eval_interval=1000 eval_episodes=1 checkpoint_interval=5 \
    seed=20260722 init_checkpoint="$CKPT" checkpoint_dir="$DIR" \
    agent.params.lr=3e-5 wandb=false > reports/train_ft_quality_s5ft.log 2>&1
  say "s5ft train rc=$?"
}

(
  live ft_quality_live_sto on stochastic &
  live ft_quality_sto off stochastic &
  live ft_quality_live_det on deterministic &
  wait
  say "LIVE RUNS DONE"
  bench ft_quality_live_sto 1 &
  bench ft_quality_live_det 1 &
  wait
  say "LIVE LANE DONE"
) &
(
  bench ft_quality 0 &
  finetune
  bench ft_quality_s5ft 0
  wait
  say "FINETUNE LANE DONE"
) &
wait
say "LIVE S5 QUEUE DONE"
