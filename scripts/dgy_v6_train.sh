#!/usr/bin/env bash
# v6 training on dgy-serval — run INSIDE tmux (long run; survives ssh drops).
#
# v6 = v5 objective/world UNCHANGED (env=train_multiscale_v5) + the
#
# NOTE: uv run uses --no-sync everywhere — the dgy venv carries a
# deliberate torch downgrade (2.7.1+cu126: the lockfile's cu130 build
# has no sm_70 kernels for these V100s); a bare `uv run` would re-sync
# the lockfile and silently reinstall the incompatible torch.
# 2026-08-04 fix package (D1 LR schedule, D2 dropout, D3 role-bound
# fusion + feature-context, D6 shared GAE, D11 boolean tokens, D12
# machine ids).  SINGLE environment (user decision: no SimPy env
# parallelisation), so the classic serial training loop is used —
# no AsyncVectorEnv machinery anywhere.
#
# eval_interval raised to 200 (a 5-episode inline eval costs ~20 min
# at single-env speed; every 50 rounds would burn ~1 day of the run).
# Best/last checkpoints canonicalised at the end (D5: keep as-is).
set -u
export PATH="$HOME/.local/bin:$PATH"  # uv lives here on dgy (tmux shells are non-login)
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4
Q=reports/v6_dgy_queue.log
say() { echo "$(date -u +%FT%TZ) [v6dgy] $*" | tee -a "$Q"; }
mkdir -p reports checkpoints

say "V6 DGY TRAIN ARMED (pid $$, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

# ---------- 1. BC warm-start with the v6 architecture ----------
if [ ! -f checkpoints/bc_topsis_v6/set_transformer_bc.pt ]; then
  say "BC collection start (v6 architecture)"
  uv run --no-sync python scripts/warmstart_bc.py \
    --env-config run_configs/benchmark_suite/train_multiscale_v5.json \
    --agent-config run_configs/agents/set_transformer_v6.json \
    --episodes 25 --sim-time 200000 --seed 7 \
    --out checkpoints/bc_topsis_v6/set_transformer_bc.pt \
    >> reports/v6_bc_collect.log 2>&1
  say "BC collection rc=$?"
fi
if [ ! -f checkpoints/bc_topsis_v6/set_transformer_bc.pt ]; then
  say "ABORT: BC checkpoint missing (see reports/v6_bc_collect.log)"
  exit 1
fi

# ---------- 2. Train (single env, 600 eps) ----------
say "v6 training start (600 eps, parallel_envs=1)"
uv run --no-sync python scripts/train_hydra.py \
  env=train_multiscale_v5 agent=set_transformer_v6 \
  episodes=600 parallel_envs=1 \
  sim_time=275000 sim_time_min=200000 sim_time_max=350000 \
  eval_interval=200 checkpoint_interval=50 seed=42 \
  init_checkpoint=checkpoints/bc_topsis_v6/set_transformer_bc.pt \
  checkpoint_dir=checkpoints/hc_v6 \
  >> reports/train_hc_v6.log 2>&1
RC=$?
say "training rc=$RC"
echo "DONE_TRAIN_V6 rc=$RC $(date -u +%FT%TZ)" >> reports/train_hc_v6.log
if [ "$RC" != "0" ]; then say "ABORT: training failed"; exit 1; fi

# ---------- 3. Canonicalise best + last ----------
# End-of-run checkpoint naming depends on the loop: the serial loop
# (parallel_envs=1, this run) writes ep*.pt + final.pt, the vec loop
# round*.pt + final.pt.
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
mkdir -p checkpoints/hc_v6_final
cp checkpoints/hc_v6/set_transformer_best.pt \
   checkpoints/hc_v6_final/set_transformer_best.pt
LAST=$(pick_last checkpoints/hc_v6)
cp "$LAST" checkpoints/hc_v6_final/set_transformer_last.pt
say "final ckpts: best + last=$(basename "$LAST")"
say "V6 DGY TRAIN DONE (benchmarks + lifecycle eval run separately, later)"
