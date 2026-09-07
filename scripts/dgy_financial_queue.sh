#!/usr/bin/env bash
# Financial-analysis re-evaluation on dgy (user request 2026-09-06):
# re-run the whole v6w roster (+ the Evo-TOPSIS pair) on the five
# benchmark scenarios with the harness's new episode-end financial
# KPIs (exact final-stage production uptime, fleet uptime, repair
# labour time, downtime) so the Rodríguez-style profit map can use a
# production-based uptime factor.  Same seeds/protocol as the v6w
# generation (eval seed 20260722, massive_scale = 3 eps @ 25k steps).
#
# Separate idempotent parts tree reports/hvp_fin_parts/<agent>/<scenario>
# — NEVER merged into reports/hvp_eval_v6w (the learned agents' 30-tech
# rows may differ by the known GPU-nondeterminism band).  Pulled to the
# desktop and assembled by scripts/financial_analysis.py.
#
# Fan-out: $LANES concurrent parts (xargs -P), longest jobs first;
# learned agents round-robin over GPUs 0-2 (GPU 3 is dead), heuristics
# CPU-only.  uv --no-sync everywhere (deliberate torch downgrade).
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export OMP_NUM_THREADS=2
# Machine ids are hash(name) % 10000 under randomised string hashing:
# seed 0 is collision-free for all five benchmark layouts (probed
# 2026-09-06), so every process sees the full park in its stats/obs.
export PYTHONHASHSEED=0
export Q=reports/financial_dgy_queue.log
export EVAL_SEED=20260722
export PARTS=reports/hvp_fin_parts
LANES="${LANES:-18}"
LEARNED="hc_v6 hc_v6_last ft_quality ft_quality_last po_v6 po_v6_last ft_fatigue ft_fatigue_last ft_protect ft_protect_last ft_gini ft_gini_last a2c_mlp a2c_mlp_last grpo_mlp grpo_mlp_last dql_mlp dql_mlp_last"
HEURISTICS="empirical_topsis empirical_spt topsis shortest_processing optimal_assignment batch_milp greedy_reward reserve_specialist shortest_queue least_busy least_fatigued round_robin random train_weakest evo_topsis evo_topsis_inf"
say() { echo "$(date -u +%FT%TZ) [fin] $*" | tee -a "$Q"; }
export -f say
mkdir -p reports "$PARTS"

# Preflight: every learned key needs its checkpoint (paths mirror the
# harness CHECKPOINTS map); missing keys are skipped, not fatal.
ckpt_of() {
  case "$1" in
    hc_v6) echo checkpoints/hc_v6_final/set_transformer_best.pt ;;
    hc_v6_last) echo checkpoints/hc_v6_final/set_transformer_last.pt ;;
    po_v6) echo checkpoints/po_v6_final/set_transformer_best.pt ;;
    po_v6_last) echo checkpoints/po_v6_final/set_transformer_last.pt ;;
    ft_*_last) echo "checkpoints/${1%_last}/set_transformer_last.pt" ;;
    ft_*) echo "checkpoints/$1/set_transformer_best.pt" ;;
    *_mlp_last) echo "checkpoints/${1%_last}_v1/${1%_last}_last.pt" ;;
    *_mlp) echo "checkpoints/${1}_v1/${1}_best.pt" ;;
  esac
}
RUN_LEARNED=""
for A in $LEARNED; do
  f=$(ckpt_of "$A")
  if [ -f "$f" ]; then RUN_LEARNED="$RUN_LEARNED $A"; else say "SKIP $A: checkpoint $f missing"; fi
done

part() {  # $1 agent, $2 scenario, $3 gpu ("" = cpu only)
  local A=$1 S=$2 G=$3 EXTRA=""
  if [ -s "$PARTS/$A/$S/episodes.csv" ]; then
    say "part $A/$S cached"; return
  fi
  [ "$S" = massive_scale ] && EXTRA="--steps 25000 --n-eps 3"
  say "part $A/$S start (gpu '${G}')"
  CUDA_VISIBLE_DEVICES="$G" nice -n 10 uv run --no-sync python \
    scripts/eval_human_vs_performance.py \
    --scenario "$S" $EXTRA --agents "$A" --eval-seed "$EVAL_SEED" \
    --record-every 500 --out-root "$PARTS/$A" \
    > "reports/fin_part_${A}_${S}.log" 2>&1
  say "part $A/$S rc=$?"
}
export -f part

JOBS=$(mktemp)
i=0
for S in very_long lifecycle; do            # the ~5-8 h parts first
  for A in $RUN_LEARNED; do echo "$A $S $((i % 3))" >> "$JOBS"; i=$((i+1)); done
done
for S in very_long lifecycle; do
  for A in $HEURISTICS; do echo "$A $S" >> "$JOBS"; done
done
for S in massive_scale baseline small_scale; do
  for A in $RUN_LEARNED; do echo "$A $S $((i % 3))" >> "$JOBS"; i=$((i+1)); done
  for A in $HEURISTICS; do echo "$A $S" >> "$JOBS"; done
done
say "FIN QUEUE ARMED (pid $$, $(wc -l < "$JOBS") parts, $LANES lanes, learned:$RUN_LEARNED)"

# shellcheck disable=SC2016
xargs -P "$LANES" -L 1 bash -c 'part "$0" "$1" "${2:-}"' < "$JOBS"
rm -f "$JOBS"
say "$(grep -c 'rc=0' "$Q") parts rc=0, $(grep -c 'rc=[1-9]' "$Q") failed"
say "FIN QUEUE DONE"
