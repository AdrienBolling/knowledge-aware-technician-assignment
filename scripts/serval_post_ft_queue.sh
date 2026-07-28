#!/usr/bin/env bash
# Post-fine-tune queue: waits for DONE_TRAIN_V3FT, canonicalises the
# final + mid-training round checkpoints (inline eval is disabled during
# the fine-tune, so there is no best.pt), then benchmarks both at the
# 5M horizon (the fine-tune's target), industrial, and baseline scales,
# merging into the existing reports/hvp_eval_v3 generation.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
Q=reports/post_ft_queue.log
SEED=20260722
say() { echo "$(date -u +%FT%TZ) [ft] $*" >> "$Q"; }

say "FT QUEUE ARMED (pid $$)"

while ! grep -q DONE_TRAIN_V3FT reports/train_hc_v3_ft.log 2>/dev/null; do
  sleep 300
done
RC=$(grep DONE_TRAIN_V3FT reports/train_hc_v3_ft.log | tail -1 \
     | grep -oE "rc=[0-9]+" | cut -d= -f2)
say "fine-tune finished rc=${RC:-?}"
if [ "${RC:-1}" != "0" ]; then
  say "ABORT: fine-tune rc=${RC:-?}"
  exit 1
fi

CKPTS=$(ls -1 checkpoints/hc_v3_ft/set_transformer_round*.pt 2>/dev/null | sort)
N=$(echo "$CKPTS" | grep -c .)
if [ "$N" -lt 1 ]; then
  say "ABORT: no round checkpoints in checkpoints/hc_v3_ft"
  exit 1
fi
LAST=$(echo "$CKPTS" | tail -1)
MID=$(echo "$CKPTS" | sed -n "$(( (N + 1) / 2 ))p")
mkdir -p checkpoints/hc_v3_ft_final
cp "$LAST" checkpoints/hc_v3_ft_final/set_transformer_last.pt
cp "$MID" checkpoints/hc_v3_ft_final/set_transformer_mid.pt
say "canonicalised last=$LAST mid=$MID (of $N round ckpts)"

vl() {
  local A=$1
  say "vl $A start"
  nice -n 10 uv run python scripts/eval_human_vs_performance.py \
    --scenario very_long --agents "$A" --eval-seed "$SEED" \
    --record-every 200 --out-root "reports/hvp_vl_parts/$A" \
    >> "reports/eval_vl_${A}.log" 2>&1
  say "vl $A rc=$?"
}
vl hc_v3_ft &
vl hc_v3_ft_mid &

say "std evals start (massive + baseline, --merge)"
nice -n 10 uv run python scripts/eval_human_vs_performance.py \
  --scenario massive_scale --steps 25000 --n-eps 3 \
  --agents hc_v3_ft,hc_v3_ft_mid --eval-seed "$SEED" --merge \
  --out-root reports/hvp_eval_v3 >> reports/eval_std_ft.log 2>&1
say "std massive rc=$?"
nice -n 10 uv run python scripts/eval_human_vs_performance.py \
  --scenario baseline --agents hc_v3_ft,hc_v3_ft_mid \
  --eval-seed "$SEED" --merge \
  --out-root reports/hvp_eval_v3 >> reports/eval_std_ft.log 2>&1
say "std baseline rc=$?"

wait
nice -n 10 uv run python scripts/merge_hvp_parts.py \
  --parts reports/hvp_vl_parts --dest reports/hvp_eval_v3/very_long \
  >> reports/eval_std_ft.log 2>&1
say "vl merge rc=$?"
nice -n 10 uv run python scripts/analyze_hvp_results.py \
  --root reports/hvp_eval_v3 >> reports/eval_std_ft.log 2>&1
say "analysis rc=$?"
say "FT QUEUE DONE"
