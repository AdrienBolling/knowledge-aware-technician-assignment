#!/usr/bin/env bash
# S4/S5 multi-run benchmark queue for the booked GPUs on perplexity.
#
# Goal: several independent episodes of very_long (S4) and lifecycle (S5) for
# every agent of the full benchmark table, so those two scenarios stop being
# n=1 and can carry confidence intervals.
#
# Design notes
#  * One job = one agent x one scenario x one run, --n-eps 1 with a per-run
#    --eval-seed.  Separate jobs (rather than --n-eps 5) so that a job killed
#    at the booking deadline costs one episode, not five.
#  * Runs are ordered BREADTH-FIRST: every agent gets run 1 before any agent
#    gets run 2.  If the deadline truncates the queue, every agent ends with
#    the same number of episodes, which is what the statistics need.
#  * Within a run, longest-estimated job first, so the tail stays short.
#  * A job is skipped when its episodes.csv already exists, so the queue is
#    restartable.
#  * Nothing is started that cannot finish before the deadline (EST_H below).
#  * The driver is meant to be launched under setsid; it records its own pgid
#    in ~/kata_s45/pgid for perplexity_deadline_guard.sh, and every job carries
#    KATA_S45_RUN=1 in its environment as a secondary marker.
#
# Usage:  setsid nohup scripts/perplexity_s45_queue.sh > ~/kata_s45/queue.log 2>&1 < /dev/null &
# Env:    SLOTS_PER_GPU (default 4), GPUS (default "0 1"), RUNS (default 5),
#         AGENTS (default: the full v6w roster), DRY=1 to print the plan only.
set -u
cd "$HOME/repositories/knowledge-aware-technician-assignment" || exit 1
export PYTHONPATH=$PWD/src
export KATA_S45_RUN=1

BASE="$HOME/kata_s45"
OUT="reports/hvp_s45"
STOP="$BASE/STOP"
DEADLINE_UTC="${KATA_DEADLINE_UTC:-2026-09-11 21:59:00}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-4}"
GPUS="${GPUS:-0 1}"
RUNS="${RUNS:-5}"
SEED_BASE="${SEED_BASE:-20260908}"
mkdir -p "$BASE" "$OUT"
ps -o pgid= -p $$ | tr -d ' ' > "$BASE/pgid"

AGENTS="${AGENTS:-hc_v6 hc_v6_last ft_quality ft_quality_last ft_fatigue ft_fatigue_last \
ft_protect ft_protect_last ft_gini ft_gini_last po_v6 po_v6_last \
a2c_mlp a2c_mlp_last grpo_mlp grpo_mlp_last dql_mlp dql_mlp_last \
topsis empirical_topsis shortest_processing empirical_spt reserve_specialist \
least_fatigued train_weakest shortest_queue round_robin least_busy random \
greedy_reward optimal_assignment batch_milp}"

# Estimated hours per episode, from the v6w wall clock scaled by the measured
# speed of this box.  Only used for ordering and for the will-it-finish check.
est_h() {  # $1 agent  $2 scenario
  local a=$1 s=$2
  case "$s" in
    very_long) case "$a" in
        ft_*|hc_v6*|po_v6*) echo 2.6 ;;
        greedy_reward)      echo 1.7 ;;
        *_mlp|*_mlp_last)   echo 1.5 ;;
        *)                  echo 0.7 ;;
      esac ;;
    *) case "$a" in
        ft_*|hc_v6*|po_v6*) echo 1.4 ;;
        greedy_reward)      echo 0.9 ;;
        *_mlp|*_mlp_last)   echo 0.8 ;;
        *)                  echo 0.4 ;;
      esac ;;
  esac
}

secs_left() { echo $(( $(date -u -d "$DEADLINE_UTC" +%s) - $(date -u +%s) )); }

build_plan() {   # breadth-first over runs, longest first inside a run
  local r a s e
  for r in $(seq 1 "$RUNS"); do
    for s in very_long lifecycle; do
      for a in $AGENTS; do
        e=$(est_h "$a" "$s")
        printf '%s %s %s %s\n' "$e" "$r" "$s" "$a"
      done
    done | sort -k1,1gr -k4,4
  done
}

launch() {   # $1 est  $2 run  $3 scenario  $4 agent  $5 gpu
  local est=$1 run=$2 sc=$3 ag=$4 gpu=$5
  local dir="$OUT/run$run/$ag"
  local log="$BASE/logs/run${run}_${sc}_${ag}.log"
  mkdir -p "$dir" "$BASE/logs"
  ( s=$(date +%s)
    CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/eval_human_vs_performance.py \
      --scenario "$sc" --n-eps 1 --agents "$ag" \
      --eval-seed $(( SEED_BASE + run )) --out-root "$dir" > "$log" 2>&1
    rc=$?
    echo "$(date -u +%FT%TZ) run=$run $sc $ag gpu=$gpu rc=$rc wall_h=$(awk -v a=$(date +%s) -v b=$s 'BEGIN{printf "%.2f",(a-b)/3600}')" >> "$BASE/DONE" ) &
}

running() { jobs -rp | wc -l; }

echo "=== queue start $(date -u +%FT%TZ) | slots/gpu=$SLOTS_PER_GPU gpus='$GPUS' runs=$RUNS ==="
total=$(build_plan | wc -l)
echo "planned jobs: $total | deadline $DEADLINE_UTC UTC ($(( $(secs_left) / 3600 )) h left)"
[ "${DRY:-0}" = 1 ] && { build_plan | head -40; echo "... ($total total)"; exit 0; }

max=$(( SLOTS_PER_GPU * $(echo $GPUS | wc -w) ))
i=0; skipped=0; started=0; declined=0
gpu_list=($GPUS); gi=0
while read -r est run sc ag; do
  i=$((i+1))
  [ -f "$STOP" ] && { echo "STOP flag -> no new jobs"; break; }
  if [ -f "$OUT/run$run/$ag/$sc/episodes.csv" ]; then skipped=$((skipped+1)); continue; fi
  need=$(awk -v e="$est" 'BEGIN{printf "%d", e*3600*1.35}')      # 35% headroom
  if [ "$(secs_left)" -lt "$need" ]; then
    declined=$((declined+1)); continue                            # cannot finish in time
  fi
  while [ "$(running)" -ge "$max" ]; do
    sleep 30
    [ -f "$STOP" ] && break 2
  done
  gpu=${gpu_list[$(( gi % ${#gpu_list[@]} ))]}; gi=$((gi+1))
  launch "$est" "$run" "$sc" "$ag" "$gpu"
  started=$((started+1))
  echo "$(date -u +%FT%TZ) [$i/$total] launched run=$run $sc $ag gpu=$gpu est=${est}h (running $(running))"
done < <(build_plan)

wait
echo "=== queue end $(date -u +%FT%TZ) | started=$started skipped=$skipped declined_no_time=$declined ==="
echo "S45 QUEUE DONE" >> "$BASE/DONE"
