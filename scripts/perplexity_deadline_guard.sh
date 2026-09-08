#!/usr/bin/env bash
# Hourly guard for the S4/S5 multi-run benchmark on perplexity.
#
# The GPU booking (GPUs 0 and 1) runs to 2026-09-11 23:59 Europe/Luxembourg
# = 21:59 UTC; perplexity's clock is UTC.  Past that instant this stops the
# queue and every job it spawned, and drops a STOP flag so nothing restarts.
#
# The queue driver runs under setsid, so it leads its own process group and the
# whole tree dies with one signal to -PGID.  Matching on the command line is NOT
# used: pgrep -f also matches the ssh remote shell that invokes it, and an env
# marker never appears in /proc/PID/cmdline at all.  A /proc/*/environ sweep is
# kept only as a safety net for anything reparented out of the group.
#
# Install (hourly at :05):
#   ( crontab -l 2>/dev/null | grep -v perplexity_deadline_guard
#     echo '5 * * * * $HOME/repositories/knowledge-aware-technician-assignment/scripts/perplexity_deadline_guard.sh' ) | crontab -
#
# Check by hand:  scripts/perplexity_deadline_guard.sh --status
set -u
DEADLINE_UTC="${KATA_DEADLINE_UTC:-2026-09-11 21:59:00}"
MARK=KATA_S45_RUN
BASE="$HOME/kata_s45"
LOG="$BASE/deadline_guard.log"
STOP="$BASE/STOP"
PGIDF="$BASE/pgid"
mkdir -p "$BASE"

now=$(date -u +%s)
dl=$(date -u -d "$DEADLINE_UTC" +%s 2>/dev/null) || { echo "bad deadline: $DEADLINE_UTC" >&2; exit 2; }
left=$(( (dl - now) / 60 ))
mypgid=$(ps -o pgid= -p $$ | tr -d ' ')

live_pgid() {   # echo the recorded pgid if a process group still exists under it
  [ -s "$PGIDF" ] || return 1
  local p; p=$(cat "$PGIDF")
  case "$p" in ''|*[!0-9]*) return 1;; esac
  [ "$p" = "$mypgid" ] && return 1          # never target our own group
  pgrep -g "$p" > /dev/null 2>&1 || return 1
  echo "$p"
}

strays() {      # tagged PIDs outside BOTH our own group and the queue group
  local pid pg qpgid
  qpgid=$(cat "$PGIDF" 2>/dev/null)
  for pid in $(ps -u "$USER" -o pid= ); do
    [ -r "/proc/$pid/environ" ] || continue
    grep -qz "^${MARK}=" "/proc/$pid/environ" 2>/dev/null || continue
    pg=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ "$pg" = "$mypgid" ] && continue
    [ -n "$qpgid" ] && [ "$pg" = "$qpgid" ] && continue
    echo "$pid"
  done
}

if [ "${1:-}" = "--status" ]; then
  echo "now       $(date -u +%FT%TZ)"
  echo "deadline  $DEADLINE_UTC UTC"
  echo "left      ${left} min"
  p=$(live_pgid) && echo "queue     pgid $p, $(pgrep -g "$p" | wc -l) processes" || echo "queue     not running"
  s=$(strays | wc -l); [ "$s" -gt 0 ] && echo "strays    $s tagged process(es) outside the group"
  [ -f "$STOP" ] && echo "STOP      flag present"
  exit 0
fi

if [ "$now" -lt "$dl" ]; then
  p=$(live_pgid) && n=$(pgrep -g "$p" | wc -l) || n=0
  echo "$(date -u +%FT%TZ) ok, ${left} min left, ${n} processes in the queue group" >> "$LOG"
  exit 0
fi

touch "$STOP"
p=$(live_pgid) || p=""
st=$(strays | tr '\n' ' ')
if [ -z "$p" ] && [ -z "${st// /}" ]; then
  echo "$(date -u +%FT%TZ) DEADLINE PASSED, nothing of ours is running" >> "$LOG"
  exit 0
fi
echo "$(date -u +%FT%TZ) DEADLINE PASSED, TERM pgid=${p:-none} strays=${st:-none}" >> "$LOG"
[ -n "$p" ] && kill -TERM -- "-$p" 2>/dev/null
[ -n "${st// /}" ] && kill -TERM $st 2>/dev/null
for _ in $(seq 30); do
  sleep 1
  p2=$(live_pgid) || p2=""
  st2=$(strays | tr '\n' ' ')
  [ -z "$p2" ] && [ -z "${st2// /}" ] && break
done
p2=$(live_pgid) || p2=""
st2=$(strays | tr '\n' ' ')
if [ -n "$p2" ] || [ -n "${st2// /}" ]; then
  echo "$(date -u +%FT%TZ) still alive, KILL pgid=${p2:-none} strays=${st2:-none}" >> "$LOG"
  [ -n "$p2" ] && kill -KILL -- "-$p2" 2>/dev/null
  [ -n "${st2// /}" ] && kill -KILL $st2 2>/dev/null
fi
p3=$(live_pgid) || p3=""
echo "$(date -u +%FT%TZ) guard done, queue group ${p3:-gone}, strays $(strays | wc -l)" >> "$LOG"
