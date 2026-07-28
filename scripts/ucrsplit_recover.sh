#!/usr/bin/env bash
# ucrsplit_recover.sh — clean restart of the ucr_split sweep after a freeze.
#
# Safe to run at any time: run_ucrsplit.sh SKIPS every job whose out-json already exists, so
# a relaunch resumes from what is on disk. Nothing completed is lost or recomputed.
#
# ORDER MATTERS. run_ucrsplit_cb.sh and ucrsplit_on_finish.sh both poll `pgrep -f
# run_ucrsplit.sh` every 120 s. If the main orchestrator is killed while they are alive,
# within 2 minutes the cb sweep starts dispatching into the same 14 slots (oversubscribing
# the moment the main sweep comes back) and on_finish writes a premature FINAL_CHECK against
# a partial table. So the two waiters are stopped FIRST and restarted LAST.
#
# Killing a wedged job does not "lose" progress: a frozen client has produced nothing since
# it froze. Stopping/restarting MPS does NOT unblock an already-attached client either --
# it is blocked in a socket poll with no timeout and has to be killed.
#
#   bash scripts/ucrsplit_recover.sh              # relaunch, MPS left DOWN (default)
#   bash scripts/ucrsplit_recover.sh --mps        # also bring MPS back up (private pipe dir)
#   bash scripts/ucrsplit_recover.sh --dry-run    # show what would happen
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PY:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10}"
LOGDIR="$REPO/logs/ucrsplit"; mkdir -p "$LOGDIR"
REC="$LOGDIR/_recover.log"

WITH_MPS=""; DRY=""
for a in "$@"; do case "$a" in
  --mps) WITH_MPS=1 ;;
  --no-mps) WITH_MPS="" ;;
  --dry-run) DRY=1 ;;
  *) echo "unknown arg: $a"; exit 2 ;;
esac; done

stamp() { date +'%F %T'; }
say() { echo "[$(stamp)] recover: $*" | tee -a "$REC"; }
run() { if [[ -n "$DRY" ]]; then echo "    would run: $*"; else eval "$@"; fi; }

say "=== recovery start (mps=${WITH_MPS:-down}${DRY:+, DRY-RUN}) ==="
say "json on disk before: $(ls -1 artifacts/ucrsplit/*__*.json 2>/dev/null | wc -l) / 492"

# 1. the two waiters, before the orchestrator they are waiting on
for pat in 'scripts/run_ucrsplit_cb\.sh' 'scripts/ucrsplit_on_finish\.sh'; do
  pids="$(pgrep -f "$pat" 2>/dev/null | paste -sd' ')"
  [[ -n "$pids" ]] && { say "stopping waiter $pat (pid $pids)"; run "kill -TERM $pids 2>/dev/null"; }
done

# 2. the orchestrator
pids="$(pgrep -f 'bash scripts/run_ucrsplit\.sh' 2>/dev/null | paste -sd' ')"
[[ -n "$pids" ]] && { say "stopping orchestrator (pid $pids)"; run "kill -TERM $pids 2>/dev/null"; }

# 3. any watchdog from the previous incarnation
pids="$(pgrep -f 'scripts/gpu_stall_watchdog\.sh' 2>/dev/null | paste -sd' ')"
[[ -n "$pids" ]] && { say "stopping old watchdog (pid $pids)"; run "kill -TERM $pids 2>/dev/null"; }

# 4. the jobs themselves
pids="$(pgrep -f 'pipeline/federated_eval\.py --dataset ucr_split' 2>/dev/null | paste -sd' ')"
if [[ -n "$pids" ]]; then
  say "stopping $(echo $pids | wc -w) job process(es)"
  run "kill -TERM $pids 2>/dev/null"
  [[ -z "$DRY" ]] && sleep 15
  pids="$(pgrep -f 'pipeline/federated_eval\.py --dataset ucr_split' 2>/dev/null | paste -sd' ')"
  [[ -n "$pids" ]] && { say "  KILL for $(echo $pids | wc -w) survivor(s)"; run "kill -KILL $pids 2>/dev/null"; }
fi

# 5. MPS
if [[ -n "$WITH_MPS" ]]; then
  say "restarting MPS on the private pipe directory"
  run "bash scripts/mps_ctl.sh restart >> '$REC' 2>&1"
else
  say "stopping MPS (leaving it down; jobs will run in plain GPU time-slicing)"
  run "bash scripts/mps_ctl.sh down >> '$REC' 2>&1"
  say "  NOTE: without MPS the measured throughput drops from ~10.8 to ~3 stage-1 rounds/min."
fi

[[ -n "$DRY" ]] && { say "=== dry run: nothing was changed ==="; exit 0; }

# 6. relaunch, orchestrator first
say "relaunching run_ucrsplit.sh (skips the $(ls -1 artifacts/ucrsplit/*__*.json 2>/dev/null | wc -l) json already on disk)"
setsid nohup bash scripts/run_ucrsplit.sh > "$LOGDIR/../ucrsplit_boot.log" 2>&1 < /dev/null &
sleep 5

# 7. liveness watchdog for the new incarnation
say "starting the stall watchdog (window 20 min, action=warn)"
WATCH_PATTERN='pipeline/federated_eval\.py --dataset ucr_split' \
WATCH_ORCH='bash scripts/run_ucrsplit\.sh' \
WATCH_LOG="$LOGDIR/_STALL_WATCHDOG.log" \
WATCH_WINDOW=1200 WATCH_ACTION=warn \
  setsid nohup bash scripts/gpu_stall_watchdog.sh > /dev/null 2>&1 < /dev/null &

# 8. the waiters, last
say "restarting the cb waiter and the on_finish reporter"
setsid nohup bash scripts/run_ucrsplit_cb.sh > "$REPO/logs/ucrsplit_cb_boot.log" 2>&1 < /dev/null &
setsid nohup bash scripts/ucrsplit_on_finish.sh > "$LOGDIR/_on_finish.log" 2>&1 < /dev/null &

sleep 5
say "=== recovery done ==="
say "  orchestrator : $(pgrep -cf 'bash scripts/run_ucrsplit\.sh') proc"
say "  watchdog     : $(pgrep -cf 'scripts/gpu_stall_watchdog\.sh') proc"
say "  cb waiter    : $(pgrep -cf 'scripts/run_ucrsplit_cb\.sh') proc"
say "  on_finish    : $(pgrep -cf 'scripts/ucrsplit_on_finish\.sh') proc"
say "  watch: tail -f $LOGDIR/_STALL_WATCHDOG.log  and  tail -f $LOGDIR/_orchestrator.log"
