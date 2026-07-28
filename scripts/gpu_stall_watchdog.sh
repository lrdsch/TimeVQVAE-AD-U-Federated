#!/usr/bin/env bash
# gpu_stall_watchdog.sh — detect jobs that are ALIVE BUT FROZEN, and say so loudly.
#
# THE FAILURE THIS EXISTS FOR (2026-07-28 00:02:24): the MPS server died with 13 clients
# attached. The clients did not crash -- CUDA blocks on a socket poll with no timeout, so
# all 14 jobs sat in do_poll forever, burning 1 second of CPU in 9 hours, with ZERO errors
# in any log. Nothing noticed for 9.9 h ~= 139 slot-hours. The crash cost minutes; the
# missing liveness check cost the rest.
#
# WHY scripts/sweep_watchdog.sh DID NOT CATCH IT: it asserts concurrency (INV-1) and GPU
# affinity (INV-2). A frozen-but-alive process satisfies both -- it is still a child, still
# counted, still pinned. Liveness is an orthogonal invariant and this file is its home.
#
# THE SIGNAL IS CPU TIME, NOT LOG FRESHNESS. Log mtime looks tempting but false-positives on
# quiet phases (a healthy stage-1 round writes one line every ~2 min, detect writes in
# bursts). Consumed CPU jiffies cannot be faked: every phase this pipeline has -- training,
# dataloading, detect, metric computation -- burns CPU. A whole process tree that consumes
# < WATCH_JIFFY_EPS jiffies across the window is not "quiet", it is stopped. GPU utilisation
# is used only to corroborate the GLOBAL verdict, never on its own (a CPU-only phase
# legitimately shows 0% GPU).
#
# TWO LEVELS:
#   per-job    one job's tree is frozen while others advance -> an isolated hang.
#              WATCH_ACTION=kill|recover kills that tree so the orchestrator reaps the slot
#              and moves on. NOTE this leaves a HOLE: the orchestrator marks the job FAILED
#              and never retries it. Holes are filled by re-running the launcher, which
#              skips any job whose out-json already exists.
#   global     every job frozen AND both GPUs idle -> the MPS/driver-level event above.
#              WATCH_ACTION=recover runs WATCH_RECOVER_CMD, which should kill the
#              orchestrator and relaunch it (resumable, so no holes).
#
#   WATCH_PATTERN='pipeline/federated_eval\.py' \
#   WATCH_ORCH='bash scripts/run_ucrsplit\.sh' \
#   WATCH_ACTION=recover WATCH_RECOVER_CMD='bash scripts/ucrsplit_recover.sh --no-mps' \
#     setsid nohup bash scripts/gpu_stall_watchdog.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

PATTERN="${WATCH_PATTERN:-pipeline/federated_eval\.py}"
# Orchestrator pattern: watch exits once that process is gone. Taken as $1 rather than only
# from the environment so it shows up in THIS process's command line -- that is what lets a
# runner ask "is a watchdog already guarding *me*?" with pgrep, instead of finding some other
# sweep's watchdog and skipping its own.
ORCH_PAT="${1:-${WATCH_ORCH:-}}"
PERIOD="${WATCH_PERIOD:-60}"               # seconds between samples
WINDOW="${WATCH_WINDOW:-1200}"             # seconds of no progress before calling it a stall
JIFFY_EPS="${WATCH_JIFFY_EPS:-100}"        # 100 jiffies = 1 CPU-second; below this = stopped
ACTION="${WATCH_ACTION:-warn}"             # warn | kill | recover
RECOVER_CMD="${WATCH_RECOVER_CMD:-}"
OUT="${WATCH_LOG:-$REPO/logs/_STALL_WATCHDOG.log}"
HEARTBEAT_EVERY="${WATCH_HEARTBEAT:-30}"   # samples between healthy heartbeats

STRIKES=$(( (WINDOW + PERIOD - 1) / PERIOD ))
(( STRIKES < 2 )) && STRIKES=2             # never fire on a single sample

mkdir -p "$(dirname "$OUT")"
stamp() { date +'%F %T'; }
say() { echo "[$(stamp)] $*" >> "$OUT"; }

say "watchdog up: pattern='$PATTERN' period=${PERIOD}s window=${WINDOW}s (${STRIKES} strikes) action=$ACTION"
[[ "$ACTION" == "recover" && -z "$RECOVER_CMD" ]] && { say "FATAL: action=recover but WATCH_RECOVER_CMD is empty"; exit 2; }

declare -A PREV_J STRIKE      # per-job-root: last jiffy total, consecutive frozen samples
gstrike=0; i=0; fired=0

# CPU jiffies (utime+stime) of one pid. /proc/<pid>/stat field 2 is comm, which may contain
# spaces and parens, so cut everything through the LAST ')' before counting fields; utime
# and stime are then fields 12 and 13.
jiffies() { awk '{ sub(/^.*\) /, ""); print $12 + $13 }' "/proc/$1/stat" 2>/dev/null || echo 0; }

kill_tree() {  # $1 = root pid of a job
  local root="$1" victims
  victims="$root $(pgrep -f "$PATTERN" 2>/dev/null | while read -r p; do
                     a="$p"; for _ in 1 2 3 4 5 6; do
                       a=$(ps -o ppid= -p "$a" 2>/dev/null | tr -d ' '); [[ -z "$a" || "$a" == 1 ]] && break
                       [[ "$a" == "$root" ]] && { echo "$p"; break; }
                     done; done)"
  say "  killing job tree rooted at $root: $(echo $victims | tr '\n' ' ')"
  kill -TERM $victims 2>/dev/null; sleep 10; kill -KILL $victims 2>/dev/null
}

while :; do
  i=$((i+1))

  if [[ -n "$ORCH_PAT" ]] && ! pgrep -f "$ORCH_PAT" >/dev/null 2>&1; then
    say "orchestrator ('$ORCH_PAT') is gone; watchdog exiting after $fired alarm(s)"
    exit 0
  fi

  # ── inventory: matching pids, their ppids, and the job ROOTS ────────────────────────
  mapfile -t ROWS < <(ps -eo pid,ppid,cmd --no-headers 2>/dev/null | grep -E "$PATTERN" | grep -v grep)
  if [[ ${#ROWS[@]} -eq 0 ]]; then
    PREV_J=(); STRIKE=(); gstrike=0
    (( i % HEARTBEAT_EVERY == 1 )) && say "idle: no job matches '$PATTERN'"
    sleep "$PERIOD"; continue
  fi

  declare -A PPID_OF=() ISJOB=() LABEL=()
  for r in "${ROWS[@]}"; do
    set -- $r; pid=$1; ppid=$2
    PPID_OF[$pid]=$ppid; ISJOB[$pid]=1
    LABEL[$pid]="$(sed -nE 's/.*--cluster ([^ ]+).*--arms ([^ ]+).*/\1_\2/p' <<<"$r")"
  done

  # root = a matching pid whose parent is not itself a matching pid
  declare -A JIF=() NAME=()
  for pid in "${!ISJOB[@]}"; do
    a="$pid"
    for _ in 1 2 3 4 5 6 7 8; do
      p="${PPID_OF[$a]:-}"
      [[ -z "$p" || -z "${ISJOB[$p]:-}" ]] && break
      a="$p"
    done
    JIF[$a]=$(( ${JIF[$a]:-0} + $(jiffies "$pid") ))
    [[ -n "${LABEL[$a]:-}" ]] && NAME[$a]="${LABEL[$a]}"
  done

  gpu_busy=0
  while read -r u; do [[ -n "$u" && "$u" -gt 0 ]] 2>/dev/null && gpu_busy=1; done \
    < <(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)

  # ── verdict ────────────────────────────────────────────────────────────────────────
  njobs=0; nfrozen=0; frozen_roots=()
  for root in "${!JIF[@]}"; do
    njobs=$((njobs+1))
    prev="${PREV_J[$root]:-}"
    if [[ -n "$prev" ]] && (( JIF[$root] - prev < JIFFY_EPS )); then
      STRIKE[$root]=$(( ${STRIKE[$root]:-0} + 1 ))
    else
      STRIKE[$root]=0
    fi
    PREV_J[$root]="${JIF[$root]}"
    (( ${STRIKE[$root]} >= STRIKES )) && { nfrozen=$((nfrozen+1)); frozen_roots+=("$root"); }
  done
  # forget roots that no longer exist
  for root in "${!PREV_J[@]}"; do [[ -z "${JIF[$root]:-}" ]] && { unset 'PREV_J[$root]'; unset 'STRIKE[$root]'; }; done

  if (( nfrozen == njobs && nfrozen > 0 && gpu_busy == 0 )); then
    gstrike=$((gstrike+1))
    fired=$((fired+1))
    # Full inventory once, then a one-liner. An unattended watchdog on a 20-minute window
    # otherwise writes 14 lines a minute for as long as the freeze lasts, and the log stops
    # being readable exactly when someone finally comes to read it.
    if (( gstrike == 1 || gstrike % 60 == 0 )); then
      say "STALL (GLOBAL): all $njobs job(s) frozen for >=${WINDOW}s and both GPUs idle."
      say "  this is the MPS/driver-level signature: alive processes, no CPU, no GPU, no errors."
      for root in "${frozen_roots[@]}"; do say "    frozen: pid=$root ${NAME[$root]:-?}"; done
    else
      say "STALL (GLOBAL) continuing: $njobs job(s) frozen, $(( gstrike * PERIOD / 60 )) min since detection"
    fi
    if [[ "$ACTION" == "recover" ]]; then
      say "  ACTION=recover -> $RECOVER_CMD"
      bash -c "$RECOVER_CMD" >> "$OUT" 2>&1
      say "  recover command returned rc=$?; watchdog exiting (the relaunch starts its own)"
      exit 0
    fi
    (( gstrike == 1 || gstrike % 60 == 0 )) && \
      say "  ACTION=$ACTION -> not intervening. Recover with: bash scripts/ucrsplit_recover.sh"
  elif (( nfrozen > 0 )); then
    fired=$((fired+1))
    # Same throttle as the global branch: with ACTION=warn nothing removes the frozen job, so
    # only report a root the sample it crosses the threshold (and hourly thereafter).
    fresh=0
    for root in "${frozen_roots[@]}"; do
      (( ${STRIKE[$root]} == STRIKES || ${STRIKE[$root]} % 60 == 0 )) && fresh=$((fresh+1))
    done
    (( fresh > 0 )) && say "STALL (PER-JOB): $nfrozen of $njobs job(s) frozen for >=${WINDOW}s (gpu_busy=$gpu_busy)"
    for root in "${frozen_roots[@]}"; do
      (( ${STRIKE[$root]} == STRIKES || ${STRIKE[$root]} % 60 == 0 )) && say "    frozen: pid=$root ${NAME[$root]:-?}"
      if [[ "$ACTION" == "kill" || "$ACTION" == "recover" ]]; then
        kill_tree "$root"
        say "    NOTE: this leaves a hole -- re-run the launcher at the end to fill it."
        unset 'PREV_J[$root]'; unset 'STRIKE[$root]'
      fi
    done
  else
    gstrike=0
    (( i % HEARTBEAT_EVERY == 1 )) && say "ok: $njobs job(s) advancing, gpu_busy=$gpu_busy, alarms so far=$fired"
  fi

  sleep "$PERIOD"
done
