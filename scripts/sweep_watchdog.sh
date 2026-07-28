#!/usr/bin/env bash
# sweep_watchdog.sh — continuously assert the two scheduling invariants of the sweep.
#
#   INV-1  concurrency: exactly SLOTS_PER_GPU*len(GPUS) jobs run whenever jobs remain
#          queued. Fewer means a slot leaked (marked busy while its process is dead)
#          and the machine is idling with work outstanding.
#   INV-2  affinity:    a job dispatched to slot G:i executes on PHYSICAL gpu G, and
#          no GPU carries more than SLOTS_PER_GPU jobs.
#
# Counting method: the orchestrator's DIRECT CHILDREN are exactly the dispatched jobs
# (plus the loop's own `sleep`, filtered out). Counting by cmdline pattern instead is
# unreliable -- bash `exec`s a `bash -c "single command"` in place, so run.py jobs have
# no surviving wrapper to match, and a naive pattern undercounts them.
#
# INV-2 is checked against nvidia-smi's per-GPU process list joined to /proc/<pid>/environ,
# i.e. where the kernels ACTUALLY landed, not what the launcher intended.
#
# Writes one line per sample; SILENT on healthy samples except a periodic heartbeat, so
# the log is a list of violations rather than a wall of OK.
#
#   setsid nohup bash scripts/sweep_watchdog.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
OUT="$REPO/logs/converged_all/_WATCHDOG.log"
EXPECT="${EXPECT_JOBS:-8}"
PERIOD="${WATCHDOG_PERIOD:-60}"
HEARTBEAT_EVERY=30                      # samples between healthy heartbeats

stamp() { date +'%F %T'; }
mkdir -p "$(dirname "$OUT")"
echo "[$(stamp)] watchdog up: expecting $EXPECT concurrent jobs, sampling every ${PERIOD}s" >> "$OUT"

gpu_uuid() { nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$1" | tr -d ' '; }
U0=$(gpu_uuid 0); U1=$(gpu_uuid 1)

i=0; viol=0
while :; do
  i=$((i+1))
  orch=$(pgrep -f 'bash scripts/run_converged_all.sh' | head -1)
  if [[ -z "$orch" ]]; then
    # No orchestrator: either the sweep finished or it died. Either way the invariants
    # no longer apply -- stragglers are expected to drain below EXPECT.
    left=$(pgrep -cf 'pipeline/federated_eval\.py|scripts/quality_pass\.py|run\.py -d ucr_ad' 2>/dev/null || echo 0)
    echo "[$(stamp)] orchestrator gone; $left worker process(es) still draining; watchdog exiting" >> "$OUT"
    echo "[$(stamp)] TOTAL VIOLATIONS: $viol" >> "$OUT"
    exit 0
  fi

  # INV-1 -- children minus the scheduler's own `sleep`
  running=0
  for c in $(pgrep -P "$orch" 2>/dev/null); do
    cmd=$(ps -o cmd= -p "$c" 2>/dev/null || true)
    [[ "$cmd" == sleep* ]] && continue
    [[ -n "$cmd" ]] && running=$((running+1))
  done

  # INV-2 -- where the kernels actually are
  n0=0; n1=0; mism=""
  while IFS=, read -r uuid pid _; do
    uuid=$(echo "$uuid" | tr -d ' '); pid=$(echo "$pid" | tr -d ' ')
    [[ -z "$pid" ]] && continue
    case "$uuid" in "$U0") phys=0; n0=$((n0+1));; "$U1") phys=1; n1=$((n1+1));; *) continue;; esac
    cvd=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -m1 '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2)
    # Empty cvd = a grandchild that inherited the pin (run.py's step-scripts); the
    # inherited value is what matters and it is already reflected by WHERE it landed.
    if [[ -n "$cvd" && "$cvd" != "$phys" ]]; then
      mism="$mism pid=$pid(env=$cvd,phys=$phys)"
    fi
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null)

  bad=""
  [[ "$running" -ne "$EXPECT" ]] && bad="$bad INV-1:running=$running(expected $EXPECT)"
  [[ "$n0" -gt $((EXPECT/2)) ]] && bad="$bad INV-2:GPU0_overloaded=$n0"
  [[ "$n1" -gt $((EXPECT/2)) ]] && bad="$bad INV-2:GPU1_overloaded=$n1"
  [[ -n "$mism" ]] && bad="$bad INV-2:AFFINITY_MISMATCH$mism"

  if [[ -n "$bad" ]]; then
    viol=$((viol+1))
    echo "[$(stamp)] VIOLATION$bad (gpu0=$n0 gpu1=$n1)" >> "$OUT"
  elif (( i % HEARTBEAT_EVERY == 1 )); then
    echo "[$(stamp)] ok: $running jobs, gpu0=$n0 gpu1=$n1, violations so far=$viol" >> "$OUT"
  fi
  sleep "$PERIOD"
done
