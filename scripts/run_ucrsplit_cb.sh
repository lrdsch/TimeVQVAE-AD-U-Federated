#!/usr/bin/env bash
# run_ucrsplit_cb.sh -- adds the two CODEBOOK-federation arms to the ucr_split table.
#
#   7 cbonly    federated_cb_only       suff-stat codebook merge, encoder/decoder/prior local
#   8 cbema     federated_cb_only_ema   + server-side codebook EMA over rounds (gamma=0.8 default)
#
# WHY THIS IS A SEPARATE SCRIPT AND NOT AN EDIT TO run_ucrsplit.sh: that orchestrator is
# RUNNING (booted 2026-07-27, ~184/492 jobs done at the time of writing). Editing its ARM
# table mid-flight would not retro-add the arms to the clusters it has already passed, and
# restarting it would abandon in-flight jobs. This script instead WAITS for it to exit, then
# sweeps the same cluster list with the two new arms, writing into the SAME results dir so
# `ucrsplit_aggregate.py` / `fed_aggregate.py` pick all eight arms up together.
#
# COMPARABILITY IS THE WHOLE POINT: COMMON below is copied verbatim from run_ucrsplit.sh
# (which itself copied run_converge60.sh), including --batch 64 and --fed-enc-prior local.
# Do not "improve" any of it -- the eight arms have to sit in one table.
# cbema takes NO extra flag: gamma=0.8 is the built-in default (federated_eval.py:925-929),
# which is exactly how run_converge60.sh:80 invoked it.
#
#   DRYRUN=1 bash scripts/run_ucrsplit_cb.sh                                  # print the plan
#   NOWAIT=1 bash scripts/run_ucrsplit_cb.sh                                  # skip the wait
#   setsid nohup bash scripts/run_ucrsplit_cb.sh > logs/ucrsplit_cb_boot.log 2>&1 < /dev/null &
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PY:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10}"
export PIPELINE_PYTHON="$PY"

# Private MPS pipe directory -- same reasoning as run_ucrsplit.sh, see scripts/mps_ctl.sh.
MPS_CTL_ENV_ONLY=1 source "$REPO/scripts/mps_ctl.sh"

LOGDIR="$REPO/logs/ucrsplit"; RESDIR="$REPO/artifacts/ucrsplit"
CKPT="$RESDIR/ckpt"
mkdir -p "$LOGDIR" "$RESDIR"
ORCH="$LOGDIR/_orchestrator_cb.log"

GPUS=(0 1); SLOTS_PER_GPU="${SLOTS_PER_GPU:-7}"      # the MEASURED optimum; the 16-core CPU is the wall
SEED="${SEED:-0}"
S1_ROUNDS="${S1_ROUNDS:-300}"; LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"; PATIENCE="${PATIENCE:-6}"
BATCH="${BATCH:-64}"
CLUSTER_FILE="${CLUSTER_FILE:-$REPO/scripts/ucr_split_clusters.txt}"
WAIT_PAT="${WAIT_PAT:-run_ucrsplit.sh}"

stamp() { date +'%F %T'; }
say() { echo "[$(stamp)] $*" | tee -a "$ORCH"; }

[[ -f "$CLUSTER_FILE" ]] || { echo "missing $CLUSTER_FILE"; exit 1; }
mapfile -t CLUSTERS < "$CLUSTER_FILE"

# ── wait for the main sweep to finish so we do not oversubscribe the 14 slots ──────────
# Match the RUNNER, not this script. `pgrep -f` treats the pattern as a REGEX, so the `.`
# in "run_ucrsplit.sh" is a wildcard; we additionally filter out any line mentioning
# `_cb.sh` so this script (and a second copy of it) can never wait on itself. The loop
# continues while at least one NON-_cb process still matches.
main_running() { pgrep -af "$WAIT_PAT" 2>/dev/null | grep -v '_cb\.sh' | grep -q . ; }
if [[ -z "${NOWAIT:-}" && -z "${DRYRUN:-}" ]]; then
  if main_running; then
    say "waiting for '$WAIT_PAT' to exit before dispatching (NOWAIT=1 to skip)"
    while main_running; do sleep 120; done
    say "main sweep finished -- dispatching the codebook arms"
  else
    say "'$WAIT_PAT' not running -- dispatching immediately"
  fi
fi

COMMON="--protocol converged --fed-enc-prior local --s1-rounds $S1_ROUNDS \
--local-epochs $LOCAL_EPOCHS --fed-patience-rounds $PATIENCE --seeds $SEED --batch $BATCH"

declare -A ARM=(
  [cbonly]="federated_cb_only|"
  [cbema]="federated_cb_only_ema|"
)
ARM_ORDER=(cbonly cbema)

# Seed suffix. EMPTY at seed 0 so existing artifacts keep their names and the
# skip-if-exists guard still recognises them; non-empty otherwise. Without this the
# filename carries no seed, so `SEED=1 bash <this script>` matches the seed-0 jsons and
# SKIPS every job — a silent no-op that looks like a completed replication.
SFX=""; [[ "$SEED" != "0" ]] && SFX="_s$SEED"

mk_cmd() {  # $1 cluster  $2 arm-tag -> stdout: the command
  local cl="$1" tag="$2"; local spec="${ARM[$tag]}"
  local id="${spec%%|*}" extra="${spec#*|}"
  echo "$PY -u pipeline/federated_eval.py --dataset ucr_split --cluster $cl \
--arms $id $COMMON --out-dir $CKPT $extra --out-json $RESDIR/${cl}__${tag}${SFX}.json"
}

# ── LPT ordering, recomputed exactly as run_ucrsplit.sh does (no stale-file dependency) ──
mapfile -t COST_ORDER < <("$PY" - "$CLUSTER_FILE" <<'PYEOF'
import json, math, sys
W, STRIDE, BATCH, R, LE, SPS, DPS, S2F = 128, 13, 64, 38, 10, 15.0, 60.0, 6000
keep = [l.strip() for l in open(sys.argv[1]) if l.strip()]
m = json.load(open("data/raw/ucr_split/metadata.json"))
agg = {}
for e in m["entities"]:
    a = agg.setdefault(e["series"], [0, 0, 0])
    w = max(0, e["train_length"] - W + 1)
    a[0] += math.ceil(w / BATCH)
    a[1] += max(0, (e["test_length"] - W) // STRIDE + 1)
    a[2] += min(S2F, max(200, 15 * math.ceil(w / 128)))
cost = {s: (R * LE * a[0] + a[2]) / SPS / 3600 + a[1] / DPS / 3600 for s, a in agg.items()}
for s in sorted(keep, key=lambda s: -cost[s]):
    print(s)
PYEOF
)
[[ ${#COST_ORDER[@]} -eq ${#CLUSTERS[@]} ]] || { echo "cost ordering lost clusters"; exit 1; }

JOBS=()
add() { JOBS+=("$1|$RESDIR/$2__$3${SFX}.json|$(mk_cmd "$2" "$3")"); }   # SFX: see above
for cl in "${COST_ORDER[@]}"; do for a in "${ARM_ORDER[@]}"; do add "${cl}_${a}" "$cl" "$a"; done; done

say "=== ucrsplit-cb: ${#JOBS[@]} jobs (${#CLUSTERS[@]} clusters x ${#ARM_ORDER[@]} arms), seed=$SEED ==="
say "    arms: ${ARM_ORDER[*]}  (cbema gamma=0.8 built-in default)"
say "    convergence: --s1-rounds $S1_ROUNDS --fed-patience-rounds $PATIENCE --local-epochs $LOCAL_EPOCHS --batch $BATCH"

if [[ -n "${DRYRUN:-}" ]]; then
  n=0
  for j in "${JOBS[@]}"; do
    IFS='|' read -r name out cmd <<<"$j"; n=$((n+1))
    [[ -f "$out" ]] && tag="[skip:exists]" || tag="[run]"
    if [[ $n -le 12 || -n "${DRYRUN_ALL:-}" ]]; then printf '  %3d  %-28s %-14s\n' "$n" "$name" "$tag"; fi
    [[ $n -eq 13 && -z "${DRYRUN_ALL:-}" ]] && echo "  ... (DRYRUN_ALL=1 to list all)"
  done
  echo "  ($n jobs; unset DRYRUN to run)"
  echo "  sample cmd: $(IFS='|' read -r _ _ c <<<"${JOBS[0]}"; echo "$c" | tr -s ' ')"
  exit 0
fi

# Liveness watchdog -- see the long note in run_ucrsplit.sh. The main sweep's watchdog exits
# when its own orchestrator goes away, so the cb sweep needs its own.
WD_ORCH='bash scripts/run_ucrsplit_cb\.sh'
if [[ -z "${NO_WATCHDOG:-}" ]] && ! pgrep -af 'gpu_stall_watchdog\.sh' 2>/dev/null | grep -q 'run_ucrsplit_cb\\\.sh'; then
  WATCH_PATTERN='pipeline/federated_eval\.py --dataset ucr_split' \
  WATCH_LOG="$LOGDIR/_STALL_WATCHDOG_cb.log" \
  WATCH_WINDOW="${WATCH_WINDOW:-1200}" WATCH_ACTION="${WATCH_ACTION:-warn}" \
    setsid nohup bash "$REPO/scripts/gpu_stall_watchdog.sh" "$WD_ORCH" > /dev/null 2>&1 < /dev/null &
  say "liveness watchdog started -> $LOGDIR/_STALL_WATCHDOG_cb.log"
fi

declare -A SLOT_PID SLOT_NAME; SLOT_KEYS=()
for g in "${GPUS[@]}"; do for ((i=0;i<SLOTS_PER_GPU;i++)); do SLOT_PID["$g:$i"]=""; SLOT_NAME["$g:$i"]=""; SLOT_KEYS+=("$g:$i"); done; done
free_slot() { local k; for k in "${SLOT_KEYS[@]}"; do [[ -z "${SLOT_PID[$k]}" ]] && { echo "$k"; return; }; done; }
NFAIL=0
reap() { local k p rc; for k in "${SLOT_KEYS[@]}"; do p="${SLOT_PID[$k]}"; [[ -z "$p" ]] && continue
  if ! kill -0 "$p" 2>/dev/null; then wait "$p" 2>/dev/null; rc=$?
    say "DONE  ${SLOT_NAME[$k]}  (slot $k, rc=$rc)"
    [[ $rc -ne 0 ]] && { NFAIL=$((NFAIL+1)); say "  !! FAILED: $LOGDIR/${SLOT_NAME[$k]}.log"; }
    SLOT_PID[$k]=""; SLOT_NAME[$k]=""; fi; done; }

for job in "${JOBS[@]}"; do
  IFS='|' read -r name out cmd <<<"$job"
  if [[ -f "$out" ]]; then say "SKIP  $name (exists)"; continue; fi
  while :; do reap; slot="$(free_slot)"; [[ -n "$slot" ]] && break; sleep 20; done
  gpu="${slot%%:*}"
  say "START $name -> GPU$gpu (slot $slot)"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    nohup bash -c "$cmd" > "$LOGDIR/$name.log" 2>&1 &
  SLOT_PID[$slot]=$!; SLOT_NAME[$slot]="$name"; sleep 3
done
say "all dispatched; waiting"
while :; do reap; r=0; for k in "${SLOT_KEYS[@]}"; do p="${SLOT_PID[$k]}"; [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null && r=$((r+1)); done; [[ $r -eq 0 ]] && break; sleep 20; done

say "=== convergence audit: any TRUNCATED row is NOT reportable (hard rule 2026-07-24) ==="
"$PY" - "$CKPT" <<'PYEOF' 2>&1 | tee -a "$ORCH"
import json, sys, glob, os
files = sorted(f for f in glob.glob(os.path.join(sys.argv[1], "**", "fed_history.json"), recursive=True)
               if "cb_only" in f)
print(f"  scanned {len(files)} codebook-arm fed_history.json" + ("  -- NONE FOUND" if not files else ""))
trunc = 0
for f in files:
    h = json.load(open(f)).get("stage1", [])
    if not h: continue
    who = "/".join(os.path.relpath(f, sys.argv[1]).split(os.sep)[0:3])
    if h[-1].get("truncated"):
        trunc += 1
        print(f"  {who:48s} rounds->{h[-1].get('round')}  TRUNCATED, NOT CONVERGED")
print(f"  {trunc} truncated / {len(files)} runs")
PYEOF
say "=== done ($NFAIL failed). jsons in $RESDIR ; logs in $LOGDIR ==="
