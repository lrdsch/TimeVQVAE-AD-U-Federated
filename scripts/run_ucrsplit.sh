#!/usr/bin/env bash
# run_ucrsplit.sh — the 6 requested arms on ucr_split, trained TO CONVERGENCE, single seed.
#
#   1 local                      one model per client, on its 10/10/20/20/30% shard alone
#   2 centralized                one model on the pooled shards = the ORIGINAL train (skyline)
#   3 federated_enc_fedavg       encoder weight-average
#   4 federated_enc_fedprox      + local prox anchor, OFFICIAL loss form, mu=0.1  (Li 2020)
#   5 federated_enc_fedproto     per-code prototype consensus, OFFICIAL count-agg, lam=0.1 (Tan 2022)
#   6 federated_enc_commoninit   shared init, never synced -- THE NULL (arms 3-5 are
#                                uninterpretable without it)
#
# Scope: 82 clusters x 6 arms = 492 jobs. The 82 are the FAMILY REPRESENTATIVES from
# scripts/ucr_split_clusters.txt -- the UCR archive ships the same signal as `X`,
# `DISTORTEDX` and `NOISEX`, so its 226 usable series are only 90 distinct signals and
# running all 226 would be pseudo-replication, not power. 8 of the 90 (taichidb, tilt*,
# resperation1/4/11, park3m, tiltAPB1) cost >6 h/arm on their own -- 3 of 90 families for
# ~15% of the budget -- and are deferred to a later replication pass.
#
# HYPERPARAMETERS ARE COPIED FROM run_converge60.sh AND NOT SWEPT, so the two tables can sit
# side by side. Verified against logs/converge60/_orchestrator.log (the 18:30:17 boot, not
# the aborted 18:26:50 one whose header said decoupled/mu1.0).
# The ONE deviation is --batch 64 (wsd used 128): the median 10% client here holds 873
# stride-1 windows, which is 7 steps/epoch at 128. Applied to EVERY arm, so local-vs-
# centralized step parity is preserved.
#
# SCHEDULING (see documentation/UCR_SPLIT_EXPERIMENT_PLAN.md sec.6):
#   * LPT -- jobs sorted by predicted cost DESCENDING. The spread is 25x (0.24h -> 5.9h);
#     FIFO leaves a multi-hour single-job tail with 11 idle slots.
#   * ...EXCEPT the first wave, which is seeded with all 6 arms on the 3 cheapest clusters.
#     The user declined a separate calibration probe, so this is the probe folded into the
#     run: within ~20 min every arm has either produced a json or crashed, at a cost of
#     ~1.5 slot-hours instead of a separate 2-3h phase.
#   * A job whose out-json exists is SKIPPED -- kill and restart freely.
#
#   DRYRUN=1 bash scripts/run_ucrsplit.sh                                    # print the plan
#   setsid nohup bash scripts/run_ucrsplit.sh > logs/ucrsplit_boot.log 2>&1 < /dev/null &
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PY:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10}"
export PIPELINE_PYTHON="$PY"

# MPS: always point CUDA at the PRIVATE pipe directory owned by scripts/mps_ctl.sh. Harmless
# when no daemon is up (CUDA falls back to plain time-slicing). What it prevents is silently
# attaching to a daemon on the compiled-in default /tmp/nvidia-mps, which is created 0777
# with a 0666 control socket -- on this shared box every other user's CUDA process lands on
# it automatically and anyone can send it control commands. See scripts/mps_ctl.sh.
MPS_CTL_ENV_ONLY=1 source "$REPO/scripts/mps_ctl.sh"

LOGDIR="$REPO/logs/ucrsplit"; RESDIR="$REPO/artifacts/ucrsplit"
CKPT="$RESDIR/ckpt"
mkdir -p "$LOGDIR" "$RESDIR"
ORCH="$LOGDIR/_orchestrator.log"

GPUS=(0 1); SLOTS_PER_GPU="${SLOTS_PER_GPU:-7}"      # 14 concurrent = the MEASURED optimum
# 7/GPU is not a guess: 2026-07-24 measured ~10.8 stage-1 rounds/min at 7/GPU under MPS
# (~2-3.5x vs no-MPS), load 16-18 on the 16-core CPU. 8/GPU oversubscribes and regresses,
# 6/GPU underuses. MPS MUST be up or this degrades to GPU time-slicing at ~3 rounds/min.
SEED="${SEED:-0}"
# Seed suffix. EMPTY at seed 0 so every existing artifact keeps its name and the
# skip-if-exists guard below still recognises it; non-empty otherwise. Without this the
# filename carries no seed, so `SEED=1 bash scripts/run_ucrsplit.sh` matches the seed-0
# jsons and SKIPS all 492 jobs — a silent no-op that looks like a completed replication.
SFX=""; [[ "$SEED" != "0" ]] && SFX="_s$SEED"
S1_ROUNDS="${S1_ROUNDS:-300}"; LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"; PATIENCE="${PATIENCE:-6}"
BATCH="${BATCH:-64}"
FEDPROX_MU="${FEDPROX_MU:-0.1}"; FEDPROX_FORM="${FEDPROX_FORM:-loss}"
FEDPROTO_LAM="${FEDPROTO_LAM:-0.1}"; FEDPROTO_AGG="${FEDPROTO_AGG:-count}"
CLUSTER_FILE="${CLUSTER_FILE:-$REPO/scripts/ucr_split_clusters.txt}"

# Dataloader workers are deliberately LEFT AT THE CONFIG DEFAULT (4). The 7-slot optimum above
# was measured with that default, so overriding it here would change two variables at once and
# invalidate the calibration. Set DEBUG_NUM_WORKERS explicitly only when re-tuning both together.
[[ -n "${DEBUG_NUM_WORKERS:-}" ]] && export DEBUG_NUM_WORKERS

stamp() { date +'%F %T'; }
say() { echo "[$(stamp)] $*" | tee -a "$ORCH"; }

[[ -f "$CLUSTER_FILE" ]] || { echo "missing $CLUSTER_FILE"; exit 1; }
mapfile -t CLUSTERS < "$CLUSTER_FILE"

COMMON="--protocol converged --fed-enc-prior local --s1-rounds $S1_ROUNDS \
--local-epochs $LOCAL_EPOCHS --fed-patience-rounds $PATIENCE --seeds $SEED --batch $BATCH"

# arm-tag -> "arm-id|extra-args"
declare -A ARM=(
  [local]="local|"
  [centralized]="centralized|"
  [fedavg]="federated_enc_fedavg|"
  [fedprox]="federated_enc_fedprox|--fedprox-mu $FEDPROX_MU --fedprox-form $FEDPROX_FORM"
  [fedproto]="federated_enc_fedproto|--fedproto-weight $FEDPROTO_LAM --fedproto-agg $FEDPROTO_AGG"
  [commoninit]="federated_enc_commoninit|"
)
ARM_ORDER=(local centralized fedavg fedprox fedproto commoninit)

mk_cmd() {  # $1 cluster  $2 arm-tag -> stdout: the command
  local cl="$1" tag="$2"; local spec="${ARM[$tag]}"
  local id="${spec%%|*}" extra="${spec#*|}"
  echo "$PY -u pipeline/federated_eval.py --dataset ucr_split --cluster $cl \
--arms $id $COMMON --out-dir $CKPT $extra --out-json $RESDIR/${cl}__${tag}${SFX}.json"
}

# ── job list: LPT over clusters, with the 3 cheapest x 6 arms hoisted to the front ────
# cost order comes from the plan's mechanistic proxy; recomputed here so the script has no
# hidden dependency on a stale file.
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

# the 3 cheapest = the last 3 of the descending order
SEEDW=("${COST_ORDER[@]: -3}")
is_seed() { local c; for c in "${SEEDW[@]}"; do [[ "$c" == "$1" ]] && return 0; done; return 1; }

JOBS=()
add() { JOBS+=("$1|$RESDIR/$2__$3${SFX}.json|$(mk_cmd "$2" "$3")"); }   # SFX: see above
# wave 0: every arm on the cheapest clusters -> fastest possible failure signal
for cl in "${SEEDW[@]}"; do for a in "${ARM_ORDER[@]}"; do add "${cl}_${a}" "$cl" "$a"; done; done
# then LPT over the rest
for cl in "${COST_ORDER[@]}"; do
  is_seed "$cl" && continue
  for a in "${ARM_ORDER[@]}"; do add "${cl}_${a}" "$cl" "$a"; done
done

say "=== ucrsplit: ${#JOBS[@]} jobs (${#CLUSTERS[@]} clusters x ${#ARM_ORDER[@]} arms), seed=$SEED, GPUs ${GPUS[*]} x $SLOTS_PER_GPU slots ==="
say "    arms: ${ARM_ORDER[*]}"
say "    convergence: --s1-rounds $S1_ROUNDS --fed-patience-rounds $PATIENCE --local-epochs $LOCAL_EPOCHS --batch $BATCH"
say "    fedprox=${FEDPROX_FORM}/mu${FEDPROX_MU}  fedproto=${FEDPROTO_AGG}/lam${FEDPROTO_LAM}  workers=${DEBUG_NUM_WORKERS:-cfg-default}"
say "    order: 6 arms x 3 cheapest clusters first (folded-in probe), then LPT descending"

if [[ -n "${DRYRUN:-}" ]]; then
  n=0
  for j in "${JOBS[@]}"; do
    IFS='|' read -r name out cmd <<<"$j"; n=$((n+1))
    [[ -f "$out" ]] && tag="[skip:exists]" || tag="[run]"
    if [[ $n -le 20 || -n "${DRYRUN_ALL:-}" ]]; then printf '  %3d  %-28s %-14s\n' "$n" "$name" "$tag"; fi
    [[ $n -eq 21 && -z "${DRYRUN_ALL:-}" ]] && echo "  ... (DRYRUN_ALL=1 to list all)"
  done
  echo "  ($n jobs; unset DRYRUN to run)"
  echo "  sample cmd: $(IFS='|' read -r _ _ c <<<"${JOBS[0]}"; echo "$c" | tr -s ' ')"
  exit 0
fi

# ── liveness watchdog ────────────────────────────────────────────────────────────────
# reap() below only tests `kill -0`, so a job that is ALIVE BUT FROZEN looks perfectly
# healthy to it and holds its slot forever. That is not hypothetical: on 2026-07-28 the MPS
# server died, all 14 jobs blocked in a socket poll with no error and no log line, and the
# sweep sat dead for 9.9 h (~139 slot-hours) before anyone looked. The watchdog measures
# consumed CPU jiffies, which no frozen process can fake. NO_WATCHDOG=1 to skip.
WD_ORCH='bash scripts/run_ucrsplit\.sh'
if [[ -z "${NO_WATCHDOG:-}" ]] && ! pgrep -af 'gpu_stall_watchdog\.sh' 2>/dev/null | grep -q 'run_ucrsplit\\\.sh'; then
  WATCH_PATTERN='pipeline/federated_eval\.py --dataset ucr_split' \
  WATCH_LOG="$LOGDIR/_STALL_WATCHDOG.log" \
  WATCH_WINDOW="${WATCH_WINDOW:-1200}" WATCH_ACTION="${WATCH_ACTION:-warn}" \
    setsid nohup bash "$REPO/scripts/gpu_stall_watchdog.sh" "$WD_ORCH" > /dev/null 2>&1 < /dev/null &
  say "liveness watchdog started -> $LOGDIR/_STALL_WATCHDOG.log (window ${WATCH_WINDOW:-1200}s, action ${WATCH_ACTION:-warn})"
else
  say "liveness watchdog: ${NO_WATCHDOG:+disabled by NO_WATCHDOG}${NO_WATCHDOG:-already running}"
fi

# ── parallel slot scheduler (GPUs x SLOTS_PER_GPU), resumable ────────────────────────
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
files = sorted(glob.glob(os.path.join(sys.argv[1], "**", "fed_history.json"), recursive=True))
print(f"  scanned {len(files)} fed_history.json" + ("  -- NONE FOUND" if not files else ""))
trunc = 0
for f in files:
    h = json.load(open(f)).get("stage1", [])
    if not h: continue
    last = h[-1]
    who = "/".join(os.path.relpath(f, sys.argv[1]).split(os.sep)[0:3])
    if last.get("truncated"):
        trunc += 1
        print(f"  {who:48s} rounds->{last.get('round')}  TRUNCATED, NOT CONVERGED")
print(f"  {trunc} truncated / {len(files)} runs")
PYEOF
say "=== done ($NFAIL failed). jsons in $RESDIR ; logs in $LOGDIR ==="
say "    next: $PY scripts/ucrsplit_aggregate.py --res $RESDIR --csv $RESDIR/pairs.csv"
say "    NOT fed_aggregate.py --primary vus_pr: on ucr_split VUS-PR is capped by the 250-point"
say "    threshold grid (ucr_191: AUPRC 0.994 vs VUS-PR 0.211) and correlates +0.766 with the"
say "    positive rate, so ordering clusters by it INVERTS detector quality. The cluster — not"
say "    the client — is the unit; ucrsplit_aggregate.py enforces both."
