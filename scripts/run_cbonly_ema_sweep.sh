#!/usr/bin/env bash
# run_cbonly_ema_sweep.sh — the federated_cb_only_ema arm (server-side codebook EMA)
# swept over γ, matched against federated_cb_only, on every wsd + toy cluster, under the
# CONVERGED protocol. Each (cluster × variant) is a separate job/json; per cluster the
# jsons are relabeled+merged into ONE fed_aggregate table so the γ curve + the ablations
# read off a single comparable table.
#
# Variants per cluster:
#   federated_cb_only            (baseline: no server memory)          -> _federated_cb_only.json
#   federated_cb_only_ema @ γ    (server EMA, γ in $GAMMAS)            -> _cb_ema_g<γ>.json
#   federated_cb_only_ema_norevive @ default γ  (A-ablation: EMA, no dead-code revival)
#
# One job per CLUSTER (not per client): the codebook merge needs all of a cluster's
# clients in one process. RESUMABLE — a job whose json exists is skipped.
#
#   setsid nohup bash scripts/run_cbonly_ema_sweep.sh > logs/cbema_boot.log 2>&1 < /dev/null &
#   DRYRUN=1 bash scripts/run_cbonly_ema_sweep.sh          # print the plan, run nothing
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
LOGDIR="$REPO/logs/cbema_sweep"; RESDIR="$REPO/artifacts/cbema_sweep"
mkdir -p "$LOGDIR" "$RESDIR"
ORCH="$LOGDIR/_orchestrator.log"
GPUS=(0 1); SLOTS_PER_GPU="${SLOTS_PER_GPU:-4}"
SEEDS="${SEEDS:-0,1,2,3}"
S1_ROUNDS="${S1_ROUNDS:-30}"; LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"   # matches run_cbonly_converged
GAMMAS="${GAMMAS:-0 0.5 0.8 0.9}"                                 # the server-EMA decay sweep

# dataset -> "cluster ...":"batch"
WSD_CLUSTERS="c0 c1 c2 c3"; WSD_BATCH=128
TOY_CLUSTERS="M1_rotary M2_valve M3_pump M4_cardiac M5_bearing M6_drive"; TOY_BATCH=64

stamp() { date +'%F %T'; }
say() { echo "[$(stamp)] $*" | tee -a "$ORCH"; }

# Build the job list. Each entry: "name|outjson|command".
JOBS=()
add_job() {  # $1 name  $2 outjson  $3 command
  JOBS+=("$1|$2|$3")
}
COMMON="--protocol converged --s1-rounds $S1_ROUNDS --local-epochs $LOCAL_EPOCHS --seeds $SEEDS"
enqueue_cluster() {  # $1 ds  $2 cl  $3 batch
  local ds="$1" cl="$2" batch="$3"; local base="$RESDIR/${ds}_${cl}"
  add_job "${ds}_${cl}_cbonly" "${base}__federated_cb_only.json" \
    "$PY -u pipeline/federated_eval.py --dataset $ds --cluster $cl --arms federated_cb_only \
$COMMON --batch $batch --out-json ${base}__federated_cb_only.json"
  local g
  for g in $GAMMAS; do
    add_job "${ds}_${cl}_ema_g${g}" "${base}__cb_ema_g${g}.json" \
      "$PY -u pipeline/federated_eval.py --dataset $ds --cluster $cl --arms federated_cb_only_ema \
--cb-server-ema-decay $g $COMMON --batch $batch --out-json ${base}__cb_ema_g${g}.json"
  done
  add_job "${ds}_${cl}_ema_norevive" "${base}__cb_ema_norevive.json" \
    "$PY -u pipeline/federated_eval.py --dataset $ds --cluster $cl --arms federated_cb_only_ema_norevive \
$COMMON --batch $batch --out-json ${base}__cb_ema_norevive.json"
}
for cl in $WSD_CLUSTERS; do enqueue_cluster wsd_fed "$cl" "$WSD_BATCH"; done
for cl in $TOY_CLUSTERS; do enqueue_cluster toy_fed_uni "$cl" "$TOY_BATCH"; done

say "=== cb_only_ema sweep: ${#JOBS[@]} jobs (γ∈{$GAMMAS} + baseline + norevive), GPUs ${GPUS[*]}, ${SLOTS_PER_GPU} slot(s)/GPU, seeds=$SEEDS, s1_rounds=$S1_ROUNDS local_epochs=$LOCAL_EPOCHS ==="
if [[ -n "${DRYRUN:-}" ]]; then
  for j in "${JOBS[@]}"; do
    IFS='|' read -r name out cmd <<<"$j"
    [[ -f "$out" ]] && tag="[skip:exists]" || tag="[run]"
    printf '  %-28s %-11s %s\n' "$name" "$tag" "$(echo "$cmd" | tr -s ' ')"
  done
  exit 0
fi

# ── parallel slot scheduler (GPUs × SLOTS_PER_GPU), resumable ──────────────────
declare -A SLOT_PID SLOT_NAME; SLOT_KEYS=()
for g in "${GPUS[@]}"; do for ((i=0;i<SLOTS_PER_GPU;i++)); do SLOT_PID["$g:$i"]=""; SLOT_NAME["$g:$i"]=""; SLOT_KEYS+=("$g:$i"); done; done
free_slot() { local k; for k in "${SLOT_KEYS[@]}"; do [[ -z "${SLOT_PID[$k]}" ]] && { echo "$k"; return; }; done; }
reap() { local k p rc; for k in "${SLOT_KEYS[@]}"; do p="${SLOT_PID[$k]}"; [[ -z "$p" ]] && continue
  if ! kill -0 "$p" 2>/dev/null; then wait "$p" 2>/dev/null; rc=$?
    say "DONE  ${SLOT_NAME[$k]}  (slot $k, rc=$rc)"; [[ $rc -ne 0 ]] && say "  !! FAILED: $LOGDIR/${SLOT_NAME[$k]}.log"
    SLOT_PID[$k]=""; SLOT_NAME[$k]=""; fi; done; }

for job in "${JOBS[@]}"; do
  IFS='|' read -r name out cmd <<<"$job"
  if [[ -f "$out" ]]; then say "SKIP  $name (exists: $out)"; continue; fi
  while :; do reap; slot="$(free_slot)"; [[ -n "$slot" ]] && break; sleep 20; done
  gpu="${slot%%:*}"
  say "START $name -> GPU$gpu (slot $slot)"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup bash -c "$cmd" > "$LOGDIR/$name.log" 2>&1 &
  SLOT_PID[$slot]=$!; SLOT_NAME[$slot]="$name"; sleep 5
done
say "all dispatched; waiting"
while :; do reap; r=0; for k in "${SLOT_KEYS[@]}"; do p="${SLOT_PID[$k]}"; [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null && r=$((r+1)); done; [[ $r -eq 0 ]] && break; sleep 20; done

# ── per-cluster relabel+merge -> ONE fed_aggregate table (γ curve + ablations) ─
say "=== aggregating: relabel+merge per cluster, then fed_aggregate ==="
aggregate_cluster() {  # $1 ds  $2 cl
  local ds="$1" cl="$2"; local base="$RESDIR/${ds}_${cl}" pairs=() g
  [[ -f "${base}__federated_cb_only.json" ]] && pairs+=("federated_cb_only=${base}__federated_cb_only.json")
  for g in $GAMMAS; do [[ -f "${base}__cb_ema_g${g}.json" ]] && pairs+=("cb_ema_g${g}=${base}__cb_ema_g${g}.json"); done
  [[ -f "${base}__cb_ema_norevive.json" ]] && pairs+=("cb_ema_norevive=${base}__cb_ema_norevive.json")
  [[ ${#pairs[@]} -eq 0 ]] && { say "  no jsons for ${ds}/${cl}"; return; }
  "$PY" scripts/fed_merge_labeled.py --out "${base}__ALL.json" "${pairs[@]}" >> "$ORCH" 2>&1 \
    && "$PY" scripts/fed_aggregate.py "${base}__ALL.json" > "$LOGDIR/table_${ds}_${cl}.log" 2>&1 \
    && say "  table -> $LOGDIR/table_${ds}_${cl}.log" \
    || say "  aggregation FAILED for ${ds}/${cl}"
}
for cl in $WSD_CLUSTERS; do aggregate_cluster wsd_fed "$cl"; done
for cl in $TOY_CLUSTERS; do aggregate_cluster toy_fed_uni "$cl"; done
say "=== CB_ONLY_EMA SWEEP COMPLETE ==="
