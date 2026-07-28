#!/usr/bin/env bash
# run_converged_all.sh — local + centralized on every dataset, under the CONVERGED
# protocol (val-driven early stopping, warmup+cosine, best-on-val weights).
#
# Job map:
#   toy_fed_uni   6 clusters x {local, centralized}   via federated_eval
#   wsd_fed       4 clusters x {local, centralized}   via federated_eval
#   ucr_ad        centralized pooled over all 248, SCORED on the 3 matched series
#   ucr_ad        local on those same 3 series        via run.py (full 7-script chain)
#
# Scheduling: a fixed pool of slots per GPU, both cards in use. Each job is one
# process pinned with CUDA_VISIBLE_DEVICES; a slot frees when its job exits.
#
#   bash scripts/run_converged_all.sh              # launch
#   tail -f logs/converged_all/_orchestrator.log   # watch
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PY:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10}"
# run.py spawns the 7 step-scripts as SUBPROCESSES and resolves the interpreter as
# `--python` -> $PIPELINE_PYTHON -> "python". There is no `python` on this box (only
# python3 and the venv), so without this every run.py job dies instantly with a bare
# FileNotFoundError from the spawn, not from the training.
export PIPELINE_PYTHON="$PY"
LOGDIR="$REPO/logs/converged_all"
# Machine-readable results. Without --out-json federated_eval only PRINTS its tables,
# so the matched summary and the run provenance (effective batch sizes, protocol,
# amp dtype, git sha) would exist nowhere but a log file.
RESDIR="$REPO/artifacts/converged_all"
mkdir -p "$LOGDIR" "$RESDIR"
ORCH="$LOGDIR/_orchestrator.log"

GPUS=(0 1)
SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"     # 16 cores / (1 main + 4 loader workers) -> 3x2=6 jobs
UCR_EVAL="ucr_001,ucr_002,ucr_003"      # the 3 series that also get a `local` model
SEEDS="${SEEDS:-0}"

stamp() { date +'%F %T'; }
say()   { echo "[$(stamp)] $*" | tee -a "$ORCH"; }

# ── job list: "name|gpu_hint|command" (gpu_hint is ignored; slots decide) ─────
JOBS=()
# Each training job is CHAINED to its own quality pass (`&&`), so the two quality
# stages run in the same slot the moment that cluster's checkpoints exist — no
# dependency graph in the scheduler, and no second GPU-contending wave at the end.
# PLOT_WORKERS stays small: every concurrent job spawns its own stage-2 plot pool.
PLOT_WORKERS="${PLOT_WORKERS:-2}"
qpass() {  # $1=dataset $2=cluster $3=arms $4=entity(optional)
  # --seed takes ONE int while SEEDS may be a list ("0,1"): pass the first only, which
  # is the seed whose checkpoints this job's chained training just wrote.
  echo "$PY -u scripts/quality_pass.py --dataset $1 --cluster $2 --seed ${SEEDS%%,*} \
--arms $3 --plot-workers $PLOT_WORKERS${4:+ --entity $4}"
}

# GRANULARITY: one job per MODEL, not per cluster.
#
# Every `local` model is independent -- training kpi_015 has nothing to do with
# kpi_025 -- but federated_eval trains a cluster's clients SEQUENTIALLY inside one
# process. Packing 5-11 of them into one job makes the sweep's makespan the length of
# the biggest cluster, and slots freed by the small clusters cannot be refilled: the
# tail of the run idles ~29% of the machine. One model per job packs with no holes.
#
# `centralized` is the exception and stays whole-cluster: it POOLS the clients'
# windows, so it genuinely needs all of them in one process.
#
# Cost of this granularity: federated_eval writes checkpoints only after finishing
# every client of an arm, so a killed job loses all of them -- with one client per
# job, that blast radius is a single model instead of eleven.
entities_of() {  # dataset cluster -> entity ids
  "$PY" -c "import json,sys; print(' '.join(json.load(open('data/raw/'+sys.argv[1]+'/clusters.json'))[sys.argv[2]]))" "$1" "$2"
}

for cl in M1_rotary M2_valve M3_pump M4_cardiac M5_bearing M6_drive; do
  for e in $(entities_of toy_fed_uni "$cl"); do
    JOBS+=("toylocal_${e}|$PY -u pipeline/federated_eval.py --dataset toy_fed_uni \
--cluster $cl --clients $e --arms local --protocol converged --seeds $SEEDS \
--out-json $RESDIR/toy_${cl}_local_${e}.json && $(qpass toy_fed_uni "$cl" local "$e")")
  done
  JOBS+=("toycentral_${cl}|$PY -u pipeline/federated_eval.py --dataset toy_fed_uni \
--cluster $cl --arms centralized --protocol converged --seeds $SEEDS \
--out-json $RESDIR/toy_${cl}_central.json && $(qpass toy_fed_uni "$cl" centralized)")
done
for cl in c0 c1 c2 c3; do
  for e in $(entities_of wsd_fed "$cl"); do
    JOBS+=("wsdlocal_${e}|$PY -u pipeline/federated_eval.py --dataset wsd_fed \
--cluster $cl --clients $e --arms local --protocol converged --seeds $SEEDS \
--out-json $RESDIR/wsd_${cl}_local_${e}.json && $(qpass wsd_fed "$cl" local "$e")")
  done
  JOBS+=("wsdcentral_${cl}|$PY -u pipeline/federated_eval.py --dataset wsd_fed \
--cluster $cl --arms centralized --protocol converged --seeds $SEEDS \
--out-json $RESDIR/wsd_${cl}_central.json && $(qpass wsd_fed "$cl" centralized)")
done
# ── ORDER ────────────────────────────────────────────────────────────────────
# With one model per job every job is now bounded by the SAME step ceiling
# (10k stage1 / 50k stage2) and the same batch size, so durations cluster within
# ~2-3x of each other instead of the 5-11x spread of the per-cluster jobs. At ~92
# jobs over ~8 slots that is ~12 waves, and makespan is set by total work / slots,
# not by ordering: the worst an unlucky order can cost is roughly the length of one
# job (~1h) at the very end.
#
# So ordering is worth only a cheap heuristic, and the useful one is not "longest
# first" but "riskiest first": jobs whose cost or correctness I am least sure of go
# at the front, where a failure surfaces in the first wave instead of hour eight.
#   1. ucr_centralized  — pools 248 series, 15.8 GB resident, by far the slowest start
#   2. ucr_local x3     — the run.py chain, the only path that just broke (PIPELINE_PYTHON)
#   3. centralized x10  — one per cluster, pooled loaders, the arm with less mileage
#   4. local x78        — the uniform bulk, packs into whatever slots remain
UCR_JOBS=()
UCR_JOBS+=("ucr_centralized|$PY -u pipeline/federated_eval.py --dataset ucr_ad --cluster pool \
--arms centralized --protocol converged --eval-clients $UCR_EVAL --seeds $SEEDS \
--out-json $RESDIR/ucr_centralized.json && $(qpass ucr_ad pool centralized)")
for e in ucr_001 ucr_002 ucr_003; do
  # -g GPUSLOT is replaced with the slot's PHYSICAL gpu id, not 0: run.py's -g is not an
  # index into the inherited mask, it is a raw device id that CLOBBERS it
  # (run.py:239 `env["CUDA_VISIBLE_DEVICES"] = str(gpu)`), so a hardcoded 0 pins every
  # run.py job to physical GPU 0 whatever slot the scheduler gave it.
  # -t 2 for the same reason: run.py rebuilds its children's env and sets
  # OMP/MKL_NUM_THREADS from --threads (default = all cores), discarding the
  # orchestrator's export. Measured: run.py children ran 15 threads vs 1 elsewhere.
  UCR_JOBS+=("ucr_local_${e}|$PY -u run.py -d ucr_ad:$e -w 1 -g GPUSLOT -t 2 --log-dir $LOGDIR/runpy")
done

CENTRAL_JOBS=(); LOCAL_JOBS=()
for j in "${JOBS[@]}"; do
  case "${j%%|*}" in *central*) CENTRAL_JOBS+=("$j");; *) LOCAL_JOBS+=("$j");; esac
done
JOBS=("${UCR_JOBS[@]}" "${CENTRAL_JOBS[@]}" "${LOCAL_JOBS[@]}")

# JOBS_ONLY="wsd,ucr" keeps only jobs whose name starts with one of the given
# comma-separated prefixes. Lets a second orchestrator pick up the REMAINDER of the
# queue at a different concurrency without duplicating the job definitions.
#
# Valid name prefixes: toylocal_ toycentral_ wsdlocal_ wsdcentral_ ucr_
# NB "wsd_" matches NOTHING — the names are wsdlocal_*/wsdcentral_*, with no underscore
# after wsd. A prefix that matches nothing is therefore treated as an ERROR rather than
# silently yielding an empty queue that still reports "SWEEP COMPLETE".
if [[ -n "${JOBS_ONLY:-}" ]]; then
  IFS=',' read -ra _pref <<< "$JOBS_ONLY"
  _bad=()
  for p in "${_pref[@]}"; do
    _n=0
    for job in "${JOBS[@]}"; do [[ "${job%%|*}" == "$p"* ]] && _n=$((_n+1)); done
    [[ $_n -eq 0 ]] && _bad+=("$p")
  done
  if ((${#_bad[@]})); then
    say "JOBS_ONLY: prefix(es) '${_bad[*]}' match NO job — refusing to run."
    say "  valid prefixes: toylocal_ toycentral_ wsdlocal_ wsdcentral_ ucr_"
    exit 2
  fi
  _kept=()
  for job in "${JOBS[@]}"; do
    for p in "${_pref[@]}"; do
      [[ "${job%%|*}" == "$p"* ]] && { _kept+=("$job"); break; }
    done
  done
  JOBS=("${_kept[@]}")
  say "JOBS_ONLY=$JOBS_ONLY -> ${#JOBS[@]} job(s) kept"
fi

say "=== converged sweep: ${#JOBS[@]} jobs, GPUs ${GPUS[*]}, ${SLOTS_PER_GPU} slot(s)/GPU ==="

if [[ -n "${DRYRUN:-}" ]]; then
  for job in "${JOBS[@]}"; do
    printf '  %-22s %s\n' "${job%%|*}" "$(echo "${job#*|}" | tr -s ' ')"
  done
  exit 0
fi

# ── slot scheduler ───────────────────────────────────────────────────────────
# Every slot key is PRE-SEEDED to "". Under `set -u` bash treats an associative
# array with no elements as unbound, so `${SLOT_PID[$k]:-}` would die on the first
# lookup; seeding also means keys are never unset, so `${!SLOT_PID[@]}` is always safe.
declare -A SLOT_PID SLOT_NAME
SLOT_KEYS=()
for g in "${GPUS[@]}"; do
  for ((i=0; i<SLOTS_PER_GPU; i++)); do
    SLOT_PID["$g:$i"]=""; SLOT_NAME["$g:$i"]=""; SLOT_KEYS+=("$g:$i")
  done
done

free_slot() {                      # echoes "gpu:idx" of a free slot, or nothing
  local k
  for k in "${SLOT_KEYS[@]}"; do
    [[ -z "${SLOT_PID[$k]}" ]] && { echo "$k"; return; }
  done
}

reap() {                           # log any slot whose process just exited
  local k p rc
  for k in "${SLOT_KEYS[@]}"; do
    p="${SLOT_PID[$k]}"
    [[ -z "$p" ]] && continue
    if ! kill -0 "$p" 2>/dev/null; then
      wait "$p" 2>/dev/null; rc=$?
      say "DONE  ${SLOT_NAME[$k]}  (slot $k, rc=$rc)"
      [[ $rc -ne 0 ]] && say "  !! FAILED: see $LOGDIR/${SLOT_NAME[$k]}.log"
      SLOT_PID[$k]=""; SLOT_NAME[$k]=""
    fi
  done
}

for job in "${JOBS[@]}"; do
  name="${job%%|*}"; cmd="${job#*|}"
  while :; do
    reap
    slot="$(free_slot)"
    [[ -n "$slot" ]] && break
    sleep 20
  done
  gpu="${slot%%:*}"
  cmd="${cmd//GPUSLOT/$gpu}"       # physical id — see the -g note on the ucr_local jobs
  say "START $name  -> GPU$gpu (slot $slot)"
  # `bash -c` because the command is a `train && quality` CHAIN: dispatching it
  # unquoted would hand "&&" to nohup as a literal argv element instead of running it.
  # OMP pinned low: 6 concurrent jobs x 4 loader workers already saturates 16 cores.
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    nohup bash -c "$cmd" > "$LOGDIR/$name.log" 2>&1 &
  SLOT_PID[$slot]=$!; SLOT_NAME[$slot]="$name"
  sleep 5                                          # stagger CUDA context creation
done

say "all jobs dispatched; waiting for stragglers"
while :; do
  reap
  running=0
  for k in "${SLOT_KEYS[@]}"; do
    p="${SLOT_PID[$k]}"
    [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null && running=$((running+1))
  done
  [[ $running -eq 0 ]] && break
  sleep 20
done
say "=== SWEEP COMPLETE ==="
