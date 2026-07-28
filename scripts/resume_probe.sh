#!/usr/bin/env bash
# resume_probe.sh — TRAIN THE FEDERATED ARM TO ACTUAL CONVERGENCE, then look at the metric.
#
# THE FACT THIS EXISTS FOR. Every converged `federated_cb_only` run on disk is TRUNCATED, not
# converged: all ten logs/cbonly_converged/*.log end at round 29 with the cohort validation
# loss still falling 1-5% per round and `restored=0` (select_on_val never fires because the
# last round is always the best one seen). So "--protocol converged" describes the baselines
# and the per-client prior, NOT the federated stage-1 encoder.
#
# WHAT THIS ANSWERS, measured on the CURRENT setting, with no appeal to older experiments:
#   1. where does the federated arm actually converge?  (patience stops it there and logs it)
#   2. does reaching convergence change VUS-PR at all?  (the only question that matters)
#
# Six clusters — 3 wsd + 3 toy — spanning cohort size, since the cost of truncation plausibly
# scales with how much data a cohort has:
#     wsd_fed      c0 (5 clients)    c3 (6)         c2 (9)
#     toy_fed_uni  M1_rotary (6)     M5_bearing (8) M4_cardiac (11)
#
# Each job RESUMES its finished 30-round run and continues for up to EXTRA_ROUNDS more,
# stopping as soon as the cohort val flattens (--fed-patience-rounds). The source runs predate
# the resume bundle, so this is a WEIGHTS-ONLY resume: each client's AdamW restarts and the run
# prints a warning saying so. Report it as a warm-started continuation, not as "the same run
# trained longer".
#
#   bash scripts/resume_probe.sh                 # wait for free GPUs, then run
#   DRYRUN=1 bash scripts/resume_probe.sh        # print the plan, touch nothing
#   NOWAIT=1 bash scripts/resume_probe.sh        # start now, do not wait for the GPUs
#   EXTRA_ROUNDS=90 PATIENCE=8 bash scripts/resume_probe.sh
#
# Launch detached so it outlives the shell:
#   mkdir -p logs/resume_probe
#   setsid nohup bash scripts/resume_probe.sh > logs/resume_probe/_boot.log 2>&1 < /dev/null &
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PY:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10}"

EXTRA_ROUNDS="${EXTRA_ROUNDS:-60}"      # generous ceiling; PATIENCE decides the real stop
PATIENCE="${PATIENCE:-6}"               # rounds without val improvement before declaring done
LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"      # same as the runs being continued
SRC_ROUNDS="${SRC_ROUNDS:-30}"          # rounds the SOURCE runs completed (run_cbonly_converged
                                        # used --s1-rounds 30). Labelling only, but without it
                                        # the continuation's history reads as a fresh run.
SRC="${SRC:-$REPO/artifacts/fed_eval}"
OUT="$REPO/artifacts/resume_probe"; LOG="$REPO/logs/resume_probe"
mkdir -p "$OUT" "$LOG"
ORCH="$LOG/_orchestrator.log"
say() { echo "[$(date +'%F %T')] $*" | tee -a "$ORCH"; }

# dataset | cluster | batch | gpu   (alternating GPUs; 3 jobs per card)
PROBES=(
  "wsd_fed|c0|128|0"
  "wsd_fed|c3|128|1"
  "wsd_fed|c2|128|0"
  "toy_fed_uni|M1_rotary|64|1"
  "toy_fed_uni|M5_bearing|64|0"
  "toy_fed_uni|M4_cardiac|64|1"
)

# ── wait for the GPUs ─────────────────────────────────────────────────────────
# The cb_only_ema sweep holds 8 slots across both cards. Poll until no pipeline job remains.
wait_for_gpus() {
  [[ -n "${NOWAIT:-}" ]] && { say "NOWAIT set — starting immediately"; return; }
  local n
  while :; do
    n=$(pgrep -fc "pipeline/federated_eval.py" 2>/dev/null || true); n="${n:-0}"
    [[ "$n" -eq 0 ]] && break
    say "waiting for GPUs: $n pipeline process(es) still running (re-check in 5 min)"
    sleep 300
  done
  say "GPUs free — starting the probe"
}

JOBS=()
for spec in "${PROBES[@]}"; do
  IFS='|' read -r ds cl batch gpu <<<"$spec"
  src="$SRC/$ds/$cl/seed0/federated_cb_only"
  json="$OUT/${ds}_${cl}__cbonly_converged.json"
  if [[ ! -d "$src" ]]; then say "SKIP $ds/$cl — no source run at $src"; continue; fi
  if [[ -f "$json" ]]; then say "SKIP $ds/$cl — already done ($json)"; continue; fi
  JOBS+=("$ds|$cl|$batch|$gpu|$src|$OUT/_src_${ds}_${cl}|$OUT/${ds}_${cl}|$json")
done

say "=== convergence probe: ${#JOBS[@]} job(s), up to +$EXTRA_ROUNDS rounds each, patience=$PATIENCE ==="
if [[ -n "${DRYRUN:-}" ]]; then
  for j in "${JOBS[@]}"; do
    IFS='|' read -r ds cl batch gpu src snap out json <<<"$j"
    echo "  [$ds/$cl GPU$gpu] resume $src"
    echo "        -> $json"
  done
  exit 0
fi

wait_for_gpus

for j in "${JOBS[@]}"; do
  IFS='|' read -r ds cl batch gpu src snap out json <<<"$j"
  # SNAPSHOT the source first. artifacts/fed_eval/<ds>/<cl>/seed0/federated_cb_only is the
  # DEFAULT scratch path, so any later `federated_cb_only` job for this cluster rewrites
  # exactly these files; resuming from a directory being rewritten would splice two runs.
  if [[ ! -d "$snap" ]]; then
    mkdir -p "$snap"; cp -r "$src"/. "$snap"/
    say "snapshot $ds/$cl -> $snap ($(find "$snap" -name stage1.ckpt | wc -l) client ckpts)"
  fi
  say "START $ds/$cl on GPU$gpu (rounds $SRC_ROUNDS -> <=$((SRC_ROUNDS+EXTRA_ROUNDS)), patience $PATIENCE)"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup \
    "$PY" -u pipeline/federated_eval.py --dataset "$ds" --cluster "$cl" \
      --arms federated_cb_only --protocol converged \
      --s1-rounds "$EXTRA_ROUNDS" --local-epochs "$LOCAL_EPOCHS" \
      --fed-patience-rounds "$PATIENCE" --resume-rounds-done "$SRC_ROUNDS" \
      --seeds 0 --batch "$batch" \
      --resume-from "$snap" --out-dir "$out" --out-json "$json" \
      > "$LOG/${ds}_${cl}.log" 2>&1 &
  sleep 10
done

say "all dispatched; waiting for completion"
wait
say "=== probe complete — reading the curves ==="
"$PY" scripts/resume_probe_report.py 2>&1 | tee -a "$ORCH"
