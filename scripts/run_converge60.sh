#!/usr/bin/env bash
# run_converge60.sh — train the 6 requested arms TO CONVERGENCE, single seed, on
# every toy and wsd cluster. One config per arm (no μ/λ sweep, no seed sweep, no
# cblocal variants, no canonical-null rows). 6 arms × (6 toy + 4 wsd) = 60 jobs.
#
# Order (the user's order): each arm is fully dispatched before the next one starts,
# and within an arm the 6 toy clusters go before the 4 wsd clusters.
#   1 fedavg             federated_enc_fedavg         encoder weight-average
#   2 fedprox            federated_enc_fedprox        OFFICIAL loss form, μ=0.1  (Li 2020)
#   3 fedproto           federated_enc_fedproto       OFFICIAL count-agg, λ=0.1  (Tan 2022)
#   4 commoninit         federated_enc_commoninit     shared init, never synced (null)
#   5 cb_only            federated_cb_only            codebook merge, enc/dec/prior local
#   6 cb_only_ema γ=0.8  federated_cb_only_ema        server codebook EMA, γ=0.8 (built-in default)
#
# OFFICIAL forms (the published algorithms, no repo-specific variants):
#   fedprox  = paper's proximal-in-loss  +(μ/2)‖w−wᵗ‖²  (--fedprox-form loss, μ=0.1).
#     ⚠ MEASURED CAVEAT, NOT a reason to deviate: this repo keeps one AdamW alive per client,
#       which divides the prox gradient by √v̂, so at μ≤0.1 prox_grad_ratio is 1e-4..5e-3 and
#       fedprox may collapse onto fedavg. We run it AS PUBLISHED and let the logged
#       prox_grad_ratio SHOW whether it degenerated — that is a finding, not a knob to hide.
#   fedproto = paper Eq.6 COUNT-weighted prototype aggregation + Eq.8 per-class-mean form
#       (--fedproto-agg count). In this VQ setting count-weighted p̄_k → the codebook itself,
#       so the prototype pull collapses toward a commitment term; the logged proto_agg_gap
#       measures how far. Again: run as published, measure the degeneracy, don't pre-empt it.
#
# CONVERGENCE, not a round count. 90 was too low (M1_rotary was still descending at r89 in
# the probe), so the ceiling is 300; --fed-patience-rounds stops each run at its own knee and
# the "TRUNCATED, NOT CONVERGED" banner fires only if even 300 was not enough. Resumable: a
# job whose out-json already exists is SKIPPED.
#
#   DRYRUN=1 bash scripts/run_converge60.sh                                   # print the plan
#   setsid nohup bash scripts/run_converge60.sh > logs/converge60_boot.log 2>&1 < /dev/null &
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PY:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10}"
export PIPELINE_PYTHON="$PY"

LOGDIR="$REPO/logs/converge60"; RESDIR="$REPO/artifacts/converge60"
CKPT="$RESDIR/ckpt"                                   # fed_history.json lives under --out-dir
mkdir -p "$LOGDIR" "$RESDIR"
ORCH="$LOGDIR/_orchestrator.log"

GPUS=(0 1); SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"       # both GPUs, 3 jobs each = 6 concurrent
SEED="${SEED:-0}"                                     # SINGLE seed
S1_ROUNDS="${S1_ROUNDS:-300}"; LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"; PATIENCE="${PATIENCE:-6}"
FEDPROX_MU="${FEDPROX_MU:-0.1}"; FEDPROX_FORM="${FEDPROX_FORM:-loss}"       # OFFICIAL FedProx
FEDPROTO_LAM="${FEDPROTO_LAM:-0.1}"; FEDPROTO_AGG="${FEDPROTO_AGG:-count}"  # OFFICIAL FedProto

TOY_CLUSTERS="${TOY_CLUSTERS:-M1_rotary M2_valve M3_pump M4_cardiac M5_bearing M6_drive}"
WSD_CLUSTERS="${WSD_CLUSTERS:-c0 c1 c2 c3}"
TOY_BATCH=64; WSD_BATCH=128

COMMON="--protocol converged --fed-enc-prior local --s1-rounds $S1_ROUNDS \
--local-epochs $LOCAL_EPOCHS --fed-patience-rounds $PATIENCE --seeds $SEED"

stamp() { date +'%F %T'; }
say() { echo "[$(stamp)] $*" | tee -a "$ORCH"; }

# Seed suffix. EMPTY at seed 0 so every existing artifact keeps its name and the
# skip-if-exists guard below still recognises it; non-empty otherwise. Without this the
# filename carries no seed, so `SEED=1 bash scripts/run_converge60.sh` matches the seed-0
# jsons and SKIPS every job — a silent no-op that looks like a completed replication.
SFX=""; [[ "$SEED" != "0" ]] && SFX="_s$SEED"

JOBS=()
add() {   # $1 ds  $2 cluster  $3 batch  $4 arm-tag  $5 arm-id  $6 extra-args
  local base="$RESDIR/$1_$2__$4$SFX"
  JOBS+=("$1_$2_$4|${base}.json|$PY -u pipeline/federated_eval.py --dataset $1 --cluster $2 \
--arms $5 $COMMON --out-dir $CKPT/$1 --batch $3 $6 --out-json ${base}.json")
}

# an arm across ALL clusters (toy 6 then wsd 4) — keeps the arm-major order the user asked for
arm_all_clusters() {   # $1 arm-tag  $2 arm-id  $3 extra-args
  local cl
  for cl in $TOY_CLUSTERS; do add toy_fed_uni "$cl" "$TOY_BATCH" "$1" "$2" "$3"; done
  for cl in $WSD_CLUSTERS; do add wsd_fed     "$cl" "$WSD_BATCH" "$1" "$2" "$3"; done
}

# ORDER (inverted 2026-07-24 per user): the trio + null FIRST, references cb_only/cb_only_ema LAST.
arm_all_clusters fedavg     federated_enc_fedavg       ""
arm_all_clusters fedprox    federated_enc_fedprox      "--fedprox-mu $FEDPROX_MU --fedprox-form $FEDPROX_FORM"
arm_all_clusters fedproto   federated_enc_fedproto     "--fedproto-weight $FEDPROTO_LAM --fedproto-agg $FEDPROTO_AGG"
arm_all_clusters commoninit federated_enc_commoninit   ""
arm_all_clusters cbonly     federated_cb_only         ""
arm_all_clusters cbema      federated_cb_only_ema      ""      # γ=0.8 is the built-in default

say "=== converge60: ${#JOBS[@]} jobs, single seed=$SEED, GPUs ${GPUS[*]} x ${SLOTS_PER_GPU} slots ==="
say "    convergence: --s1-rounds $S1_ROUNDS --fed-patience-rounds $PATIENCE ; fedprox=${FEDPROX_FORM}/μ${FEDPROX_MU} fedproto=λ${FEDPROTO_LAM}"
if [[ -n "${DRYRUN:-}" ]]; then
  n=0
  for j in "${JOBS[@]}"; do
    IFS='|' read -r name out cmd <<<"$j"; n=$((n+1))
    [[ -f "$out" ]] && tag="[skip:exists]" || tag="[run]"
    printf '  %2d  %-32s %-14s\n' "$n" "$name" "$tag"
  done
  echo "  ($n jobs; set DRYRUN= to actually run)"
  exit 0
fi

# ── parallel slot scheduler (GPUs × SLOTS_PER_GPU), resumable, arm-major FIFO ──
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

# ── convergence audit: which runs hit the ceiling still descending (TRUNCATED) ─
say "=== convergence audit (read this: any TRUNCATED row is NOT reportable) ==="
"$PY" - "$CKPT" <<'EOF' 2>&1 | tee -a "$ORCH"
import json, sys, glob, os
files = sorted(glob.glob(os.path.join(sys.argv[1], "**", "fed_history.json"), recursive=True))
print(f"  scanned {len(files)} fed_history.json"
      + ("  -- NONE FOUND" if not files else ""))
for f in files:
    h = json.load(open(f)).get("stage1", [])
    if not h: continue
    last = h[-1]
    who = os.path.relpath(f, sys.argv[1]).split(os.sep)[0:3]
    flag = "TRUNCATED, NOT CONVERGED" if last.get("truncated") else "converged"
    print(f"  {'/'.join(who):40s} rounds->{last.get('round')}  {flag}")
EOF
say "=== done. jsons in $RESDIR ; logs in $LOGDIR ==="
