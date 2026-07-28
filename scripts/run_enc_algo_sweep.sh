#!/usr/bin/env bash
# run_enc_algo_sweep.sh — the STAGE-1 ENCODER federation trio, swept and matched.
#
#   federated_enc_fedavg    whole encoder, server weight-average          (McMahan 2017)
#   federated_enc_fedprox   the same, + local prox anchor  (μ sweep)      (Li 2020)
#   federated_enc_fedproto  NO weight averaging, per-code prototype pull  (Tan 2022)
#   federated_enc_commoninit  THE NULL: common init, then nothing federated
#
# plus the two reference rows without which none of the four means anything:
#   local                   no federation at all
#   federated_cb_only       codebook federated, encoder NOT — the trio's direct reference
#
# One job per (dataset × cluster × sweep point). Each job writes its OWN json (arms are
# never mixed in one file: scripts/fed_merge_labeled.py relabels one arm per json), and a
# job whose json exists is SKIPPED — the sweep is resumable and interruptible.
#
#   setsid nohup bash scripts/run_enc_algo_sweep.sh > logs/encalgo_boot.log 2>&1 < /dev/null &
#   DRYRUN=1 bash scripts/run_enc_algo_sweep.sh              # print the plan, run nothing
#   PHASE=0 bash scripts/run_enc_algo_sweep.sh               # toy only (HP selection)
#   PHASE=1 MU=0.1 LAMBDA=0.1 bash scripts/run_enc_algo_sweep.sh   # wsd confirmatory
#
# PHASE 0 (default) sweeps μ and λ on TOY ONLY. Pick ONE μ* and ONE λ* from it — on
# stage-1 val loss and token agreement, NOT on test VUS-PR — then run PHASE 1 on wsd with
# exactly those. Sweeping on wsd and reporting the best point is the same error the ledger
# already caught once (the fa_coldstart cherry-pick).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PY:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10}"
export PIPELINE_PYTHON="$PY"
PHASE="${PHASE:-0}"
LOGDIR="$REPO/logs/encalgo_sweep"; RESDIR="$REPO/artifacts/encalgo_sweep"
mkdir -p "$LOGDIR" "$RESDIR"
ORCH="$LOGDIR/_orchestrator.log"
GPUS=(0 1); SLOTS_PER_GPU="${SLOTS_PER_GPU:-4}"        # both GPUs — the reservation is lifted
SEEDS="${SEEDS:-0,1,2}"
# CONVERGENCE, NOT A ROUND COUNT. 30 rounds was measured (2026-07-24) to leave every probed
# cluster still improving, and on wsd c3 it left 4 of 6 entities with a near-random detector
# (VUS-PR 0.16 vs 0.69 once converged). The knee ranges from round 37 to beyond 89 across
# clusters, so no fixed number is right for all of them: ask for a generous ceiling and let
# patience decide where each one actually stops.
S1_ROUNDS="${S1_ROUNDS:-90}"; LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"
PATIENCE="${PATIENCE:-6}"        # rounds without val improvement before declaring convergence
MUS="${MUS:-0.001 0.01 0.1 1 10}"                      # decades: μ's effect is preconditioned
LAMBDAS="${LAMBDAS:-0.01 0.1 1 10}"
PROX_FORMS="${PROX_FORMS:-loss decoupled}"
MU="${MU:-0.1}"; LAMBDA="${LAMBDA:-0.1}"               # PHASE 1: the selected point
# PHASE 1 must also carry the FORM phase 0 selected. Defaulting to `loss` here would ship
# the confirmatory run in the very regime phase 0 may have proven to be a solver no-op.
FORM="${FORM:-loss}"

WSD_CLUSTERS="${WSD_CLUSTERS:-c0 c1 c2 c3}"; WSD_BATCH=128
TOY_CLUSTERS="${TOY_CLUSTERS:-M1_rotary M4_cardiac}"; TOY_BATCH=64   # one large, one small

stamp() { date +'%F %T'; }
say() { echo "[$(stamp)] $*" | tee -a "$ORCH"; }

# --out-dir keeps checkpoints AND fed_history.json under this sweep's own tree instead of
# the shared artifacts/fed_eval/<dataset>/<cluster>. Sweep points cannot collide there
# because the per-arm directory now carries the knobs (federated_enc_fedprox_mu0.1).
CKPT="$RESDIR/ckpt"   # NOTE: --out-dir replaces artifacts/fed_eval/<dataset>, so the
                      # dataset segment must be added here or toy and wsd would share a root.
COMMON="--protocol converged --fed-enc-prior local --s1-rounds $S1_ROUNDS \
--local-epochs $LOCAL_EPOCHS --fed-patience-rounds $PATIENCE --seeds $SEEDS"

JOBS=()
add_job() { JOBS+=("$1|$2|$3"); }     # name | outjson | command

enqueue() {   # $1 ds  $2 cluster  $3 batch  $4 tag  $5 extra-args  $6 arm
  local base="$RESDIR/$1_$2__$4"
  add_job "$1_$2_$4" "${base}.json" \
    "$PY -u pipeline/federated_eval.py --dataset $1 --cluster $2 --arms $6 \
$COMMON --out-dir $CKPT/$1 --batch $3 $5 --out-json ${base}.json"
}

enqueue_cluster() {   # $1 ds  $2 cluster  $3 batch
  local ds="$1" cl="$2" b="$3"
  # ── reference rows (no encoder federation) ──────────────────────────────────
  enqueue "$ds" "$cl" "$b" local          ""  local
  enqueue "$ds" "$cl" "$b" cbonly         ""  federated_cb_only
  enqueue "$ds" "$cl" "$b" commoninit     ""  federated_enc_commoninit
  # ── FedAvg: no knob to sweep, it is the fixed point of the comparison ───────
  enqueue "$ds" "$cl" "$b" fedavg         ""  federated_enc_fedavg
  if [[ "$PHASE" == "0" ]]; then
    local m f l
    for f in $PROX_FORMS; do for m in $MUS; do
      enqueue "$ds" "$cl" "$b" "fedprox_${f}_mu${m}" \
        "--fedprox-mu $m --fedprox-form $f" federated_enc_fedprox
    done; done
    for l in $LAMBDAS; do
      enqueue "$ds" "$cl" "$b" "fedproto_lam${l}" \
        "--fedproto-weight $l" federated_enc_fedproto
      # CANONICAL-FedProto null: count aggregation + count code-weighting reproduces the
      # authors' own per-sample formulation, which in this setting is ALGEBRAICALLY the VQ
      # commitment loss (unit test 8, rel. err 7e-8). If the headline row ties this, the arm
      # is a commitment-weight sweep. This is a required row, not an ablation.
      enqueue "$ds" "$cl" "$b" "fedproto_lam${l}_canonical" \
        "--fedproto-weight $l --fedproto-agg count --fedproto-code-weight count" \
        federated_enc_fedproto
    done
    # Encoder federated, DICTIONARY LOCAL — the variant that keeps `local`'s own quantizer.
    enqueue "$ds" "$cl" "$b" "fedavg_cblocal"  "--fed-enc-cb local" federated_enc_fedavg
    enqueue "$ds" "$cl" "$b" "commoninit_cblocal" "--fed-enc-cb local" federated_enc_commoninit
    for m in $MUS; do
      enqueue "$ds" "$cl" "$b" "fedprox_decoupled_cblocal_mu${m}" \
        "--fed-enc-cb local --fedprox-mu $m --fedprox-form decoupled" federated_enc_fedprox
    done
  else
    enqueue "$ds" "$cl" "$b" "fedprox_${FORM}_mu${MU}" \
      "--fedprox-mu $MU --fedprox-form $FORM" federated_enc_fedprox
    enqueue "$ds" "$cl" "$b" "fedproto_lam${LAMBDA}" "--fedproto-weight $LAMBDA" federated_enc_fedproto
    enqueue "$ds" "$cl" "$b" "fedproto_lam${LAMBDA}_count" \
      "--fedproto-weight $LAMBDA --fedproto-agg count" federated_enc_fedproto
    # rare-code guard: same λ, commitment's own token-frequency weighting
    enqueue "$ds" "$cl" "$b" "fedproto_lam${LAMBDA}_cwcount" \
      "--fedproto-weight $LAMBDA --fedproto-code-weight count" federated_enc_fedproto
  fi
}

if [[ "$PHASE" == "0" ]]; then
  for cl in $TOY_CLUSTERS; do enqueue_cluster toy_fed_uni "$cl" "$TOY_BATCH"; done
else
  for cl in $WSD_CLUSTERS; do enqueue_cluster wsd_fed "$cl" "$WSD_BATCH"; done
fi

say "=== encoder-algo sweep PHASE=$PHASE: ${#JOBS[@]} jobs, GPUs ${GPUS[*]}, ${SLOTS_PER_GPU} slot(s)/GPU, seeds=$SEEDS ==="
if [[ -n "${DRYRUN:-}" ]]; then
  for j in "${JOBS[@]}"; do
    IFS='|' read -r name out cmd <<<"$j"
    [[ -f "$out" ]] && tag="[skip:exists]" || tag="[run]"
    printf '  %-40s %-13s %s\n' "$name" "$tag" "$(echo "$cmd" | tr -s ' ')"
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

# ── per-cluster relabel+merge -> ONE fed_aggregate table ───────────────────────
say "=== aggregating: relabel+merge per cluster ==="
aggregate_cluster() {   # $1 ds  $2 cl
  local ds="$1" cl="$2" pairs=() f tag
  rm -f "$RESDIR/${ds}_${cl}__ALL.json"      # else the previous run's merge is re-merged as arm "ALL"
  for f in "$RESDIR/${ds}_${cl}__"*.json; do
    [[ -f "$f" ]] || continue
    tag="$(basename "$f" .json)"; tag="${tag##*__}"
    [[ "$tag" == "ALL" ]] && continue
    pairs+=("${tag}=${f}")
  done
  [[ ${#pairs[@]} -eq 0 ]] && { say "  no jsons for ${ds}/${cl}"; return; }
  "$PY" scripts/fed_merge_labeled.py --out "$RESDIR/${ds}_${cl}__ALL.json" "${pairs[@]}" >> "$ORCH" 2>&1 \
    && "$PY" scripts/fed_aggregate.py "$RESDIR/${ds}_${cl}__ALL.json" \
         --ref-arm cbonly --primary-contrast fedavg --primary vus_pr \
         > "$LOGDIR/table_${ds}_${cl}.log" 2>&1 \
    && say "  table -> $LOGDIR/table_${ds}_${cl}.log" \
    || say "  aggregation FAILED for ${ds}/${cl}"
}
if [[ "$PHASE" == "0" ]]; then
  for cl in $TOY_CLUSTERS; do aggregate_cluster toy_fed_uni "$cl"; done
else
  for cl in $WSD_CLUSTERS; do aggregate_cluster wsd_fed "$cl"; done
fi

# ── the two gates that decide whether the arms mean anything ───────────────────
say "=== sanity gates (read these BEFORE the tables) ==="
"$PY" - "$CKPT" <<'EOF' 2>&1 | tee -a "$ORCH"
import json, sys, glob, os
# fed_history.json lives next to the checkpoints, i.e. under --out-dir (=$CKPT), NOT next
# to the result jsons — this used to glob the wrong tree and silently found nothing.
files = sorted(glob.glob(os.path.join(sys.argv[1], "**", "fed_history.json"), recursive=True))
print(f"  [gate] scanned {len(files)} fed_history.json under {sys.argv[1]}"
      + ("  -- NONE FOUND, the gates below prove nothing" if not files else ""))
for f in files:
    h = json.load(open(f)).get("stage1", [])
    if not h:
        continue
    last = h[-1]
    gr, gap = last.get("prox_grad_ratio"), last.get("proto_agg_gap")
    msgs = []
    if gr is not None and gr == gr and gr < 1e-2:
        msgs.append(f"PROX IS A NO-OP (grad ratio {gr:.1e} < 1e-2) -> raise mu or use --fedprox-form decoupled")
    if gap is not None and gap == gap and gap < 0.05:
        msgs.append(f"PROTO TARGET ~= THE CODEBOOK (agg gap {gap:.1%} < 5%) -> the arm is a reweighted commitment term")
    if msgs:
        print(f"  [gate] {os.path.relpath(f, sys.argv[1])}")
        for m in msgs:
            print(f"         {m}")
EOF
say "=== ENC-ALGO SWEEP COMPLETE (phase $PHASE) ==="
