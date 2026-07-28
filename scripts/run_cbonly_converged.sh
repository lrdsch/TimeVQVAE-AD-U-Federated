#!/usr/bin/env bash
# run_cbonly_converged.sh — the federated cb_only arm (encoder LOCAL + codebook SHARED
# via suff-stat + prior LOCAL) under the CONVERGED protocol, on every wsd + toy cluster.
#
# One job per CLUSTER (not per client): cb_only's codebook merge needs all of a cluster's
# clients in one process, so — unlike the local arm — it cannot be split per entity.
#
# Matched against the local (0.617 wsd / 0.930 toy) and centralized numbers already on
# disk in artifacts/converged_all/.
#
# ⚠ "The ONLY difference from `local` is the shared codebook" — DELETED 2026-07-27, it was
# false. It holds ONLY for the stage-2 prior (both arms call _build_train_stage2_converged).
# Stage 1 differs in FIVE further ways, and the DIRECTIONS are not all the same:
#   (a) round-sum accumulation vs local's per-step EMA (γ=0.99, ~100-minibatch window)
#   (b) common random broadcast init vs local's per-client k-means seeding (~1 round of
#       recovery out of 37-93)
#   (c) once-per-round jittered-centroid revival vs local's per-step reseed from raw
#       samples — fires in 5.0% of rounds (187/3723), ~20x rarer
#   (d) constant lr=1e-3 vs local's 10% warmup + cosine-to-zero, 2000-step patience,
#       best-on-val restore
#   (e) an UNMATCHED per-client optimizer-step budget, up to ~6x in BOTH directions within
#       one cluster (20/31 local clients hit the 10_000-step cap at config.py:255; cb_only
#       runs rounds x local_epochs x batches_per_epoch, e.g. 3600 vs 10000 on kpi_015 but
#       24k-61k on kpi_025/089/099)
# (a)-(d) HANDICAP the federated arm, so they bound the NEGATIVE claim from below — safe.
# (e) FAVOURS it, so it threatens any positive claim, and because the ratio is a
# deterministic function of series length it will masquerade as a client property in the
# RQ4 regression. State the ratio in the caption of every cb_only/cb_ema vs local table.
#
#   setsid nohup bash scripts/run_cbonly_converged.sh > logs/cbonly_boot.log 2>&1 < /dev/null &
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PY:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10}"
export PIPELINE_PYTHON="$PY"
LOGDIR="$REPO/logs/cbonly_converged"; RESDIR="$REPO/artifacts/cbonly_converged"
mkdir -p "$LOGDIR" "$RESDIR"
ORCH="$LOGDIR/_orchestrator.log"
GPUS=(0 1); SLOTS_PER_GPU="${SLOTS_PER_GPU:-4}"
SEEDS="${SEEDS:-0}"
S1_ROUNDS="${S1_ROUNDS:-30}"; LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"   # stage1 ~9k steps; select_on_val keeps best round

stamp() { date +'%F %T'; }
say() { echo "[$(stamp)] $*" | tee -a "$ORCH"; }

JOBS=()
for cl in c0 c1 c2 c3; do
  JOBS+=("cbonly_wsd_${cl}|$PY -u pipeline/federated_eval.py --dataset wsd_fed --cluster $cl \
--arms federated_cb_only --protocol converged --s1-rounds $S1_ROUNDS --local-epochs $LOCAL_EPOCHS \
--seeds $SEEDS --out-json $RESDIR/wsd_${cl}.json")
done
for cl in M1_rotary M2_valve M3_pump M4_cardiac M5_bearing M6_drive; do
  JOBS+=("cbonly_toy_${cl}|$PY -u pipeline/federated_eval.py --dataset toy_fed_uni --cluster $cl \
--arms federated_cb_only --protocol converged --s1-rounds $S1_ROUNDS --local-epochs $LOCAL_EPOCHS \
--seeds $SEEDS --out-json $RESDIR/toy_${cl}.json")
done

say "=== cb_only converged: ${#JOBS[@]} cluster jobs, GPUs ${GPUS[*]}, ${SLOTS_PER_GPU} slot(s)/GPU, s1_rounds=$S1_ROUNDS local_epochs=$LOCAL_EPOCHS ==="
[[ -n "${DRYRUN:-}" ]] && { for j in "${JOBS[@]}"; do printf '  %-20s %s\n' "${j%%|*}" "$(echo "${j#*|}"|tr -s ' ')"; done; exit 0; }

declare -A SLOT_PID SLOT_NAME; SLOT_KEYS=()
for g in "${GPUS[@]}"; do for ((i=0;i<SLOTS_PER_GPU;i++)); do SLOT_PID["$g:$i"]=""; SLOT_NAME["$g:$i"]=""; SLOT_KEYS+=("$g:$i"); done; done
free_slot() { local k; for k in "${SLOT_KEYS[@]}"; do [[ -z "${SLOT_PID[$k]}" ]] && { echo "$k"; return; }; done; }
reap() { local k p rc; for k in "${SLOT_KEYS[@]}"; do p="${SLOT_PID[$k]}"; [[ -z "$p" ]] && continue
  if ! kill -0 "$p" 2>/dev/null; then wait "$p" 2>/dev/null; rc=$?
    say "DONE  ${SLOT_NAME[$k]}  (slot $k, rc=$rc)"; [[ $rc -ne 0 ]] && say "  !! FAILED: $LOGDIR/${SLOT_NAME[$k]}.log"
    SLOT_PID[$k]=""; SLOT_NAME[$k]=""; fi; done; }

for job in "${JOBS[@]}"; do
  name="${job%%|*}"; cmd="${job#*|}"
  while :; do reap; slot="$(free_slot)"; [[ -n "$slot" ]] && break; sleep 20; done
  gpu="${slot%%:*}"
  say "START $name -> GPU$gpu (slot $slot)"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup bash -c "$cmd" > "$LOGDIR/$name.log" 2>&1 &
  SLOT_PID[$slot]=$!; SLOT_NAME[$slot]="$name"; sleep 5
done
say "all dispatched; waiting"
while :; do reap; r=0; for k in "${SLOT_KEYS[@]}"; do p="${SLOT_PID[$k]}"; [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null && r=$((r+1)); done; [[ $r -eq 0 ]] && break; sleep 20; done
say "=== CB_ONLY SWEEP COMPLETE ==="
