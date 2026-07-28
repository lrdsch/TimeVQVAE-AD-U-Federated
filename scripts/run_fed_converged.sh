#!/usr/bin/env bash
# =============================================================================
#  CONVERGED per-cluster federated sweep (2026-07-10 redo).
#
#  Replaces the undertrained 3/3/3/3 pilot: this trains each model to a budget
#  proven to converge on the sibling dataset (toy_fed @ 15r/30e gave fed≈local).
#  The federated_eval harness has NO early-stopping, so "trained well" = a fixed
#  budget above the convergence knee + a per-epoch loss trace to VERIFY plateau.
#
#    Budget : s1/s2 epochs=35 (local/centralized), s1/s2 rounds=18 x local_epochs=2
#             (=36 federated local epochs) — above toy_fed's converged 30 and the
#             ~35 epochs the smallest wsd client needs (~1500 steps @ its window count).
#    Arms(6): local, centralized, federated, federated_fedavg_cb (naive-cb ablation),
#             federated_shared (no local head), federated_cb_only (codebook-only FL).
#    Seeds  : 0,1,2,3 (n=4 — n=3 can never clear the Wilcoxon floor).
#
#  Writes to artifacts/fed_eval/CONVERGED/<ds>/ so it never collides with or skips
#  the old 3/3/3/3 results under artifacts/fed_eval/<ds>/.
#
#  RESUMABLE (skips a cluster whose .json exists). A failing cluster is recorded,
#  its truncated json removed, and the sweep continues.
#
#  Usage:  bash scripts/run_fed_converged.sh
# =============================================================================
set -uo pipefail

REPO=/home/leonardo/PhD/TimeVQVAE-AD-U-Federated
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10

# GPU 0 is occupied by another user on this shared node. Pin to GPU 1.
export CUDA_VISIBLE_DEVICES=1
# Paper numbers: detect stays fp32 (DETECT_AMP=fp16 shifts AUROC ~3e-3).
unset DETECT_AMP || true

OUT_ROOT=artifacts/fed_eval/converged
SEEDS=0,1,2,3
S1_EPOCHS=35; S2_EPOCHS=35     # local / centralized arms
S1_ROUNDS=18; S2_ROUNDS=18     # federated arms (x local_epochs)
LOCAL_EPOCHS=2                 # 18 x 2 = 36 federated local epochs/client
ARMS=local,centralized,federated,federated_fedavg_cb,federated_shared,federated_cb_only

cd "$REPO" || exit 1
mkdir -p logs "$OUT_ROOT"

FAILED=()
STARTED=$(date +%s)
stamp() { date '+%Y-%m-%d %H:%M:%S'; }

run_cluster() {
  local ds="$1" cl="$2" batch="$3"
  local outdir="${OUT_ROOT}/${ds}"
  local out="${outdir}/${cl}.json"
  local log="logs/conv_${ds}_${cl}.log"

  if [[ -f "$out" ]]; then
    echo "[$(stamp)] SKIP ${ds}/${cl} — ${out} already exists"
    return 0
  fi

  echo "[$(stamp)] START ${ds}/${cl}  (batch=${batch}, seeds=${SEEDS})"
  local t0=$(date +%s)

  "$PY" pipeline/federated_eval.py \
      --dataset "$ds" --cluster "$cl" \
      --arms "$ARMS" \
      --out-dir "$outdir" \
      --s1-epochs "$S1_EPOCHS" --s2-epochs "$S2_EPOCHS" \
      --s1-rounds "$S1_ROUNDS" --s2-rounds "$S2_ROUNDS" \
      --local-epochs "$LOCAL_EPOCHS" \
      --batch "$batch" --seeds "$SEEDS" \
      --out-json "$out" > "$log" 2>&1
  local rc=$?
  local dt=$(( $(date +%s) - t0 ))

  if [[ $rc -ne 0 ]]; then
    echo "[$(stamp)] FAIL  ${ds}/${cl}  rc=${rc}  after ${dt}s  -> tail of ${log}:"
    tail -15 "$log" | sed 's/^/        | /'
    FAILED+=("${ds}/${cl} (rc=${rc})")
    rm -f "$out"
    return 1
  fi
  echo "[$(stamp)] DONE  ${ds}/${cl}  in $((dt/60))m $((dt%60))s  -> ${out}"
  return 0
}

echo "==============================================================="
echo " CONVERGED federated per-cluster sweep"
echo " started : $(stamp)"
echo " gpu     : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo " out     : ${OUT_ROOT}"
echo " budget  : arms=${ARMS}"
echo "           seeds=${SEEDS} s1/s2_epochs=${S1_EPOCHS}/${S2_EPOCHS}"
echo "           s1/s2_rounds=${S1_ROUNDS}/${S2_ROUNDS} local_epochs=${LOCAL_EPOCHS}"
echo "==============================================================="

# ── 1. synthetic: 6 clusters (batch 64) — cheapest first so timing lands early ──
echo; echo "########## toy_fed_uni (6 synthetic clusters) ##########"
for C in M1_rotary M2_valve M3_pump M4_cardiac M5_bearing M6_drive; do
  run_cluster toy_fed_uni "$C" 64
done

# ── 2. real: 4 clusters (batch 128), ascending cost ───────────────────────────
echo; echo "########## wsd_fed (4 real clusters) ##########"
for C in c3 c0 c2 c1; do
  run_cluster wsd_fed "$C" 128
done

# ── 3. aggregate (one call per dataset; tolerances differ, never share a table) ─
echo; echo "########## aggregation ##########"
for DS in toy_fed_uni wsd_fed; do
  shopt -s nullglob
  JSONS=(${OUT_ROOT}/${DS}/*.json)
  shopt -u nullglob
  if [[ ${#JSONS[@]} -eq 0 ]]; then
    echo "[$(stamp)] no jsons for ${DS} — skipping aggregation"; continue
  fi
  echo "[$(stamp)] aggregating ${#JSONS[@]} cluster(s) of ${DS}"
  "$PY" scripts/fed_aggregate.py "${JSONS[@]}" \
        --csv "${OUT_ROOT}/${DS}.csv" \
        > "logs/aggregate_conv_${DS}.log" 2>&1 \
    && echo "  -> logs/aggregate_conv_${DS}.log  +  ${OUT_ROOT}/${DS}.csv" \
    || { echo "  aggregation FAILED for ${DS}"; FAILED+=("aggregate/${DS}"); }
done

TOTAL=$(( $(date +%s) - STARTED ))
echo; echo "==============================================================="
echo " finished: $(stamp)   total: $((TOTAL/3600))h $(((TOTAL%3600)/60))m"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo " FAILURES (${#FAILED[@]}):"; printf '   - %s\n' "${FAILED[@]}"
  echo "==============================================================="; exit 1
fi
echo " all clusters OK"; echo "==============================================================="
touch "${OUT_ROOT}/.sweep_complete"
exit 0
