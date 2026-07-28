#!/usr/bin/env bash
# Clean FedSGD re-run on wsd c3: same stage-1 budget as the converged arms
# (s1-rounds 18, was 10) to REMOVE the weak-tokenizer confound behind the earlier
# ~0.01 collapse. This is the last door of federation-as-joint-training: if even a
# proper tokenizer + adaptive server opt still collapses below cb_only across tau,
# the joint-training door is closed too. tau=1 is the ~centralized-equivalent point.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
TOTAL=600
SEEDS=0,1,2
{
  echo "===== FEDSGD-CLEAN start $(date) ====="
  for TAU in 1 4 16; do
    ROUNDS=$(( TOTAL / TAU ))
    OUT=artifacts/fed_eval/fedsgd_clean/wsd_fed/tau${TAU}
    echo "----- c3 tau=$TAU rounds=$ROUNDS s1-rounds=18 (server=fedadam) $(date) -----"
    $PY pipeline/federated_eval.py --dataset wsd_fed --cluster c3 \
        --arms federated_fedsgd --seeds "$SEEDS" \
        --s1-rounds 18 --local-epochs 2 --s2-rounds "$ROUNDS" \
        --tau-steps "$TAU" --server-opt fedadam --server-lr 0.01 --batch 16 \
        --out-dir "$OUT" --out-json "${OUT}/c3_summary.json"
  done
  echo "===== FEDSGD-CLEAN done $(date) ====="
} > logs/exp_fedsgd_clean.log 2>&1
