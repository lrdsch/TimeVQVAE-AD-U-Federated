#!/usr/bin/env bash
# FedProto AT THE PRIOR LEVEL. Per-client prior trained on own data + lambda*KL toward
# the ensemble-consensus predictive distribution on a shared probe. No weight averaging.
# Budget matched to the converged federated arms (18/18, local-epochs 2) so it is
# directly comparable to federated_cb_only (the lambda=0 null).
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
LOG=logs/exp_protoprior.log
PW=1.0                 # prototype consensus weight (lambda)
SEEDS=0,1,2
run() {  # dataset cluster
  local DS=$1 CL=$2
  local OUT=artifacts/fed_eval/protoprior/${DS}
  echo "----- protoprior $DS $CL (lambda=$PW) $(date) -----"
  $PY pipeline/federated_eval.py --dataset "$DS" --cluster "$CL" \
      --arms federated_protoprior --seeds "$SEEDS" \
      --s1-rounds 18 --s2-rounds 18 --local-epochs 2 \
      --proto-weight "$PW" --proto-probe-windows 1024 --batch 16 \
      --out-dir "$OUT" --out-json "${OUT}/${CL}_summary.json"
}
{
  echo "===== PROTOPRIOR start $(date) ====="
  for CL in M4_cardiac M1_rotary M2_valve M3_pump M5_bearing M6_drive; do run toy_fed_uni "$CL"; done
  for CL in c3 c0 c2; do run wsd_fed "$CL"; done
  echo "===== PROTOPRIOR done $(date) ====="
} > "$LOG" 2>&1
