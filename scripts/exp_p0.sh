#!/usr/bin/env bash
# P0 batch — matched-fusion matrix (NaiveAvg) + FedDF capacity probe + c1 fills.
# Sequential on GPU1 (GPU1-only house rule; no cross-GPU parallelism).
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
LOG=logs/exp_p0.log
{
  echo "===== P0 BATCH start $(date) ====="
  echo "--- [1/5] NaiveAvg wsd (all clusters, seeds 0-3) — the linchpin arm ---"
  $PY scripts/naiveavg_eval.py --dataset wsd_fed --clusters all --seeds 0,1,2,3
  echo "--- [2/5] NaiveAvg toy (all clusters, seeds 0-3) ---"
  $PY scripts/naiveavg_eval.py --dataset toy_fed_uni --clusters all --seeds 0,1,2,3
  echo "--- [3/5] FedDF capacity probe wsd (all clusters, seeds 0,1): KL(p̄‖q) ---"
  $PY scripts/feddf_probe.py --dataset wsd_fed --clusters all --seeds 0,1 --distill-steps 1500
  echo "--- [4/5] mixture fill wsd c1 (Ensemble column, 4-cluster matrix) ---"
  $PY scripts/mixture_eval.py --dataset wsd_fed --clusters c1 --seeds 0,1,2,3 --combines mixture,best,mean_nll
  echo "--- [5/5] feddf fill wsd c1 (FedDF column, 4-cluster matrix) ---"
  $PY scripts/feddf_distill.py --dataset wsd_fed --clusters c1 --seeds 0,1,2,3 --distill-steps 1500 --probe-windows 4096
  echo "===== P0 BATCH done $(date) ====="
} > "$LOG" 2>&1
