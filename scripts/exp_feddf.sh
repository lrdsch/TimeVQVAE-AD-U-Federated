#!/usr/bin/env bash
# #3 FedDF one-shot ensemble distillation of the converged cb_only priors.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
LOG=logs/exp_feddf.log
{
  echo "===== #3 FEDDF start $(date) ====="
  echo "--- toy_fed_uni (all clusters, seeds 0-3, 1500 distill steps) ---"
  $PY scripts/feddf_distill.py --dataset toy_fed_uni --clusters all --seeds 0,1,2,3 \
      --distill-steps 1500 --probe-windows 4096
  echo "--- wsd_fed (c0,c2,c3 complete seeds; incomplete auto-skipped) ---"
  $PY scripts/feddf_distill.py --dataset wsd_fed --clusters c0,c2,c3 --seeds 0,1,2,3 \
      --distill-steps 1500 --probe-windows 4096
  echo "===== #3 FEDDF done $(date) ====="
} > "$LOG" 2>&1
