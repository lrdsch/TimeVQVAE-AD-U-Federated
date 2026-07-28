#!/usr/bin/env bash
# #1 Mixture-of-priors at inference (distribution-space collaboration).
# Inference-only over the converged federated_cb_only checkpoints.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
LOG=logs/exp_mixture.log
{
  echo "===== #1 MIXTURE start $(date) ====="
  echo "--- toy_fed_uni (all clusters, seeds 0-3) ---"
  $PY scripts/mixture_eval.py --dataset toy_fed_uni --clusters all --seeds 0,1,2,3 \
      --combines mixture,best,mean_nll
  echo "--- wsd_fed (c0,c2,c3 complete seeds; incomplete auto-skipped) ---"
  $PY scripts/mixture_eval.py --dataset wsd_fed --clusters c0,c2,c3 --seeds 0,1,2,3 \
      --combines mixture,best,mean_nll
  echo "===== #1 MIXTURE done $(date) ====="
} > "$LOG" 2>&1
