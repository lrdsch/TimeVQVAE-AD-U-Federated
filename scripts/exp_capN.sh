#!/usr/bin/env bash
# Diversity-vs-quantity probe on wsd c3 (biggest centralized-local gap, +0.30 VUS-PR,
# yet well-sampled at low order). Train a centralized model on only ~3200 DIVERSE
# windows (= a single client's data budget, drawn across all 6 clients). Compare to
# converged local (~3200 own windows/client) and full centralized (~28k windows).
#   cap ~ full  -> diversity/regularization drives the win (local overfits). [user's reading]
#   cap ~ local -> raw data quantity drives it.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
OUT=artifacts/fed_eval/capN/wsd_fed
{
  echo "===== CAP-N probe start $(date) ====="
  $PY pipeline/federated_eval.py --dataset wsd_fed --cluster c3 \
      --arms centralized_cap --seeds 0,1,2 \
      --s1-epochs 35 --s2-epochs 35 --pool-cap 3200 --batch 16 \
      --out-dir "$OUT" --out-json "${OUT}/c3_capN_summary.json"
  echo "===== CAP-N probe done $(date) ====="
} > logs/exp_capN.log 2>&1
