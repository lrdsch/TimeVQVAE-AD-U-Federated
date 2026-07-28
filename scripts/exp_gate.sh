#!/usr/bin/env bash
# Density-weighted mixture probe (zero training): tests whether a context-marginal
# gated combine of the frozen cb_only priors beats uniform mixture / cb_only.
# ctx_gate is the theoretically-correct gate; target_gate is the broken foil; nk / best
# are controls. Decision: ctx_gate vs cb_only (+MDE 0.087) on wsd.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
LOG=logs/exp_gate.log
{
  echo "===== GATE PROBE start $(date) ====="
  echo "--- wsd_fed (the real test: heterogeneous, near-disjoint support) ---"
  $PY scripts/mixture_eval.py --dataset wsd_fed --clusters c0,c2,c3 --seeds 0,1,2 \
      --combines mixture,best,nk,target_gate,ctx_gate
  echo "--- toy_fed_uni (homogeneous control) ---"
  $PY scripts/mixture_eval.py --dataset toy_fed_uni --clusters all --seeds 0,1,2 \
      --combines mixture,best,nk,target_gate,ctx_gate
  echo "===== GATE PROBE done $(date) ====="
} > "$LOG" 2>&1
