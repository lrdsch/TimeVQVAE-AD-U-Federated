#!/usr/bin/env bash
# COMPLETE the tokenizer×prior decomposition + the DEPLOYABLE federated-tokenizer methods.
# Three families, all on the frozen/cached POOLED tokenizer or a genuinely-federated one:
#   1. CELL C  federated_pooltok_centralprior — cached POOLED tok + POOLED prior, step-based,
#              effective batch K×16, at the τ=1 budget (900 steps) AND a high budget (5400).
#              C − pooltok = pure PRIOR-federation penalty (identical tok AND budget); removes
#              the batch/budget confound between `centralized` and the FedSGD sweep.
#   2. FEDENC  federated_fedsgd_fedenc — FedAvg the WHOLE encoder (genuinely federated tokenizer,
#              NO data pooling) + the same fully-shared τ FedSGD prior. Deployable method #1.
#   3. ALIGN   federated_fedsgd_align — encoders stay LOCAL, aligned on a shared public probe
#              (permutation-free) + τ FedSGD prior. Deployable method #2.
# fedenc/align at τ∈{1,16} bracket the comms axis. Compare against: pooltok (pooled tok, 0.244),
# fedsgd (fed suffstat tok, 0.0155), local (0.246), centralized (0.13 @3ep / 0.49 @35ep).
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
LOG=logs/exp_complete.log
SEEDS=0,1,2
S1E=35             # cached pooled tokenizer (Cell C loads it; no retrain)
{
  echo "===== COMPLETE (2x2 close + deployable tok-fed) start $(date) ====="
  # wsd first (the gap dataset); cheap+decisive Cell C first, then the federated-tokenizer methods.
  for DS_CL in "wsd_fed:c3" "toy_fed_uni:M4_cardiac"; do
    DS=${DS_CL%%:*}; CL=${DS_CL##*:}

    # ---- CELL C: POOLED prior on the CACHED pooled tokenizer (step-based budget) ----
    for STEPS in 900 5400; do
      OUT=artifacts/fed_eval/complete/${DS}/centralprior_steps${STEPS}
      echo "----- $DS $CL CELL-C centralprior steps=$STEPS s1e=$S1E $(date) -----"
      $PY pipeline/federated_eval.py --dataset "$DS" --cluster "$CL" \
          --arms federated_pooltok_centralprior --seeds "$SEEDS" \
          --s1-epochs "$S1E" --s2-rounds "$STEPS" --batch 16 \
          --out-dir "$OUT" --out-json "${OUT}/${CL}_summary.json"
    done

    # ---- DEPLOYABLE federated tokenizer (fedavg-encoder & probe-align) + τ FedSGD prior ----
    for ARM in federated_fedsgd_fedenc federated_fedsgd_align; do
      for TAU in 1 16; do
        ROUNDS=$(( 900 / TAU ))
        OUT=artifacts/fed_eval/complete/${DS}/${ARM}/tau${TAU}
        echo "----- $DS $CL $ARM tau=$TAU rounds=$ROUNDS $(date) -----"
        $PY pipeline/federated_eval.py --dataset "$DS" --cluster "$CL" \
            --arms "$ARM" --seeds "$SEEDS" \
            --s1-rounds 10 --local-epochs 2 --s2-rounds "$ROUNDS" \
            --tau-steps "$TAU" --server-opt fedadam --server-lr 0.01 --align-weight 0.5 --batch 16 \
            --out-dir "$OUT" --out-json "${OUT}/${CL}_summary.json"
      done
    done
  done
  echo "===== COMPLETE done $(date) ====="
} > "$LOG" 2>&1
