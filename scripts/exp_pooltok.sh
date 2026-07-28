#!/usr/bin/env bash
# CELL A of the tokenizer×prior 2×2 — decompose the wsd gap (centralized ≫ fedsgd).
# A POOLED (centralized) tokenizer is trained ONCE, FROZEN, and the fully-shared prior
# is federated at τ optimizer STEPS/round (FedSGD + FedAdam server) over each client's
# own token stream. Same prior budget & τ grid as exp_fedsgd.sh; ONLY the tokenizer
# differs (pooled here vs federated-suffstat there). This gives:
#   centralized − pooltok = pure PRIOR-federation penalty (identical POOLED tokenizer)
#   pooltok     − fedsgd    = pure TOKENIZER penalty        (identical τ prior)
# τ=1 ⇒ the prior is distributed SGD from a common broadcast init ⇒ should recover the
# centralized prior (the sanity upper bound). s1-epochs=35 matches the converged
# centralized skyline tokenizer; the pooled tokenizer is CACHED per (seed,s1e) so the
# sweep trains it once and keeps it bit-identical across τ.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
LOG=logs/exp_pooltok.log
TOTAL=900          # total local prior steps/client, held constant across τ (fair compute)
SEEDS=0,1,2
S1E=35             # pooled tokenizer epochs — matches the converged centralized skyline
{
  echo "===== CELL-A POOLTOK sweep start $(date) ====="
  # wsd first (the dataset where the gap lives), then toy. τ=1 first (the decisive cell).
  for DS_CL in "wsd_fed:c3" "toy_fed_uni:M4_cardiac"; do
    DS=${DS_CL%%:*}; CL=${DS_CL##*:}
    for TAU in 1 2 4 8 16; do
      ROUNDS=$(( TOTAL / TAU ))
      OUT=artifacts/fed_eval/pooltok/${DS}/tau${TAU}
      echo "----- $DS $CL tau=$TAU rounds=$ROUNDS s1e=$S1E (POOLED tok, server=fedadam) $(date) -----"
      $PY pipeline/federated_eval.py --dataset "$DS" --cluster "$CL" \
          --arms federated_fedsgd_pooltok --seeds "$SEEDS" \
          --s1-epochs "$S1E" --local-epochs 2 --s2-rounds "$ROUNDS" \
          --tau-steps "$TAU" --server-opt fedadam --server-lr 0.01 --batch 16 \
          --out-dir "$OUT" --out-json "${OUT}/${CL}_summary.json"
    done
  done
  echo "===== CELL-A POOLTOK sweep done $(date) ====="
} > "$LOG" 2>&1
