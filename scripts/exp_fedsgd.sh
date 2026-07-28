#!/usr/bin/env bash
# #2 Small-tau FedSGD (R1-vero): frozen suff-stat tokenizer, fully-shared prior,
# tau optimizer STEPS/round + FedAdam server optimizer. Sweep tau to map the knee
# where step-scale federation still beats `local` on VUS-PR. tau=1 ~ centralized.
# Fixed total local-step budget across tau (fair compute); only comms frequency varies.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
LOG=logs/exp_fedsgd.log
TOTAL=900          # total local steps/client, held constant across tau
SEEDS=0,1,2
{
  echo "===== #2 FEDSGD sweep start $(date) ====="
  for DS_CL in "toy_fed_uni:M4_cardiac" "wsd_fed:c3"; do
    DS=${DS_CL%%:*}; CL=${DS_CL##*:}
    for TAU in 1 2 4 8 16; do
      ROUNDS=$(( TOTAL / TAU ))
      OUT=artifacts/fed_eval/fedsgd/${DS}/tau${TAU}
      echo "----- $DS $CL tau=$TAU rounds=$ROUNDS (server=fedadam) $(date) -----"
      $PY pipeline/federated_eval.py --dataset "$DS" --cluster "$CL" \
          --arms federated_fedsgd --seeds "$SEEDS" \
          --s1-rounds 10 --local-epochs 2 --s2-rounds "$ROUNDS" \
          --tau-steps "$TAU" --server-opt fedadam --server-lr 0.01 --batch 16 \
          --out-dir "$OUT" --out-json "${OUT}/${CL}_summary.json"
    done
  done
  echo "===== #2 FEDSGD sweep done $(date) ====="
} > "$LOG" 2>&1
