#!/usr/bin/env bash
# Le 2 celle `ucr043_proto_count` rimaste orfane quando ho fermato il ciclo `g4_chain.sh`.
#
# CONTESTO. Alle 19:53 ho ucciso `g4_chain.sh` (solo il ciclo, non i job) perche' dopo
# `ucr_043` la sua lista proseguiva con ucr_086, ucr_170 e ucr_083 — tutte in corso altrove.
# Fermarlo prima che dispatchasse eliminava la corsa contro il tempo, ma lasciava scoperta
# l'ultima tappa di ucr_043, che il ciclo avrebbe fatto: eccola.
#
# ⚠ DEVE ESSERE UNO SCRIPT, non un comando passato a `run_on_g4.sh`. Il primo tentativo era:
#     run_on_g4.sh "... --extra '--fedproto-agg count'" ...
# e le virgolette singole si sono scontrate con quelle del `bash -lc '...'` dentro
# run_on_g4.sh: su g4 e' arrivato `--extra --fedproto-agg` con `count` perso, e argparse ha
# risposto "expected one argument". Fallimento pulito (niente scritto), ma due job sprecati.
set -u
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
SLOTS_PER_GPU=1 bash scripts/launch.sh --cohort ucr043 --tag ucr043_proto_count \
  --arms federated_enc_fedproto --extra "--fedproto-agg count"
echo "=== [$(date '+%F %T')] ucr043_proto_count completo ==="
