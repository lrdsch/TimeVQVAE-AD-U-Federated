#!/usr/bin/env bash
# Secondo flusso su g4, sulla sola GPU 0. Terza scheda concessa dall'utente il 2026-08-01.
#
# PERCHE' LA 0 E NON UN'ALTRA. Su g4 le schede sono 3x RTX 4500 Ada (0, 4, 5) e 3x RTX 3090
# (1, 2, 3). La 2 e' occupata da `jhuertas`. Prendere una 3090 avrebbe messo architetture
# diverse (sm_89 vs sm_86) dentro lo stesso pool: job di durata diversa e, con
# cudnn.benchmark=True che sceglie i kernel a tempo, traiettorie di training diverse dentro la
# stessa serie. La 0 e' l'unica scelta che conserva l'omogeneita' che rendeva sicura 4+5.
#
# PERCHE' UN FLUSSO SEPARATO E NON UNA CATENA PIU' LARGA. Allargare `g4_chain.sh` a tre schede
# avrebbe richiesto di riavviarla, e dentro c'e' il replicato `ucr001_repg4` -- il numero che
# decide se le tabelle di g2 e g4 si possono mescolare, gia' ripartito da zero una volta oggi
# per il deadlock del ponte. Non lo si uccide per guadagnare uno slot.
#
# PERCHE' `ucr_083` E NON UN'ALTRA. E' l'ULTIMA della coda di g4_chain.sh. I due flussi
# scrivono lo stesso `artifacts/runs`, e launch.sh salta una cella solo se ne trova il
# report.json AL MOMENTO DEL DISPATCH: due processi che partono sulla stessa cella nello stesso
# istante si pestano i piedi in silenzio. Prendendo la coda, quando la catena principale
# arrivera' a ucr_083 (fra ~68 h: 014, 043, 086, 170 a ~17 h l'una) qui sara' finita da un
# pezzo e le celle verranno saltate pulitamente. Prendere `ucr_170` avrebbe incrociato i due
# flussi a meta' serie.
set -u
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
PY=${PY:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10}

export SLOTS_PER_GPU=4      # stesso valore per slot della catena principale: 4 job sulla scheda

s=ucr_083; c=ucr083
echo "########## [$(date '+%F %T')] g4/GPU0 — SERIE $s ##########"
[ -f "cohorts/$c.json" ] || \
  "$PY" scripts/cohort.py new "$c" --datasets ucr_split,ucr_split_w2p --clusters "$s"

bash scripts/launch.sh --cohort "$c" --arms paper --tag "${c}_v1"
bash scripts/launch.sh --cohort "$c" --tag "${c}_enc" \
  --arms federated_enc_fedavg,federated_enc_fedprox,federated_enc_fedproto,federated_enc_commoninit
bash scripts/launch.sh --cohort "$c" --tag "${c}_cb128" \
  --arms federated_cb_only,federated_fedavg_cb_only --extra "--codebook-size 128"
bash scripts/launch.sh --cohort "$c" --tag "${c}_proto_count" \
  --arms federated_enc_fedproto --extra "--fedproto-agg count"

echo "=== [$(date '+%F %T')] g4/GPU0: ucr_083 completa (30 celle) ==="
