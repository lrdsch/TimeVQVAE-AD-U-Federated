#!/usr/bin/env bash
# Terzo flusso su g4, sulla sola GPU 5.
#
# PERCHE' ESISTE. Alle 17:42 la GPU 5 era a 1 MiB: la catena principale (LAUNCH_GPUS="4 5")
# stava girando solo il replicato `ucr001_repg4`, 2 sole celle, entrambe finite negli slot 4:0
# e 4:1 perche' il dispatcher riempie la prima scheda prima di passare alla seconda. Finche' il
# replicato non chiude (5-8 h) la 5 resterebbe ferma. Questo flusso la riempie senza toccare la
# catena principale -- e senza toccare il replicato, che e' il numero critico.
#
# PERCHE' `ucr_170`. Come per g4_chain_gpu0.sh (ucr_083), si prende dalla CODA della lista
# della catena principale, perche' due flussi che scrivono lo stesso artifacts/runs collidono
# se partono sulla stessa cella nello stesso istante (launch.sh salta solo cio' che ha gia' un
# report.json al momento del dispatch).
#
# Margine calcolato, non sperato:
#   catena principale -> 014, 043, 086 prima di arrivare a 170: ~20 h l'una in contesa = ~60 h
#   questo flusso     -> ucr_170 da solo su una scheda: ~35-40 h
# Finisce con ~20 h di margine, e le celle verranno saltate pulitamente. Il controllo
# automatico ogni 15 min sorveglia comunque l'incrocio: se la catena principale dovesse
# arrivare a ucr_170 mentre questo flusso e' ancora dentro, va fermata e rilanciata con la
# lista ridotta PRIMA che dispatchi.
set -u
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
PY=${PY:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10}

export SLOTS_PER_GPU=4

s=ucr_170; c=ucr170
echo "########## [$(date '+%F %T')] g4/GPU5 — SERIE $s ##########"
[ -f "cohorts/$c.json" ] || \
  "$PY" scripts/cohort.py new "$c" --datasets ucr_split,ucr_split_w2p --clusters "$s"

bash scripts/launch.sh --cohort "$c" --arms paper --tag "${c}_v1"
bash scripts/launch.sh --cohort "$c" --tag "${c}_enc" \
  --arms federated_enc_fedavg,federated_enc_fedprox,federated_enc_fedproto,federated_enc_commoninit
bash scripts/launch.sh --cohort "$c" --tag "${c}_cb128" \
  --arms federated_cb_only,federated_fedavg_cb_only --extra "--codebook-size 128"
bash scripts/launch.sh --cohort "$c" --tag "${c}_proto_count" \
  --arms federated_enc_fedproto --extra "--fedproto-agg count"

echo "=== [$(date '+%F %T')] g4/GPU5: ucr_170 completa (30 celle) ==="
