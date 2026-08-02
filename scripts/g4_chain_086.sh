#!/usr/bin/env bash
# Quarto flusso su g4: `ucr_086`, sulla sola GPU 4. È IL RIBILANCIAMENTO, anticipato.
#
# PERCHE' ADESSO E NON QUANDO UNA SCHEDA SI LIBERA. Misurato alle 06:25: solo **4 job padre per
# scheda** (12 in tutto), load 18,3 su 48 core, 2-8 GB di memoria su 24 per scheda. La macchina
# era mezza ferma mentre `chain_g4` aveva ancora 66 celle su 8 slot — aspettare la scheda libera
# avrebbe sprecato ore di capacita' gia' disponibile.
#
# Ripartizione risultante: GPU4 = 4 (chain) + 4 (qui) = 8 · GPU5 = 4 (chain) + 4 (gpu5_g4) = 8 ·
# GPU0 = 4 (gpu0_g4). Venti job su tre schede identiche, dentro il carico che g4 ha gia' retto.
#
# 🔴 COLLISIONE DA CHIUDERE A MANO. `chain_g4` ha in lista ucr_043 e POI ucr_086: se arrivasse
# a ucr_086 mentre questo flusso e' ancora dentro, i due partirebbero sulla stessa cella
# (launch.sh salta solo cio' che ha gia' un report.json al dispatch). **Appena `ucr_043` e'
# completa (30/30), `chain_g4` va UCCISO**: dopo ucr_043 non gli resta altro che ucr_086,
# ucr_170 e ucr_083, tutte e tre coperte dai flussi dedicati.
set -u
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
PY=${PY:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10}

export SLOTS_PER_GPU=4
s=ucr_086; c=ucr086
[ -f "cohorts/$c.json" ] || \
  "$PY" scripts/cohort.py new "$c" --datasets ucr_split,ucr_split_w2p --clusters "$s"

echo "########## [$(date '+%F %T')] g4/GPU4 — SERIE $s (ribilanciamento) ##########"
bash scripts/launch.sh --cohort "$c" --arms paper --tag "${c}_v1"
bash scripts/launch.sh --cohort "$c" --tag "${c}_enc" \
  --arms federated_enc_fedavg,federated_enc_fedprox,federated_enc_fedproto,federated_enc_commoninit
bash scripts/launch.sh --cohort "$c" --tag "${c}_cb128" \
  --arms federated_cb_only,federated_fedavg_cb_only --extra "--codebook-size 128"
bash scripts/launch.sh --cohort "$c" --tag "${c}_proto_count" \
  --arms federated_enc_fedproto --extra "--fedproto-agg count"
echo "=== [$(date '+%F %T')] g4/GPU4: ucr_086 completa (30 celle) ==="
