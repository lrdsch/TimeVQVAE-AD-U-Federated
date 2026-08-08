#!/usr/bin/env bash
# g2 — catena di chiusura. Sostituisce sia `zn_chain.sh g2` (MORTA, vedi sotto) sia il
# raccoglitore a due stadi.
#
# ⚠ COSA E' SUCCESSO. Lo screen `zn_g2` e' sparito e con lui il processo della catena, che
# aveva gia' dispatchato i 12 job di `zn_main` ma NON era ancora arrivato a `zn_enc`. I job
# sono sopravvissuti (nohup, quindi orfani ma vivi) e hanno continuato a chiudere celle: da
# fuori sembrava tutto normale. Le 8 celle `zn_enc` di ucr_001/ucr_011 non sarebbero mai
# partite, e nessun controllo se ne sarebbe accorto -- lo stato conta le celle chiuse, non
# quelle che nessuno ha in coda.
#
# ORDINE, e non e' arbitrario:
#   1. `zn_enc` di ucr_001/ucr_011 -- il loro `zn_main` e' girato QUI, su sm_75, quindi il
#      contrasto `enc_* - cb_only` resta su architettura omogenea. Valore scientifico pieno.
#   2. `zn_enc` di ucr_222/ucr_229 -- le Ada non fanno in tempo (coprifuoco alle 09:00) e
#      dopo non ci sono piu': queste celle cambiano architettura in OGNI scenario possibile.
#      Sono anche le due serie a pavimento (AUPRC autori 0,014), escluse dai contrasti.
#
# `ucr_170` va invece a `zn_g4_pickup.sh` sulle 3090: e' l'unica delle tre serie Ada che
# conta (AUPRC autori 0,768) e li' ha 9 slot invece di 6.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

export LAUNCH_GPUS="1"                # la 0 e' di ssanchez
export SLOTS_PER_GPU="${SLOTS_PER_GPU:-6}"
export FEDVQ_AMP=fp16
ZN='--window-normalization zscore'
ENC_ARMS=federated_enc_fedavg,federated_enc_fedprox,federated_enc_fedproto,federated_enc_commoninit
ENC_EXTRA="$ZN --fed-enc-cb suffstat --fed-enc-prior local --fedproto-agg uniform"
say() { echo "[$(date -u '+%F %T') UTC][g2-finish] $*"; }

# ⚠ NON si aspetta l'orfano qui. `ucr_011/federated_cb_only_ema` sta ancora girando dalla
# catena morta, ma e' nel tag `zn_main` e i passi 1-2 lavorano su `zn_enc`: celle diverse,
# nessun conflitto. Aspettarlo prima di tutto lasciava g2 al 16% per ~50 minuti -- misurato.
# L'attesa serve solo prima del passo 3, che tocca `zn_main` e altrimenti ri-dispatcherebbe
# la cella che l'orfano sta gia' facendo (launch.sh salta su out-json, che l'orfano scrive
# solo alla fine).

say "1/3 · zn_enc su ucr_001,ucr_011 (architettura omogenea, valore pieno)"
LAUNCH_ONLY_CLUSTERS="ucr_001,ucr_011" \
  bash scripts/launch.sh --cohort ucr2p_10 --tag zn_enc --arms "$ENC_ARMS" --extra "$ENC_EXTRA"

say "2/3 · zn_enc su ucr_222,ucr_229 (spostamento forzato dal coprifuoco)"
LAUNCH_ONLY_CLUSTERS="ucr_222,ucr_229" \
  bash scripts/launch.sh --cohort ucr2p_10 --tag zn_enc --arms "$ENC_ARMS" --extra "$ENC_EXTRA"

say "attendo l'orfano di zn_main prima della rete di sicurezza"
while ps -eo args | grep -q "[f]ederated_eval.py.*--cluster ucr_011 --arms federated_cb_only_ema"; do
  sleep 120
done
say "orfano chiuso."

say "3/3 · rete di sicurezza: residuo zn_main su tutte le serie di g2 e delle Ada"
LAUNCH_ONLY_CLUSTERS="ucr_001,ucr_011,ucr_222,ucr_229" \
  bash scripts/launch.sh --cohort ucr2p_10 --tag zn_main \
  --arms centralized,local,federated,federated_cb_only,federated_cb_only_ema,federated_fedavg_cb_only \
  --extra "$ZN"
say "G2 COMPLETO"
