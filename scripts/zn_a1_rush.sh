#!/usr/bin/env bash
# RAMO A1 SUBITO, su una RTX 4500 Ada di g4 (richiesto dall'utente il 2026-08-05 alle 17:05).
#
# A1 = `federated_enc_fedavg` con `--fed-enc-bn shared`. E' l'unico intervento con una
# previsione teorica precisa: col default `buffers_local` ogni client valuta con una CHIMERA
# (pesi del consenso, statistiche BN sue), che la sonda D2b ha misurato PEGGIORE dell'encoder
# straniero intero. Con `shared` le statistiche sono messe in comune per legge della varianza
# totale e i 5 encoder diventano identici.
#
# ⚠ SERIE SCELTE PER AVERE SUBITO IL CONTRASTO. `ucr_001`, `ucr_011`, `ucr_229` sono le tre
# in cui la cella partner `zn_enc/federated_enc_fedavg` e' GIA' su disco: appena queste
# chiudono, `zn_a1 - zn_enc` e' calcolabile. `ucr_222` resta a g2, la cui catena la fara'.
#
# ⚠ NIENTE DOPPIO DISPATCH. `zn_branches.sh g2` aspetta che non ci sia piu' nessun
# `launch.sh` **su g2**: non vede questo processo, che gira su g4, e senza precauzioni
# potrebbe dispatchare le stesse tre celle nella stessa cartella di checkpoint. Per questo
# la lista A1 di g2 va ridotta a `ucr_222` PRIMA di lanciare qui.
#
# ⚠ CONFONDENTE HARDWARE, DICHIARATO. Il partner di queste tre celle e' girato su g2
# (Quadro RTX 8000, sm_75); qui girano su Ada sm_89. Il contrasto A1 porta quindi dentro
# una differenza di architettura. E' accettato consapevolmente: i tre replicati veri del
# 2026-07-31 avevano misurato il confondente g2<->g4 come invisibile rispetto al rumore, e
# la velocita' di risposta vale piu' della purezza qui. `zn_hardware_history.py` lo segnala.
#
# ⚠ La guardia va ristretta a `4,5` PRIMA di lanciare, o uccide questo job entro 60 s.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

export LAUNCH_GPUS="${1:-0}"                     # una sola Ada
export SLOTS_PER_GPU="${2:-3}"                   # 3 celle in parallelo, il massimo misurato
export LAUNCH_ONLY_CLUSTERS="${3:-ucr_001,ucr_011,ucr_229}"
export NO_MPS=1                                  # host condiviso: mai un daemon MPS qui
export FEDVQ_AMP=fp16                            # stessa aritmetica del resto della campagna

ZN='--window-normalization zscore'
say() { echo "[$(date -u '+%F %T') UTC | $(TZ=Europe/Madrid date '+%H:%M') Madrid][A1-rush] $*"; }
say "GPU $LAUNCH_GPUS · $SLOTS_PER_GPU slot · serie $LAUNCH_ONLY_CLUSTERS"

# NIENTE --fedproto-agg: il suo unico proprietario e' federated_enc_fedproto, e federated_eval
# esce con errore se nessun arm richiesto legge un flag passato.
bash scripts/launch.sh --cohort ucr2p_10 --tag zn_a1 \
  --arms federated_enc_fedavg \
  --extra "$ZN --fed-enc-cb suffstat --fed-enc-prior local --fed-enc-bn shared"

say "A1-rush finito (rc=$?)"
