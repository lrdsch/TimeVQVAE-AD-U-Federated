#!/usr/bin/env bash
# Catena SUPPLEMENTARE su g4, sulle tre RTX 4500 Ada (0, 4, 5).
#
# Concessa dall'utente il 2026-08-05 alle 01:20 Madrid, valida FINO ALLE 09:00 Madrid.
# Alle 07:00 UTC `scripts/zn_ada_curfew.sh` la smonta e restituisce le Ada.
#
# ⚠ ARCHITETTURA OMOGENEA. 0/4/5 sono Ada sm_89, 1/2/3 sono 3090 sm_86. Non si mescolano
# dentro una sola invocazione: con cudnn.benchmark=True i kernel scelti a tempo darebbero
# traiettorie di training diverse DENTRO la stessa serie. Per questo e' una catena a parte.
#
# ⚠ SERIE DISGIUNTE. Prende le tre serie che la catena sulle 3090 non ha ancora iniziato e
# che sono ULTIME nella sua coda: le raggiungerebbe fra ~14 h, mentre qui si chiudono in ~8.
# Quando ci arrivera' trovera' l'out-json e le saltera' (launch.sh e' idempotente).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

export LAUNCH_ONLY_CLUSTERS="ucr_170,ucr_222,ucr_229"
export LAUNCH_GPUS="0 4 5"
export SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"   # misurato: oltre 3/GPU senza MPS il throughput CALA
export NO_MPS=1                              # host condiviso: mai un daemon MPS qui
export FEDVQ_AMP=fp16                        # stessa aritmetica delle altre due catene

ZN='--window-normalization zscore'
say() { echo "[$(date -u '+%F %T') UTC][ada] $*"; }

say "serie: $LAUNCH_ONLY_CLUSTERS   gpu: $LAUNCH_GPUS   slot/gpu: $SLOTS_PER_GPU"
say "coprifuoco alle 07:00 UTC (09:00 Madrid)"

bash scripts/launch.sh --cohort ucr2p_10 --tag zn_main \
  --arms centralized,local,federated,federated_cb_only,federated_cb_only_ema,federated_fedavg_cb_only \
  --extra "$ZN"

say "catena Ada finita (rc=$?)"
