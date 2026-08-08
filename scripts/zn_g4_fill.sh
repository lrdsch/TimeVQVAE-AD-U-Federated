#!/usr/bin/env bash
# RIEMPITIVO su una 3090 rimasta vuota, senza rischio di doppio dispatch.
#
# IL PROBLEMA. `launch.sh` non ritorna finche' TUTTI i job che ha dispatchato non chiudono.
# Il 2026-08-05 alle 13:30 la coda `zn_main` di g4 era ridotta alle 4 celle di `ucr_222`,
# distribuite su due schede: GPU2 era **completamente vuota** e lo sarebbe restata per ore,
# perche' la catena passa a `zn_enc` solo dopo. Cinque slot su nove fermi.
#
# PERCHE' `zn_norev` E NON `zn_enc`. Due dispatcher che possono pescare la STESSA cella la
# scriverebbero nella stessa cartella di checkpoint -- e la catena principale, appena
# liberata, dispatchera' `zn_enc` su tutte e 8 le sue serie, quindi non esiste un
# sottoinsieme di `zn_enc` disgiunto da lei. `zn_norev` invece ha **tag diverso E arm
# diverso** da qualunque cosa la catena principale abbia in coda: la collisione non e'
# improbabile, e' impossibile.
#
# ⚠ OMOGENEITA' HARDWARE. Solo serie il cui partner `cb_only_ema` e' girato su 3090: il
# contrasto `norevive - cb_only_ema` e' DENTRO la serie, e deve restare sulla stessa
# architettura. ucr_222/229 sono escluse apposta (partner su Ada o su g2).
#
# ⚠ BUDGET CHIUSO. Tante celle quante slot: finito il primo giro il dispatcher esce, invece
# di continuare a competere con la catena principale quando quella arriva a `zn_enc` e
# rimette 3 job per scheda (oltre i 3/GPU misurati il throughput CALA).
#
#   bash scripts/zn_g4_fill.sh [gpu] [slot] [serie,csv]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

export LAUNCH_GPUS="${1:-2}"
export SLOTS_PER_GPU="${2:-3}"
export LAUNCH_ONLY_CLUSTERS="${3:-ucr_014,ucr_043,ucr_082}"
export NO_MPS=1
export FEDVQ_AMP=fp16

say() { echo "[$(date -u '+%F %T') UTC | $(TZ=Europe/Madrid date '+%H:%M') Madrid][riempitivo] $*"; }
say "GPU $LAUNCH_GPUS · $SLOTS_PER_GPU slot · serie $LAUNCH_ONLY_CLUSTERS · tag zn_norev"

bash scripts/launch.sh --cohort ucr2p_10 --tag zn_norev \
  --arms federated_cb_only_ema_norevive --extra "--window-normalization zscore"

say "riempitivo finito (rc=$?)"
