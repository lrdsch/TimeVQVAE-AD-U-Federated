#!/usr/bin/env bash
# RIEMPITIVO A1 sugli slot fermi delle 3090 (2026-08-05 19:50).
#
# PERCHE'. `zn_chain.sh g4` ha esaurito la coda tranne le 3 celle di `ucr_222`, che chiudono
# verso le 03:00; `launch.sh` non ritorna finche' non finiscono, e `zn_g4_after_chain.sh`
# aspetta la sua uscita. Risultato: quattro slot su nove fermi per ~7 h.
#
# COSA PRENDE. Le due celle A1 che il manifesto (`cohorts/zn_ownership.json`) ha appena
# riassegnato da `after_chain a1` a questo riempitivo. Se `after_chain` ci arrivasse prima
# che io abbia chiuso, `launch.sh` salterebbe la cella (out-json) oppure interverrebbe
# `zn_dupe_guard.py`: il rischio e' coperto due volte.
#
# OMOGENEITA'. Il partner di 014/043 e' `zn_enc/enc_fedavg`, che girera' su 3090 quando la
# catena arrivera' allo stadio enc: qui giriamo su 3090, il contrasto resta omogeneo.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10

export LAUNCH_GPUS="${1:-2}"
export SLOTS_PER_GPU="${2:-2}"
export LAUNCH_ONLY_CLUSTERS="$("$PY" scripts/zn_owner.py --owner zn_a1_fill.sh --tag zn_a1)"
export NO_MPS=1
export FEDVQ_AMP=fp16
ZN='--window-normalization zscore'
say() { echo "[$(date -u '+%F %T') UTC | $(TZ=Europe/Madrid date '+%H:%M') Madrid][A1-fill] $*"; }
say "GPU $LAUNCH_GPUS · $SLOTS_PER_GPU slot · serie $LAUNCH_ONLY_CLUSTERS (dal manifesto)"

bash scripts/launch.sh --cohort ucr2p_10 --tag zn_a1 \
  --arms federated_enc_fedavg \
  --extra "$ZN --fed-enc-cb suffstat --fed-enc-prior local --fed-enc-bn shared"
say "riempitivo A1 finito (rc=$?)"
