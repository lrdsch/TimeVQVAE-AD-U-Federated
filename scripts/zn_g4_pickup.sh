#!/usr/bin/env bash
# RACCOGLITORE su g4-3090 — stadio 2 di 2.
#
# Stadio 1 (`zn_g2_pickup.sh`) prende `ucr_222` e `ucr_229` su g2. Resta `ucr_170`, che e' la
# sola delle tre serie Ada a contare davvero per lo studio (AUPRC autori 0,768, l'unica nella
# banda intermedia; le altre due sono a pavimento).
#
# Perche' su g4-3090 e non su g2: g2 con 6 slot avrebbe in coda anche le 8 celle dello stadio
# 1, e ammucchiarci sopra anche `ucr_170` porta la fine a mer 13:44. Aspettare che le nove
# slot delle 3090 si liberino e darle a loro chiude verso mer ~12:00.
#
# ⚠ NON parte prima che la catena delle 3090 abbia finito: 3 job per scheda e' il massimo
# misurato (a 5 il throughput fa 0,93x, cioe' PEGGIO -- vedi la memoria sul tema).
# ⚠ NON tocca le Ada: dopo le 09:00 tornano al dipartimento.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

SER=${1:-ucr_170}
say() { echo "[$(date -u '+%F %T') UTC][pickup-g4] $*"; }

say "attendo che la catena g4-3090 finisca (serie da raccogliere: $SER)"
while pgrep -f "zn_chain\.sh g4" >/dev/null 2>&1; do sleep 180; done
say "catena g4-3090 libera."

export LAUNCH_ONLY_CLUSTERS="$SER"
export LAUNCH_GPUS="1 2 3"            # SOLO le 3090: 0/4/5 sono Ada e sono state restituite
export SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"
export NO_MPS=1
export FEDVQ_AMP=fp16
ZN='--window-normalization zscore'

say "raccolgo zn_enc su $SER"
bash scripts/launch.sh --cohort ucr2p_10 --tag zn_enc \
  --arms federated_enc_fedavg,federated_enc_fedprox,federated_enc_fedproto,federated_enc_commoninit \
  --extra "$ZN --fed-enc-cb suffstat --fed-enc-prior local --fedproto-agg uniform"

say "controllo il residuo zn_main su $SER"
bash scripts/launch.sh --cohort ucr2p_10 --tag zn_main \
  --arms centralized,local,federated,federated_cb_only,federated_cb_only_ema,federated_fedavg_cb_only \
  --extra "$ZN"
say "STADIO 2 FINITO"
