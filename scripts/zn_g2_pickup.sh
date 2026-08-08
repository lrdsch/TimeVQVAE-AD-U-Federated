#!/usr/bin/env bash
# RACCOGLITORE su g2 — stadio 1 di 2.
#
# La catena Ada finisce `zn_main` verso le 06:58 e il coprifuoco scatta alle 07:00: di fatto
# NON fara' nemmeno una cella di `zn_enc` per le sue tre serie. Quelle 12 celle devono girare
# altrove, e quindi cambieranno architettura QUALUNQUE COSA SI FACCIA (le Ada dopo le 09:00
# non ci sono piu'). Non e' una scelta: e' forzata dal coprifuoco.
#
# Dato che il costo scientifico e' identico in ogni scenario, l'unica variabile che resta e'
# il TEMPO. Simulato sulle code reali:
#   tutte su g2, dopo il coprifuoco   -> ultima cella mer 13:44
#   distribuite fra g2 e g4-3090      -> ultima cella mer ~12:00
#
# Questo stadio prende `ucr_222` e `ucr_229`, che la catena Ada non tocchera' con certezza
# (sono ULTIME nella sua coda `zn_enc`, che non inizia nemmeno). Percio' NON aspetta il
# coprifuoco: parte appena g2 e' libero, ~45 minuti prima.
#
# ⚠ Aspetta comunque che la catena di g2 abbia finito: su g2 il muro e' la CPU (16 core
# condivisi con ssanchez), quindi due catene insieme si danneggerebbero a vicenda.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

SER=${1:-ucr_222,ucr_229}
say() { echo "[$(date -u '+%F %T') UTC][pickup-g2] $*"; }

say "attendo che la catena g2 finisca (serie da raccogliere: $SER)"
while pgrep -f "zn_chain\.sh g2" >/dev/null 2>&1; do sleep 120; done
say "catena g2 libera."

export LAUNCH_ONLY_CLUSTERS="$SER"
export LAUNCH_GPUS="1"                # la 0 e' di ssanchez
export SLOTS_PER_GPU="${SLOTS_PER_GPU:-6}"
export FEDVQ_AMP=fp16
ZN='--window-normalization zscore'

# Solo `zn_enc`: lo `zn_main` di queste serie lo fanno le Ada e sara' gia' a posto.
# launch.sh e' idempotente, quindi se per qualche motivo ne mancasse una la riprende.
say "raccolgo zn_enc su $SER"
bash scripts/launch.sh --cohort ucr2p_10 --tag zn_enc \
  --arms federated_enc_fedavg,federated_enc_fedprox,federated_enc_fedproto,federated_enc_commoninit \
  --extra "$ZN --fed-enc-cb suffstat --fed-enc-prior local --fedproto-agg uniform"

# Il residuo di `zn_main` che le Ada non avessero chiuso (idempotente: se c'e' tutto, salta).
say "controllo il residuo zn_main su $SER"
bash scripts/launch.sh --cohort ucr2p_10 --tag zn_main \
  --arms centralized,local,federated,federated_cb_only,federated_cb_only_ema,federated_fedavg_cb_only \
  --extra "$ZN"
say "STADIO 1 FINITO"
