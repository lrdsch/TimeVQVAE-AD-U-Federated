#!/usr/bin/env bash
# CODA POST-CATENA sulle 3090 di g4 (GPU 1/2/3) — sostituisce SIA il caso g4 di
# `zn_branches.sh` (mai partito) SIA `zn_g4_pickup.sh` (waiter ucciso il 2026-08-05 ~18:15,
# il suo lavoro e' replicato qui identico nel modo `gpu3`).
#
#   bash scripts/zn_g4_after_chain.sh a1     # GPU 1+2: le 6 celle A1 mancanti, tutte in parallelo
#   bash scripts/zn_g4_after_chain.sh gpu3   # GPU 3:   partner di ucr_170 (zn_enc+zn_main), poi norev
#
# ⚠ NON usare `zn_branches.sh g4` dopo questo script: le liste collidono (doppio dispatch
#   sulle stesse celle = stessa cartella di checkpoint). E NON editare zn_branches.sh finche'
#   l'istanza g2 (in attesa) e' viva: bash rilegge lo script dal byte offset salvato.
#
# Il waiter aspetta SOLO la catena 3090 (`zn_chain.sh g4` e il suo launch.sh --tag zn_main):
# un pattern generico su launch.sh — quello di zn_branches — aspetterebbe anche i dispatcher
# A2/rush sulle Ada, che girano per giorni e non toccano le 3090.
#
# Ordine dentro gpu3: PRIMA i partner di ucr_170 (sbloccano i contrasti A1-170 e A2-170 e la
# riga zn_main), POI norev — il suo risultato di meccanismo e' gia' estratto, resta solo la
# detection a convergenza, e l'utente ha chiesto A1/A2 il prima possibile.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

MODE="${1:?a1 | gpu3}"
export SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"   # 3 = massimo misurato su g4 (a 5/GPU fa 0,93x)
export NO_MPS=1
export FEDVQ_AMP=fp16
ZN='--window-normalization zscore'
say() { echo "[$(date -u '+%F %T') UTC | $(TZ=Europe/Madrid date '+%H:%M') Madrid][coda-g4/$MODE] $*"; }

busy() { ps -eo args --no-headers \
         | grep -Eq '^bash scripts/zn_chain\.sh g4|^bash scripts/launch\.sh .*--tag zn_main'; }

if busy; then
  say "aspetto la catena 3090 (zn_chain g4 / launch.sh zn_main), controllo ogni 3 min"
  while busy; do sleep 180; done
fi
say "3090 libere, parto."

case "$MODE" in
  a1)
    # Le 6 serie A1 non coperte dal rush (001/011/229 su Ada-4) ne' da g2 (222).
    # 6 celle su 6 slot: tutte in parallelo al primo colpo.
    export LAUNCH_GPUS="1 2"
    export LAUNCH_ONLY_CLUSTERS="ucr_014,ucr_043,ucr_082,ucr_083,ucr_086,ucr_170"
    say "=== A1 su $LAUNCH_ONLY_CLUSTERS ==="
    bash scripts/launch.sh --cohort ucr2p_10 --tag zn_a1 \
      --arms federated_enc_fedavg \
      --extra "$ZN --fed-enc-cb suffstat --fed-enc-prior local --fed-enc-bn shared"
    say "=== A1 finito (rc=$?) ==="
    ;;
  gpu3)
    export LAUNCH_GPUS="3"
    # 1) partner di ucr_170 — replica ESATTA delle due launch di zn_g4_pickup.sh.
    export LAUNCH_ONLY_CLUSTERS="ucr_170"
    say "=== partner ucr_170: zn_enc ==="
    bash scripts/launch.sh --cohort ucr2p_10 --tag zn_enc \
      --arms federated_enc_fedavg,federated_enc_fedprox,federated_enc_fedproto,federated_enc_commoninit \
      --extra "$ZN --fed-enc-cb suffstat --fed-enc-prior local --fedproto-agg uniform"
    say "=== partner ucr_170: residuo zn_main ==="
    bash scripts/launch.sh --cohort ucr2p_10 --tag zn_main \
      --arms centralized,local,federated,federated_cb_only,federated_cb_only_ema,federated_fedavg_cb_only \
      --extra "$ZN"
    # 2) norev, quota g4 (g2 fa 001/011 per conto suo via zn_branches).
    export LAUNCH_ONLY_CLUSTERS="ucr_014,ucr_043,ucr_222,ucr_229"
    say "=== norev su $LAUNCH_ONLY_CLUSTERS ==="
    bash scripts/launch.sh --cohort ucr2p_10 --tag zn_norev \
      --arms federated_cb_only_ema_norevive --extra "$ZN"
    say "=== coda gpu3 finita (rc=$?) ==="
    ;;
  *) echo "modo sconosciuto '$MODE' (a1|gpu3)" >&2; exit 2;;
esac
