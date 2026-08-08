#!/usr/bin/env bash
# Catena della campagna z-norm, una per host.
#
# Divisione per SERIE, non per arm. g2 e' sm_75 e g4 sm_86: con cudnn.benchmark=True i
# kernel scelti a tempo cambiano la traiettoria di training, quindi dividendo per arm il
# confondente hardware finirebbe DENTRO un contrasto appaiato. Dividendo per serie resta
# fra serie, dove i test appaiati (che sono tutti within-series) non lo vedono.
#
# Una sola coorte `ucr2p_10` -> un solo cohort_fingerprint per entrambe le meta'.
#
#   bash scripts/zn_chain.sh g2
#   bash scripts/zn_chain.sh g4     (via scripts/run_on_g4.sh, dentro screen)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

HOST="${1:?serve g2 o g4}"
case "$HOST" in
  g2) export LAUNCH_ONLY_CLUSTERS="ucr_001,ucr_011"
      export LAUNCH_GPUS="1"                       # GPU0 e' di ssanchez
      export SLOTS_PER_GPU="${SLOTS_PER_GPU:-6}"   # 16 core, load ~3.5, 2 thread/job
      ;;
  g4) export LAUNCH_ONLY_CLUSTERS="ucr_014,ucr_043,ucr_082,ucr_083,ucr_086,ucr_170,ucr_222,ucr_229"
      # 1/2/3 sono le tre 3090 (sm_86), TUTTE della stessa architettura -- 0/4/5 sono
      # Ada e per giunta occupate al 100% da altri utenti.
      export LAUNCH_GPUS="1 2 3"
      export SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"   # host condiviso: 9 job, 18 dei 48 core
      export NO_MPS=1                              # MAI un daemon MPS su un host con 20+ utenti
      ;;
  *) echo "host sconosciuto '$HOST' (g2|g4)" >&2; exit 2;;
esac

# La precisione la fissa launch.sh (FEDVQ_AMP=fp16). Esplicitata qui perche' questa catena
# gira su DUE architetture: senza il pin g2 farebbe fp16 e g4 bf16, cioe' due aritmetiche
# diverse dentro la stessa tabella.
export FEDVQ_AMP=fp16

ZN='--window-normalization zscore'
say() { echo "[$(date -u '+%F %T') UTC][zn_chain/$HOST] $*"; }

say "serie: $LAUNCH_ONLY_CLUSTERS"
say "gpu: $LAUNCH_GPUS  slot/gpu: $SLOTS_PER_GPU  amp: $FEDVQ_AMP"

# ── 1) tabella principale — 6 arm ────────────────────────────────────────────────
# centralized (skyline) · local (limite inferiore) · federated (il metodo) ·
# cb_only + cb_only_ema (primitiva a statistiche sufficienti, senza e con memoria server) ·
# fedavg_cb_only (-(A): la primitiva sbagliata). Il contrasto cb_only/_ema vs fedavg_cb_only
# E' il contributo (A).
say "=== 1/2 tabella principale ==="
bash scripts/launch.sh --cohort ucr2p_10 --tag zn_main \
  --arms centralized,local,federated,federated_cb_only,federated_cb_only_ema,federated_fedavg_cb_only \
  --extra "$ZN"
say "=== 1/2 finita (rc=$?) ==="

# ── 2) trio encoder + controllo nullo — 4 arm ────────────────────────────────────
# cb=suffstat e' OBBLIGATO, non scelto: fedproto aggrega un prototipo per CODICE, e con
# dizionari per-client il codice 7 denota cose diverse su client diversi (federated.py:1427
# rifiuta la combinazione). prior=local tiene lo stadio 2 fuori, cosi' l'unica superficie
# federata e' l'encoder e il riferimento e' `federated_cb_only`, gia' prodotto al passo 1.
say "=== 2/2 trio encoder ==="
bash scripts/launch.sh --cohort ucr2p_10 --tag zn_enc \
  --arms federated_enc_fedavg,federated_enc_fedprox,federated_enc_fedproto,federated_enc_commoninit \
  --extra "$ZN --fed-enc-cb suffstat --fed-enc-prior local --fedproto-agg uniform"
say "=== 2/2 finita (rc=$?) ==="

say "CATENA $HOST COMPLETA"
