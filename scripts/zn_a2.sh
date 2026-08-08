#!/usr/bin/env bash
# RAMO A2 — «il metodo fatto bene»: encoder FedAvg + BN pooled + prior PARZIALMENTE CONDIVISO.
# Autorizzato dall'utente il 2026-08-05 ~18:10 Madrid, insieme alla revoca del coprifuoco Ada
# (tutte e 6 le GPU di g4 in uso nostro).
#
# Gemello di `zn_a1` con UNA sola differenza: `--fed-enc-prior partial` invece di `local`.
# Il razionale (memoria fed-cure-plan-critique-2026-08-05): A1 allinea la lingua (encoder
# identici ⇒ token id coerenti) ma tiene il prior locale, quindi non puo' catturare il premio
# pooling (+0,187 di centralized>local, che vive in stage 2). A2 e' la prima cella del
# progetto in cui mediare il prior e' SENSATO: il guard di federated_eval.py:1802 rifiuta
# prior federato + token id client-specifici, ed e' esattamente la patologia che A1 rimuove.
# Predizione falsificabile: A2 deve atterrare VICINO a centralized, non solo sopra local.
#
# ⚠ TAG NUOVO OBBLIGATORIO: il cohort_fingerprint non copre ne' fed_enc_prior ne' fed_enc_bn
#   (CLAUDE.md) — senza tag distinto queste celle sovrascriverebbero l'out-json di zn_a1.
# ⚠ NIENTE --fedproto-agg: federated_eval esce se un flag passato non ha un arm proprietario.
# ⚠ Prior partial ⇒ il body condiviso del prior si allena a ROUND: la trappola del budget
#   fisso e' chiusa perche' launch.sh pinna --fed-patience-rounds (riga 402).
# ⚠ Una GPU per dispatcher, stessa architettura (Ada), 3 slot = il massimo misurato su g4.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

export LAUNCH_GPUS="${1:?GPU Ada (0 o 5)}"
export LAUNCH_ONLY_CLUSTERS="${2:?lista serie, es. ucr_001,ucr_011}"
export SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"
export NO_MPS=1                                  # host condiviso: mai un daemon MPS qui
export FEDVQ_AMP=fp16                            # stessa aritmetica del resto della campagna

ZN='--window-normalization zscore'
say() { echo "[$(date -u '+%F %T') UTC | $(TZ=Europe/Madrid date '+%H:%M') Madrid][A2] $*"; }
say "GPU $LAUNCH_GPUS · $SLOTS_PER_GPU slot · serie $LAUNCH_ONLY_CLUSTERS"

bash scripts/launch.sh --cohort ucr2p_10 --tag zn_a2 \
  --arms federated_enc_fedavg \
  --extra "$ZN --fed-enc-cb suffstat --fed-enc-prior partial --fed-enc-bn shared"

say "A2 finito su GPU $LAUNCH_GPUS (rc=$?)"
