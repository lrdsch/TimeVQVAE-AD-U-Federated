#!/usr/bin/env bash
# CODA DELLA CAMPAGNA sulle due Ada rimaste vuote (GPU 0 e 4) — 2026-08-06 01:40.
#
#   bash scripts/zn_ada_tail.sh 0    # zn_enc / ucr_170, i 4 arm dell'encoder
#   bash scripts/zn_ada_tail.sh 4    # zn_a1 / ucr_170, poi zn_norev / ucr_222
#
# PERCHE'. Alle 01:28 la campagna aveva 20 celle in coda e una coda stimata fino a Thu 18:55,
# mentre GPU 0 e 4 erano a ZERO processi: il coprifuoco Ada e' revocato dal 2026-08-05 sera
# (`evidence/.ada_guard_off`) e `zn_a2.sh 0` ha esaurito le sue cinque serie. Sei slot fermi
# per diciassette ore mentre il percorso critico e' pieno.
#
# QUALI CELLE. Le SEI PIU' LONTANE nella coda, non le prime disponibili: prendere le prime
# accorcerebbe la coda di poco e metterebbe due dispatcher a un'ora di distanza sulla stessa
# cella. Queste sono a 9-12 h dal loro vecchio proprietario, e una cella qui dura 2-3 h.
#
#   zn_enc/ucr_170 x4   era di `zn_chain.sh g4` (pid 3180568, ucr_170 e' ULTIMA nella sua
#                       lista dopo 082/083/086: 11 celle su 9 slot davanti, ETA Thu 13:39)
#   zn_a1/ucr_170       era del riempitivo A1 ad hoc (pid 3047111, dietro 082+086: ETA 14:17)
#   zn_norev/ucr_222    era di `zn_g4_after_chain.sh gpu3`, che ancora ASPETTA l'uscita della
#                       catena 3090 (ETA 18:55)
#
# La riassegnazione e' scritta in `cohorts/zn_ownership.json` PRIMA di lanciare, e
# `zn_owner.py --check` conferma che resta una partizione. Se nonostante lo slack il vecchio
# proprietario ci arrivasse comunque, `launch.sh` salta la cella (out-json gia' scritto) e in
# ultima istanza `zn_dupe_guard.py` uccide il piu' giovane: tre difese, non una.
#
# ⚠ CONFONDENTE HARDWARE DA DICHIARARE NEL PAPER. Queste sei celle girano su Ada sm_89,
# mentre i loro partner di serie (zn_main/ucr_170, zn_a2/ucr_170) hanno girato su 3090 sm_86.
# La sonda a 3 replicati del 2026-07-31 non vede differenza fra GPU0 (Ada) e GPU1 (3090) su
# g4, quindi il rischio e' basso — ma va scritto, non dimenticato. Stesso caso delle celle
# A1 del rush (001/011/229), gia' segnate.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10

GPU="${1:?0 | 4}"
export LAUNCH_GPUS="$GPU"
export NO_MPS=1                 # g4 e' condiviso con 20+ utenti: MPS non si tocca mai
export FEDVQ_AMP=fp16
ZN='--window-normalization zscore'
say() { echo "[$(date -u '+%F %T') UTC | $(TZ=Europe/Madrid date '+%H:%M') Madrid][ada-tail/$GPU] $*"; }

# La partizione e' una precondizione, non un commento: se e' rotta non si parte.
"$PY" scripts/zn_owner.py --check >/dev/null || { say "!! partizione ROTTA — non parto"; exit 2; }

case "$GPU" in
  0)
    export SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"
    export LAUNCH_ONLY_CLUSTERS="$("$PY" scripts/zn_owner.py --owner zn_ada_tail.sh --tag zn_enc)"
    say "zn_enc su $LAUNCH_ONLY_CLUSTERS · $SLOTS_PER_GPU slot"
    bash scripts/launch.sh --cohort ucr2p_10 --tag zn_enc \
      --arms federated_enc_fedavg,federated_enc_fedprox,federated_enc_fedproto,federated_enc_commoninit \
      --extra "$ZN --fed-enc-cb suffstat --fed-enc-prior local --fedproto-agg uniform"
    say "zn_enc finito (rc=$?)"
    ;;
  4)
    export SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"
    # A1 per primo: sblocca il contrasto `zn_a1 - zn_enc` a n=10, che e' il numero che manca.
    export LAUNCH_ONLY_CLUSTERS="$("$PY" scripts/zn_owner.py --owner zn_ada_tail.sh --tag zn_a1)"
    say "zn_a1 su $LAUNCH_ONLY_CLUSTERS"
    bash scripts/launch.sh --cohort ucr2p_10 --tag zn_a1 \
      --arms federated_enc_fedavg \
      --extra "$ZN --fed-enc-cb suffstat --fed-enc-prior local --fed-enc-bn shared"
    export LAUNCH_ONLY_CLUSTERS="$("$PY" scripts/zn_owner.py --owner zn_ada_tail.sh --tag zn_norev)"
    say "zn_norev su $LAUNCH_ONLY_CLUSTERS"
    bash scripts/launch.sh --cohort ucr2p_10 --tag zn_norev \
      --arms federated_cb_only_ema_norevive --extra "$ZN"
    say "coda GPU4 finita (rc=$?)"
    ;;
  *) echo "GPU '$GPU' non prevista (0|4)" >&2; exit 2;;
esac
