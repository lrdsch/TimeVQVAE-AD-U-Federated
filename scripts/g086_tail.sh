#!/usr/bin/env bash
# La CODA delle ablazioni di `ucr_086` (cb128 + proto_count), tolta dal ramo più lungo.
#
# PERCHE'. Alle 16:07 `g086_g4` aveva 20 celle rimaste — il percorso critico — con soli 4 job,
# mentre la GPU 5 si stava liberando (`ucr_170` a 2 celle dalla fine).
#
# PERCHE' PROPRIO cb128 E proto_count. La lista di `g086_g4` è: v1 (6 rimaste), enc (8),
# cb128 (4), proto_count (2). Prendo le ULTIME DUE tappe: quel flusso ci arriverà fra ~9 h
# (deve prima chiudere v1 e enc) mentre queste 6 celle su 4 slot chiudono in ~4 h. Margine
# ampio, e quando ci arriverà troverà i report.json e salterà. Nessuna uccisione programmata.
#
# Prendere `enc` invece li incrocerebbe: `g086_g4` ci arriva fra ~5 h e io ci metterei ~5 h,
# cioè esattamente la sovrapposizione che launch.sh non sa evitare (salta una cella solo se ne
# trova il report.json AL MOMENTO del dispatch, quindi una cella in corso non protegge).
#
# ⚠ Deve stare nel REPO, non in /tmp: g4 vede il repo attraverso il mount sshfs, mentre un
# file copiato con scp finirebbe nel /tmp locale di g4 e `run_on_g4.sh` non lo troverebbe.
# È l'errore che ha fatto morire il primo tentativo, in silenzio.
set -u
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated

bash scripts/launch.sh --cohort ucr086 --tag ucr086_cb128 \
  --arms federated_cb_only,federated_fedavg_cb_only --extra "--codebook-size 128"
bash scripts/launch.sh --cohort ucr086 --tag ucr086_proto_count \
  --arms federated_enc_fedproto --extra "--fedproto-agg count"
echo "=== [$(date '+%F %T')] coda ablazioni ucr_086 completa ==="
