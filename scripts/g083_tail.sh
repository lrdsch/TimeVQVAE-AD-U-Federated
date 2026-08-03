#!/usr/bin/env bash
# La CODA di `ucr_083` (cb128 + proto_count), tolta a `gpu0_g4` che e' il vero percorso critico.
#
# PERCHE'. Misurato alle 22:50 sui timestamp di `ucr083_v1`: 16 celle dal 01/08 17:42 al 02/08
# 19:03 = **1,6 h per cella su 4 slot, cioe' ~6,4 h per onda**. A `gpu0_g4` restano 14 celle
# (enc 8 + cb128 4 + proto 2) = **3,5 onde ≈ 22 h**, fine verso le **20:00 di domani**: otto ore
# OLTRE la scadenza di mezzogiorno. Anche nell'ipotesi ottimistica (onda da 4 h) finirebbe alle
# 11:00, cioe' sul filo.
#
# `ucr_083` e' il percorso critico vero, non `ucr_086`: me n'ero accorto solo dopo aver
# parallelizzato quest'ultima, perche' guardavo il residuo per serie e non il residuo diviso
# per gli slot del flusso che lo possiede. Sono due numeri diversi e conta il secondo.
#
# PERCHE' PROPRIO cb128 E proto_count. Sono le ULTIME DUE tappe della lista di `gpu0_g4`:
# ci arrivera' dopo aver chiuso `enc` (2 onde, ~8-12 h da adesso, cioe' fra le 07:00 e le 11:00),
# mentre queste 6 celle su 6 slot chiudono in una sola onda, ~5 h. Margine 3-7 h, e quando ci
# arrivera' trovera' i report.json e saltera'. Nessuna uccisione, nessuna corsa.
#
# Prendere `enc` invece li incrocerebbe: `gpu0_g4` ci sta dentro ADESSO con 4 job a p4.
#
# Le opzioni --extra sono copiate alla lettera da scripts/g4_chain_gpu0.sh righe 36-39: una
# differenza anche minima qui produrrebbe celle non confrontabili con le altre otto serie.
set -u
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated

export SLOTS_PER_GPU=3           # 2 schede x 3 = 6 slot per 6 celle: una sola onda
bash scripts/launch.sh --cohort ucr083 --tag ucr083_cb128 \
  --arms federated_cb_only,federated_fedavg_cb_only --extra "--codebook-size 128"
bash scripts/launch.sh --cohort ucr083 --tag ucr083_proto_count \
  --arms federated_enc_fedproto --extra "--fedproto-agg count"
echo "=== [$(date '+%F %T')] coda ucr_083 completa (6 celle) ==="
