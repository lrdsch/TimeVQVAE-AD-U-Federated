#!/usr/bin/env bash
# `ucr086_enc` — le 8 celle del percorso critico, in UNA sola onda invece di due.
#
# PERCHE'. Misurato alle 22:40 sui timestamp reali di `ucr086_v1`: 13 celle da 06:23 a 21:25,
# cioe' **1,15 h per cella su 4 slot = ~4,6 h per onda**. Con `g4_chain_086.sh` le 8 celle di
# `enc` sarebbero due onde da 4 = **9,2 h**, e non possono nemmeno iniziare prima che finiscano
# le ultime 3 celle di `v1` (partite alle 21:25, attese verso le 02:00). Fine stimata: **11:15**,
# a quarantacinque minuti dalla scadenza di mezzogiorno. Troppo poco per una notte non
# sorvegliata: qualunque intoppo la manca.
#
# 9 slot su 3 schede fanno entrare tutte e 8 le celle nella stessa onda: ~4,6 h, fine ~03:30.
#
# PERCHE' 3 SLOT PER SCHEDA E NON 8 SU UNA. La memoria non e' il vincolo — misurato: ogni job
# occupa 436-1488 MiB su schede da 24 GB, con oltre 20 GB liberi per scheda. Il muro e' la CPU
# (load ~20 su 48 core, tetto ~35): 8 job in piu' portano a ~28, dentro il tetto. Spalmarli su
# tre schede tiene basso anche il contention sui SM.
#
# 🔴 RICHIEDE CHE `g4_chain_086.sh` SIA MORTO. La sua lista dopo `v1` e' proprio `enc`, e
# `launch.sh` salta una cella solo se ne trova il report.json AL MOMENTO DEL DISPATCH: una cella
# in corso non protegge, i due partirebbero sulle stesse cartelle. Si uccide **solo** il pid di
# `g4_chain_086.sh` (non il pgid): il `launch.sh` di `v1` gia' in volo e i suoi 3 job restano
# vivi e finiscono — e' lo stesso intervento fatto alle 19:53 su `g4_chain.sh`, che ha funzionato.
#
# Uccidere la catena non orfanizza niente: le sue tappe successive sono gia' coperte.
# Verificato prima di agire: `ucr086_cb128` 4/4 completa · `ucr086_proto_count` in corso su
# `g086tail` · `ucr086_cb128base` in corso su `cb128base_g4`.
set -u
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated

export SLOTS_PER_GPU=3           # 3 schede x 3 = 9 slot per 8 celle: una sola onda
bash scripts/launch.sh --cohort ucr086 --tag ucr086_enc \
  --arms federated_enc_fedavg,federated_enc_fedprox,federated_enc_fedproto,federated_enc_commoninit
echo "=== [$(date '+%F %T')] ucr086_enc completo (8 celle) ==="
