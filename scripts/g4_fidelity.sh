#!/usr/bin/env bash
# `fidelity` — TUTTE le differenze con upstream insieme, su ucr_001, arm centralized.
# Una sola cella, una sola scheda (la 2 di g4, una 3090 come quella di `nostop`).
#
# Wrapper SENZA argomenti: il comando remoto di run_on_g4.sh e' gia' dentro `bash -lc '...'`,
# quindi qualunque lista quotata passata da fuori (es. "2 3") si spezza, gli argomenti slittano
# di posto e il dispatcher si duplica. E' successo il 2026-08-03 alle 17:35, 60 processi sulla
# stessa scheda. Vedi scripts/g4_wave1.sh.
#
# ⚠️ E' la corsa piu' lunga della famiglia, come `nostop`: senza early stopping gira i tetti
# interi, 10 000 + 50 000 step contro i ~22 000 di una cella normale.
set -u
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
bash scripts/wave_runner.sh logs/probe/wave_fidelity.jobs "2" 1 logs/probe
