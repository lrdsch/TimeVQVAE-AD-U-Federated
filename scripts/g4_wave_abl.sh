#!/usr/bin/env bash
# Le 4 varianti di ablazione restanti su `ucr_001`, arm `centralized`.
# 2 slot per scheda su GPU 2 e 3: quattro celle, una per slot, contesa minima.
# Wrapper senza argomenti — vedi l'intestazione di scripts/g4_wave1.sh sul perche'.
set -u
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
bash scripts/wave_runner.sh logs/probe/wave_g4b.jobs "2 3" 2 logs/probe
