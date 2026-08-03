#!/usr/bin/env bash
# La normalizzazione per finestra — la differenza trovata il 2026-08-03 leggendo il loro codice.
# Tre celle, una per scheda, tutte sulle RTX 4500 Ada di g4 (0/4/5): stessa architettura, cosi'
# il confondente hardware non entra nel contrasto. Wrapper senza argomenti (vedi g4_fidelity.sh).
set -u
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
cat logs/probe/wave_znorm.jobs logs/probe/wave_fidznorm.jobs logs/probe/wave_znorm043.jobs > logs/probe/wave_znorm_all.jobs
bash scripts/wave_runner.sh logs/probe/wave_znorm_all.jobs "0 4 5" 1 logs/probe
