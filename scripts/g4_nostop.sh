#!/usr/bin/env bash
# `nostop` — il criterio di arresto di upstream, su ucr_001. Terzo slot sulla GPU 2.
# Wrapper senza argomenti (vedi scripts/g4_wave1.sh sul perche').
set -u
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
bash scripts/wave_runner.sh logs/probe/wave_nostop.jobs "1" 1 logs/probe
