#!/usr/bin/env bash
# `fidelity_prior` — SOLO lo stack del prior di upstream (x-transformers + RMSNorm +
# post-emb-norm + pos-emb 1-D piatta + embed 64), su ucr_001, arm centralized.
#
# Isola l'unico dei due assi che l'ablazione del 2026-08-03 non ha potuto misurare, perche'
# non esisteva l'implementazione. Tiene `--batch 64` e ogni altro default nostro, quindi e'
# appaiata al riferimento della campagna (ucr001_v1) — a differenza di `fidelity`, che muove
# tutto insieme e non attribuisce niente.
#
# Su g2, GPU 1. La 0 resta libera per ssanchez.
# Wrapper senza argomenti: vedi scripts/g4_fidelity.sh.
set -u
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
bash scripts/wave_runner.sh logs/probe/wave_fidelity_prior.jobs "1" 1 logs/probe
