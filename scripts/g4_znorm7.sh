#!/usr/bin/env bash
# Generalizzazione della z-norm: le 7 serie mancanti delle nostre 9, arm centralized.
# Confronto APPAIATO — il riferimento di campagna esiste per tutte, e gli autori hanno
# rilasciato i punteggi per timestep, quindi ogni serie avra' anche la loro AUPRC
# ricalcolata con la nostra stessa metrica.
# Sei schede: 0/4/5 sono Ada, 1/2/3 sono 3090. Architetture MISTE, ma qui e' innocuo:
# ogni serie e' una cella a se' e il confronto e' contro il proprio riferimento, non
# fra serie. Wrapper senza argomenti (vedi g4_fidelity.sh).
set -u
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
bash scripts/wave_runner.sh logs/probe/wave_znorm7.jobs "0 1 2 3 4 5" 1 logs/probe
