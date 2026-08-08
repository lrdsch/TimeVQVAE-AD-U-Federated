#!/usr/bin/env bash
# SECONDA CATENA per `zn_ot` — la META' DI CODA, sulle 3090 rimaste libere.
#
#   bash scripts/zn_ot_fill.sh          # GPU 1,2,3 · 3 slot ciascuna
#
# PERCHE'. Alle 11:53 `zn_ot` aveva 20 celle da fare e girava su TRE slot di GPU4, mentre
# GPU0/1/2/3 erano a ZERO job (1-3 MiB) e la load di g4 era 6,9 su 48 core: la campagna era
# drenata. A 3 slot le 20 celle sono ~37 ore, a 12 sono ~9.
#
# QUALI SERIE, E PERCHE' PROPRIO QUELLE. La catena viva (`zn_budget.sh ot`, GPU4) ha in lista
# tutte e 10 le serie e le dispaccia in ORDINE DI COORTE — l'ordine della sua variabile non
# conta, `launch.sh` lo ignora (verificato: le ho messe in ordine di importanza e ha comunque
# iniziato da 001). Quindi:
#
#   catena viva  001 · 011 · 014 · 043 · 082 · 083 · 086 · 170 · 222 · 229
#                └── sta qui ──┘                     └──── ci arriva fra ore ────┘
#   questa                                  082 · 083 · 086 · 170 · 222 · 229
#
# Si prende dalla CODA, non dalla testa: quando la prima ci arriva, questa le ha gia' chiuse e
# `launch.sh` le salta (l'out-json e' su disco). Prendere dalla testa sarebbe una collisione a
# minuti di distanza invece che a ore.
#
# ⚠️ NON EDITARE `zn_budget.sh` PER FARGLI CAMBIARE LISTA. E' in esecuzione, e bash rilegge lo
# script dall'offset di byte salvato: spostare le righe non ancora lette gli fa eseguire
# spazzatura da li' in poi. Per questo qui c'e' un file nuovo invece di un parametro in piu'.
#
# ⚠️ I DUE FLAG CHE DEFINISCONO `ot`. `EARLY_STOPPING=0` da solo NON basta: `_converged_loop`
# ripristina comunque `best_state` a fine corsa, quindi produrrebbe lo stesso modello di
# `zn_es`, ore dopo, senza che nulla lo segnali. Serve `KEEP_LAST_WEIGHTS=1` insieme.
# Si verifica nel log del JOB, non nel banner della catena:
#     grep '^\[config\]' logs/runs/zn_ot/<...>.log
#
# ⚠️ Solo le 3090 (1,2,3), non le Ada: tiene omogenea la classe di scheda dentro il contrasto.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10

export S1_MAX_STEPS="${S1_MAX_STEPS:-25000}"
export S2_MAX_STEPS="${S2_MAX_STEPS:-100000}"
export EARLY_STOPPING=0
export KEEP_LAST_WEIGHTS=1
export LAUNCH_GPUS="${LAUNCH_GPUS:-1 2 3}"
export SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"
export NO_MPS=1                       # g4 e' condivisa con 20+ utenti: MPS non si tocca mai
export FEDVQ_AMP=fp16
export LAUNCH_ONLY_CLUSTERS="${LAUNCH_ONLY_CLUSTERS:-ucr_082,ucr_083,ucr_086,ucr_170,ucr_222,ucr_229}"
ZN='--window-normalization zscore'
say() { echo "[$(date -u '+%F %T') UTC | $(TZ=Europe/Madrid date '+%H:%M') Madrid][ot-fill] $*"; }

# Nessuna scheda con processi di ALTRI utenti — regola dura su g4.
for g in $LAUNCH_GPUS; do
  uu=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$g" '$1==g{print $2}')
  for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
             | awk -F', ' -v u="$uu" '$1==u{print $2}'); do
    o=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
    [ -n "$o" ] && [ "$o" != "$(whoami)" ] && { say "!! GPU$g ha processi di $o — NON parto"; exit 3; }
  done
done

say "tetto s1=$S1_MAX_STEPS s2=$S2_MAX_STEPS · GPU [$LAUNCH_GPUS] × $SLOTS_PER_GPU slot"
say "serie (meta' di coda): $LAUNCH_ONLY_CLUSTERS"

say "=== local ==="
bash scripts/launch.sh --cohort ucr2p_10 --tag zn_ot --arms local --extra "$ZN"
say "=== centralized ==="
bash scripts/launch.sh --cohort ucr2p_10 --tag zn_ot --arms centralized --extra "$ZN"
say "=== ot-fill finito (rc=$?) ==="
