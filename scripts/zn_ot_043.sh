#!/usr/bin/env bash
# LA CELLA ORFANA DI `zn_ot`: ucr_043 / local, sovrallenata.
#
#   bash scripts/zn_ot_043.sh          # GPU3 di g4, 1 slot
#
# PERCHE' ESISTE. Il 2026-08-06 alle 11:12 questa cella era partita su **GPU0**, che dalle
# 13:xx dello stesso giorno e' RISERVATA (l'utente cede 0 e 5). L'ho fermata a 2 minuti di
# vita — nessun out-json scritto, checkpoint parziale rimosso — e da allora era rimasta
# scoperta: `zn_budget.sh ot` aveva gia' dispacciato le sue 3 celle (001/011/014 su GPU4) e
# `zn_ot_fill.sh` le sue 6 (082/083/086/170/222/229 su GPU1/2). Nessuna delle due catene la
# ripesca, perche' entrambe hanno la loro lista di serie CABLATA e un processo gia' avviato
# non rilegge il manifesto. Il proprietario nel manifesto e' `zn_budget.sh ot`: qui non si
# riassegna niente, si esegue la cella che quella catena non puo' piu' raggiungere.
#
# ⚠️ LE VARIABILI SONO LA CONDIZIONE SPERIMENTALE, NON UN DETTAGLIO. `zn_ot` e' la variante
# SOVRALLENATA: tetto alzato **e** early stopping spento. Spegnere `EARLY_STOPPING` da solo
# NON basta — `_converged_loop` ripristina comunque `best_state` a fine giro, quindi serve
# anche `KEEP_LAST_WEIGHTS=1`, o si ottiene una cella `es` con un tag `ot`. I quattro valori
# qui sotto devono restare identici a quelli di `zn_budget.sh ot`.
#
# ⛔ GPU0 e GPU5 sono riservate: il guardiano sotto rifiuta con rc=4 se qualcuno cambia
# LAUNCH_GPUS per distrazione. GPU3 e' la scelta perche' e' l'unica scheda a 1 job.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10

export LAUNCH_GPUS="${LAUNCH_GPUS:-3}"
export SLOTS_PER_GPU="${SLOTS_PER_GPU:-1}"
export S1_MAX_STEPS=25000
export S2_MAX_STEPS=100000
export EARLY_STOPPING=0
export KEEP_LAST_WEIGHTS=1
export NO_MPS=1
export FEDVQ_AMP=fp16

say() { echo "[$(date -u '+%F %T') UTC | $(TZ=Europe/Madrid date '+%H:%M') Madrid][ot/043] $*"; }

bash "$REPO/scripts/_g4_gpu.sh" "$LAUNCH_GPUS" || exit 4
"$PY" scripts/zn_owner.py --check >/dev/null || { say "!! partizione ROTTA — non parto"; exit 2; }

say "ucr_043 local · GPU $LAUNCH_GPUS · tetto s1=$S1_MAX_STEPS s2=$S2_MAX_STEPS · ES=off keep_last=on"
LAUNCH_ONLY_CLUSTERS=ucr_043 bash scripts/launch.sh --cohort ucr2p_10 --tag zn_ot \
  --arms local --extra "--window-normalization zscore"
say "=== finito (rc=$?) ==="
