#!/usr/bin/env bash
# Worker di una CODA CONDIVISA fra g2 e g4. Ogni worker tiene UNA scheda e tira il job
# successivo dalla coda finche' ce ne sono.
#
#   bash scripts/pool_worker.sh <jobfile> <gpu> <logdir> [<stop_utc_HHMM>]
#
# PERCHE' NON `wave_runner.sh`. Quello assegna i job round-robin per indice, a slot fissi:
# va bene quando le celle costano uguale. Qui no — `ucr_083` ha finestra 970 e 157 000 punti
# di test, `ucr_229` ne ha 254 e 42 000, e le celle `fidelity_*` girano i tetti pieni senza
# early stopping (2-3x una cella normale). Con l'assegnazione statica una scheda finirebbe
# ore prima delle altre. Qui chi si libera prende il prossimo.
#
# CLAIM ATOMICO: `mkdir` fallisce se la directory esiste, ed e' atomico anche attraverso il
# mount sshfs (mappa su SFTP MKDIR). Un file di lock con flock NON sarebbe affidabile li'.
#
# GUARDIA ORARIA: `stop_utc_HHMM` (es. 0700 = 09:00 a Madrid) fa uscire il worker PRIMA di
# prendere un nuovo job. Il job in corso NON viene interrotto — si perderebbe un'ora di
# calcolo per rispettare un limite al minuto. Serve a scalare da 3 schede a 2 su g4 la
# mattina senza intervento manuale.
set -u

JOBS=${1:?serve il jobfile}
GPU=${2:?serve lindice della GPU}
LOGDIR=${3:-logs/probe}
STOP=${4:-}

cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
mkdir -p "$LOGDIR"
CLAIMS="${JOBS}.claims"
mkdir -p "$CLAIMS"

PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
HOST=$(hostname)
say() { echo "[$(date -u '+%F %T') UTC][$HOST gpu$GPU] $*"; }

say "worker avviato su $JOBS${STOP:+ (si ferma alle $STOP UTC)}"

n=0
while IFS=$'\t' read -r etichetta comando; do
  n=$((n + 1))
  [ -z "${etichetta:-}" ] && continue
  case "$etichetta" in \#*) continue ;; esac

  if [ -n "$STOP" ] && [ "$(date -u +%H%M)" -ge "$STOP" ]; then
    say "guardia oraria: sono le $(date -u +%H%M) UTC, non prendo altri job. Esco."
    exit 0
  fi

  # claim atomico: se la mkdir riesce il job e' mio, altrimenti l'ha gia' preso un altro
  mkdir "$CLAIMS/$n" 2>/dev/null || continue

  say "PRENDO $etichetta"
  t0=$(date +%s)
  # stdin da /dev/null: senza, il job EREDITA il jobfile come stdin e puo' consumarne
  # righe, facendo saltare job al `read` del ciclo. Bug classico dei loop `while read`.
  CUDA_VISIBLE_DEVICES="$GPU" NO_MPS=1 PY="$PY" \
    bash -c "$comando" > "$LOGDIR/$etichetta.log" 2>&1 < /dev/null
  rc=$?
  say "FINITO $etichetta rc=$rc in $((($(date +%s) - t0) / 60)) min"
done < "$JOBS"

say "coda esaurita, esco."
