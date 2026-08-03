#!/usr/bin/env bash
# Dispatcher generico per le SONDE (fedscaler, ablazione): una riga del jobfile = una cella.
#
# PERCHE' NON `launch.sh`. Il launcher della campagna sa dispacciare solo celle di coorte, con
# la configurazione della coorte. Le sonde sono per definizione fuori coorte — cambiano un campo
# del Config o lo scaler — quindi servono un dispatcher che non sappia niente di coorti e si
# limiti a tenere N job vivi su M schede.
#
#   bash scripts/wave_runner.sh <jobfile> "<gpu ...>" <slot_per_gpu> <dir_log>
#
# Formato del jobfile: una riga per cella, `<etichetta><TAB><comando>`. Righe vuote e `#` saltate.
# L'etichetta diventa il nome del log, quindi dev'essere unica.
#
# ⚠️ NON usa MPS: su un host condiviso instraderebbe i contesti CUDA degli altri utenti
# attraverso il nostro server (vedi l'intestazione di `scripts/run_on_g4.sh`).
#
# ⚠️ Assegnazione round-robin per indice, non "la scheda piu' scarica". Va bene perche' le celle
# hanno costo simile; se un giorno il jobfile mescolasse celle molto diverse, andrebbe cambiata.
set -u

JOBS=${1:?serve il jobfile}
GPUS=${2:-"2 3"}
SLOTS=${3:-4}
LOGDIR=${4:-logs/probe}

cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
mkdir -p "$LOGDIR"

PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
read -r -a GARR <<< "$GPUS"
NG=${#GARR[@]}
TOT=$(( NG * SLOTS ))

echo "[$(date '+%F %T')] wave: schede=($GPUS) slot/scheda=$SLOTS -> $TOT paralleli"
echo "[$(date '+%F %T')] jobfile=$JOBS log=$LOGDIR"

i=0
while IFS=$'\t' read -r etichetta comando; do
  [ -z "${etichetta:-}" ] && continue
  case "$etichetta" in \#*) continue ;; esac

  # Attende uno slot libero PRIMA di lanciare, cosi' non si supera mai $TOT.
  while [ "$(jobs -rp | wc -l)" -ge "$TOT" ]; do sleep 15; done

  g=${GARR[$(( i % NG ))]}
  log="$LOGDIR/${etichetta}.log"
  echo "[$(date '+%F %T')] -> GPU$g  $etichetta"
  (
    export CUDA_VISIBLE_DEVICES="$g"
    export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
    export PY
    eval "$comando"
  ) > "$log" 2>&1 &
  i=$(( i + 1 ))
done < "$JOBS"

wait
echo "[$(date '+%F %T')] wave completa: $i celle"
