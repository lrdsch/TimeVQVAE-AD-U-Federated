#!/usr/bin/env bash
# COPRIFUOCO sulla terza scheda di g4.
#
# La regola concordata e': 3 schede su g4 di notte, 2 dalle 09:00 di Madrid in poi. g4 e' un
# server di dipartimento con 20+ utenti, quindi il limite si rispetta all'ora, non "quando
# capita".
#
# `pool_worker.sh` ha gia' una guardia oraria, ma controlla solo PRIMA di prendere un job
# nuovo: non interrompe quello in corso. Con celle `fidelity_*` da ~2,5 h, un job preso alle
# 06:55 UTC terrebbe la terza scheda fino alle 11:30 di Madrid. Questo script chiude quel buco.
#
# Alle STOP UTC: uccide il processo sulla GPU 5, chiude il worker, e RILASCIA il claim del job
# interrotto, cosi' un altro worker lo riprende da capo invece di perderlo dalla coda.
set -u

REPO=/home/leonardo/PhD/TimeVQVAE-AD-U-Federated
JOBS=$REPO/logs/probe/wave_night.jobs
STOP=${1:-0700}          # 0700 UTC = 09:00 Madrid
GPU=${2:-5}
WORKER=${3:-w5}

say() { echo "[$(date -u '+%F %T') UTC][coprifuoco] $*"; }
say "attendo le $STOP UTC per liberare la GPU $GPU di g4"

while [ "$(date -u +%H%M)" -lt "$STOP" ]; do sleep 120; done

say "e' l'ora. Libero la GPU $GPU."

uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$GPU" '$1==g{print $2}')
pids=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
       | awk -F', ' -v u="$uuid" '$1==u{print $2}')

for p in $pids; do
  cmd=$(ps -o cmd= -p "$p" 2>/dev/null)
  ser=$(sed -n 's/.*--series \([0-9]*\).*/\1/p' <<<"$cmd")
  var=$(sed -n 's/.*--variant \([A-Za-z0-9_]*\).*/\1/p' <<<"$cmd")
  if [ -n "${ser:-}" ] && [ -n "${var:-}" ]; then
    # riga del jobfile che contiene QUESTA combinazione -> numero del claim da rilasciare
    n=$(grep -n -- "--series $ser .*--variant $var\$" "$JOBS" | cut -d: -f1 | head -1)
    if [ -n "${n:-}" ]; then
      rm -rf "$JOBS.claims/$n"
      say "claim $n rilasciato (serie $ser, variante $var): un altro worker lo riprendera'"
    fi
    # la run parziale va rimossa, altrimenti sembra una cella valida
    rm -rf "$REPO/artifacts/runs/ucr${ser}_abl_${var}"
    say "rimossa la run parziale ucr${ser}_abl_${var}"
  fi
  kill "$p" 2>/dev/null && say "ucciso il pid $p"
done

sleep 5
screen -S "$WORKER" -X quit 2>/dev/null && say "worker $WORKER chiuso"
say "GPU $GPU libera. Da ora g4 gira su 2 schede."
