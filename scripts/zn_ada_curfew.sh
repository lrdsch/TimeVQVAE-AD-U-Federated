#!/usr/bin/env bash
# COPRIFUOCO su g4 — garantisce 3 GPU LIBERE dalle 09:00 di Madrid.
#
# Il vincolo dato dall'utente e': "dalle 9 devono esserci 3 GPU libere su g4, non importa
# quali". Quindi il coprifuoco NON ragiona per serie o per catena: interroga `nvidia-smi`,
# prende i processi che stanno davvero sulle schede da liberare, e li uccide. E' l'unico modo
# di garantire la PROPRIETA' richiesta invece di una sua approssimazione.
#
# ⚠ La versione precedente uccideva per nome-serie (`--cluster ucr_222|229|170`). Fragile: se
# una cella fosse finita su una scheda diversa da quella prevista, o se una catena avesse
# dispatchato altrove, la GPU sarebbe rimasta occupata e il vincolo violato senza accorgersene.
#
# ⚠ ISTANTE ASSOLUTO, non confronto HHMM. `[ "$(date -u +%H%M)" -lt 0700 ]` confronta INTERI:
# lanciato alle 23:05 da' 2305 < 700 = FALSO e il coprifuoco scatta subito. Successo davvero
# il 2026-08-04 alle 23:05 UTC.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

STOP=${1:-0700}                    # 07:00 UTC = 09:00 Madrid
GPUS=${2:-0,4,5}                   # le schede da restituire
say() { echo "[$(date -u '+%F %T') UTC | $(TZ=Europe/Madrid date '+%H:%M') Madrid][coprifuoco] $*"; }

_h=${STOP:0:2}; _m=${STOP:2:2}
TARGET=$(date -u -d "today ${_h}:${_m}" +%s)
[ "$TARGET" -le "$(date -u +%s)" ] && TARGET=$(date -u -d "tomorrow ${_h}:${_m}" +%s)
say "attendo fino a $(TZ=Europe/Madrid date -d "@$TARGET" '+%H:%M') Madrid -- mancano $(( (TARGET - $(date -u +%s)) / 60 )) min"
while [ "$(date -u +%s)" -lt "$TARGET" ]; do sleep 60; done
say "e' l'ora. Libero le GPU $GPUS."

# 1) le CATENE che potrebbero ri-dispatchare su quelle schede.
#    Filtro sul TERZO campo (lo script), non sul secondo: `pgrep -f zn_g4_ada` prenderebbe
#    anche lo screen che lo ospita, e ucciderlo lascerebbe i job orfani ma vivi.
for p in $(ps -eo pid,args | awk '$2=="bash" && $3 ~ /zn_g4_ada\.sh/ {print $1}'); do
  kill "$p" 2>/dev/null && say "catena Ada fermata (pid $p)"
done
sleep 3

# 2) i PROCESSI SULLE SCHEDE, letti da nvidia-smi. Due passate: la seconda prende chi era
#    in avvio durante la prima.
for pass in 1 2; do
  n=0
  for g in ${GPUS//,/ }; do
    uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v x="$g" '$1==x{print $2}')
    [ -z "$uuid" ] && continue
    for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
                 | awk -F', ' -v u="$uuid" '$1==u{print $2}'); do
      # solo processi NOSTRI: mai toccare quelli di altri utenti sull'host condiviso
      [ "$(ps -o user= -p "$pid" 2>/dev/null | xargs)" = "$(id -un)" ] || continue
      kill "$pid" 2>/dev/null && n=$((n+1))
    done
  done
  say "passata $pass: uccisi $n processi sulle GPU $GPUS"
  sleep 10
done

# 3) i parziali su disco: solo celle SENZA out-json. Quelle finite non si toccano.
rm_n=0
for tag in zn_main zn_enc; do
  for ck in artifacts/runs/$tag/ckpt/ucr_split_w2p/*/seed0/*; do
    [ -d "$ck" ] || continue
    ser=$(basename "$(dirname "$(dirname "$ck")")"); arm=$(basename "$ck")
    [ -f "artifacts/runs/$tag/ucr_split_w2p/${ser}__${arm}.json" ] && continue
    # solo se NON c'e' piu' un processo vivo che ci sta lavorando
    ps -eo args | grep -q "[f]ederated_eval.py.*--cluster $ser --arms $arm\b" && continue
    rm -rf "$ck" && rm_n=$((rm_n+1))
  done
done
say "rimossi $rm_n alberi di checkpoint parziali (le celle le rifara' il raccoglitore)"

# 4) VERIFICA che il vincolo sia davvero rispettato, non che lo sembri.
sleep 5
busy=""
for g in ${GPUS//,/ }; do
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v x="$g" '$1==x{print $2}')
  [ "${used:-0}" -gt 200 ] && busy="$busy $g(${used}MiB)"
done
if [ -n "$busy" ]; then
  say "!! ATTENZIONE: ancora occupate ->$busy"
else
  say "OK: le GPU $GPUS sono libere."
fi
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'
