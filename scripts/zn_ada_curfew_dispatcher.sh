#!/usr/bin/env bash
# COPRIFUOCO, SECONDA META': il DISPATCHER, non i job.
#
# ⚠ IL BUCO CHE CHIUDE. `zn_ada_curfew.sh` uccide (1) la catena `zn_g4_ada.sh` e (2) i
# processi che stanno sulle GPU 0/4/5. Ma fra i due c'e' un terzo processo che nessuno dei
# due filtri prende: il `bash scripts/launch.sh` che la catena ha generato come FIGLIO.
# Quello e' il dispatcher vero -- ha il pool di slot, e quando un job muore ne fa partire un
# altro sulla stessa scheda. Uccidere la catena non lo tocca (la catena lo aspetta e basta),
# e uccidere i job lo fa solo LAVORARE: vede tre slot liberi e li riempie. Il risultato
# sarebbe stato che alle 09:10 le Ada risultano di nuovo occupate, con il coprifuoco che
# nel suo log dichiara di averle liberate.
#
# PERCHE' UN SECONDO PROCESSO E NON UNA PATCH. `zn_ada_curfew.sh` e' GIA' IN ESECUZIONE, e
# bash legge lo script a offset di byte mentre lo esegue: modificarlo ora sposterebbe le
# righe che quel processo non ha ancora letto. Due esecutori indipendenti e idempotenti
# valgono comunque piu' di uno: se uno dei due muore, il vincolo regge lo stesso.
#
# COME INDIVIDUA IL DISPATCHER. Non per tag: `zn_chain.sh g4` (3090) e `zn_g4_ada.sh` (Ada)
# lanciano ENTRAMBI `launch.sh --tag zn_main`, quindi il tag non li distingue e ucciderli
# per tag fermerebbe anche le 3090, che l'utente non ha chiesto di liberare. Li distingue
# la PARENTELA -- e, come rete di sicurezza, la RISALITA dai processi che stanno davvero
# sulle schede 0/4/5, che per costruzione possono discendere solo dal dispatcher Ada.
#
#   bash scripts/zn_ada_curfew_dispatcher.sh [HHMM UTC] [gpu,csv] [minuti di presidio]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

STOP=${1:-0700}                   # 07:00 UTC = 09:00 Madrid
GPUS=${2:-0,4,5}
GUARD_MIN=${3:-20}                # presidio: un dispatcher puo' morire lentamente
ME="$(id -un)"
say() { echo "[$(date -u '+%F %T') UTC | $(TZ=Europe/Madrid date '+%H:%M') Madrid][coprifuoco-disp] $*"; }

_h=${STOP:0:2}; _m=${STOP:2:2}
TARGET=$(date -u -d "today ${_h}:${_m}" +%s)
[ "$TARGET" -le "$(date -u +%s)" ] && TARGET=$(date -u -d "tomorrow ${_h}:${_m}" +%s)
say "attendo fino a $(TZ=Europe/Madrid date -d "@$TARGET" '+%H:%M') Madrid -- mancano $(( (TARGET - $(date -u +%s)) / 60 )) min"
while [ "$(date -u +%s)" -lt "$TARGET" ]; do sleep 60; done
say "e' l'ora. Fermo i DISPATCHER che possono riempire le GPU $GPUS."

# ── chi possiede questo pid, risalendo la catena dei padri ───────────────────────
ancestor_launch() {          # stampa il pid dell'antenato `bash scripts/launch.sh`, se c'e'
  local p=$1 n=0
  while [ "$p" -gt 1 ] && [ $n -lt 12 ]; do
    local args; args=$(ps -o args= -p "$p" 2>/dev/null)
    case "$args" in "bash scripts/launch.sh"*) echo "$p"; return;; esac
    p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' '); [ -z "$p" ] && return
    n=$((n+1))
  done
}

kill_dispatchers() {
  local killed=0 pid ppid pargs
  # (a) per PARENTELA: launch.sh il cui padre e' la catena Ada.
  while read -r pid ppid; do
    pargs=$(ps -o args= -p "$ppid" 2>/dev/null)
    case "$pargs" in *zn_g4_ada.sh*) kill "$pid" 2>/dev/null && { say "  dispatcher Ada fermato (pid $pid, figlio di $ppid)"; killed=$((killed+1)); };; esac
  done < <(ps -eo pid,ppid,args --no-headers | awk '$3=="bash" && $4 ~ /launch[.]sh/ {print $1, $2}')
  # (b) rete di sicurezza: risalita dai processi che stanno sulle schede da liberare.
  local g uuid jpid anc
  for g in ${GPUS//,/ }; do
    uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v x="$g" '$1==x{print $2}')
    [ -z "$uuid" ] && continue
    for jpid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
                  | awk -F', ' -v u="$uuid" '$1==u{print $2}'); do
      [ "$(ps -o user= -p "$jpid" 2>/dev/null | xargs)" = "$ME" ] || continue
      anc=$(ancestor_launch "$jpid")
      [ -n "$anc" ] && kill "$anc" 2>/dev/null && { say "  dispatcher fermato per risalita da GPU$g (pid $anc)"; killed=$((killed+1)); }
    done
  done
  echo "$killed"
}

# ── presidio: i dispatcher PRIMA, i job DOPO ─────────────────────────────────────
# In quest'ordine e non nell'altro: uccidendo prima i job, il dispatcher ancora vivo vede
# gli slot liberi e ne fa partire altri nei secondi che passano.
END=$(( $(date -u +%s) + GUARD_MIN * 60 ))
round=0
while [ "$(date -u +%s)" -lt "$END" ]; do
  round=$((round+1))
  nd=$(kill_dispatchers)
  nj=0
  for g in ${GPUS//,/ }; do
    uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v x="$g" '$1==x{print $2}')
    [ -z "$uuid" ] && continue
    for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
                 | awk -F', ' -v u="$uuid" '$1==u{print $2}'); do
      # solo processi NOSTRI: mai toccare quelli degli altri utenti dell'host condiviso
      [ "$(ps -o user= -p "$pid" 2>/dev/null | xargs)" = "$ME" ] || continue
      kill "$pid" 2>/dev/null && nj=$((nj+1))
    done
  done
  busy=""
  for g in ${GPUS//,/ }; do
    used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v x="$g" '$1==x{print $2}')
    [ "${used:-0}" -gt 200 ] && busy="$busy $g(${used}MiB)"
  done
  [ $((round % 5)) -eq 1 ] || [ "$nd" -gt 0 ] || [ "$nj" -gt 0 ] \
    && say "giro $round: dispatcher $nd, job $nj, ancora occupate ->${busy:- nessuna}"
  [ -z "$busy" ] && [ "$nd" -eq 0 ] && [ "$nj" -eq 0 ] && [ $round -ge 3 ] && { say "stabile e libero, esco dal presidio"; break; }
  sleep 20
done

say "presidio finito. Stato delle schede:"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'
