#!/usr/bin/env bash
# GUARDIANO delle GPU restituite: tiene 0/4/5 LIBERE, comunque ci finisca del lavoro sopra.
#
# ⚠ PERCHE' ESISTE, cioe' come ha fallito il coprifuoco il 2026-08-05.
# Alle 09:01 `zn_ada_curfew.sh` aveva liberato le tre Ada davvero (log: 0, 4, 5 a 1 MiB).
# Alle 09:10 il verificatore indipendente le ha trovate TUTTE E SEI occupate, e lo sono
# rimaste fino alle 12:39. Non c'entrava `launch.sh`: era una catena di `latent_probe.py`
# dentro uno screen, pinnata sulle Ada con CUDA_VISIBLE_DEVICES.
#
# L'errore di progetto e' che i due esecutori del coprifuoco modellavano i PRODUTTORI noti
# (la catena `zn_g4_ada.sh` e il suo `launch.sh`) invece della PROPRIETA' richiesta. Un
# produttore che non era nella lista ha rimesso il lavoro sopra nove minuti dopo, e per tre
# ore e mezza nessuno se n'e' accorto: il coprifuoco aveva gia' scritto "OK" ed era uscito.
#
# Secondo errore, nello stesso script: l'uscita anticipata dal presidio. `stabile e libero`
# dopo 3 giri = 70 secondi. Una finestra di 70 secondi non prova niente su una macchina dove
# i job partono ogni pochi minuti.
#
# Qui la regola e' una sola e non nomina nessun produttore: **su queste schede non ci deve
# stare NIENTE di nostro**, e il controllo continua fino a fine giornata.
#
# ⚠ NON tocca i processi di altri utenti: g4 e' un host di dipartimento con 20+ persone, e
# le loro schede sono affar loro. Il filtro sull'utente non e' negoziabile.
# ⚠ Si disattiva creando `evidence/.ada_guard_off` -- serve un modo di spegnerlo che non
# richieda di trovarne il pid, se un giorno quelle schede tornano nostre.
#
#   bash scripts/zn_ada_guard.sh [gpu,csv] [HHMM UTC di fine]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

GPUS=${1:-0,4,5}
UNTIL=${2:-2200}                  # 22:00 UTC = mezzanotte di Madrid
ME="$(id -un)"
OFF="$REPO/evidence/.ada_guard_off"
LOG="$REPO/evidence/ada_guard_$(date -u +%Y%m%d).log"
mkdir -p "$REPO/evidence"
say() { echo "[$(TZ=Europe/Madrid date '+%F %H:%M:%S') Madrid][guardia] $*" | tee -a "$LOG"; }

_h=${UNTIL:0:2}; _m=${UNTIL:2:2}
END=$(date -u -d "today ${_h}:${_m}" +%s)
[ "$END" -le "$(date -u +%s)" ] && END=$(date -u -d "tomorrow ${_h}:${_m}" +%s)

say "=== GUARDIA ATTIVA su GPU $GPUS fino alle $(TZ=Europe/Madrid date -d "@$END" '+%H:%M') Madrid ==="
say "controllo ogni 60 s · uccido SOLO processi di $ME · spegnimento: touch $OFF"

clean=0
while [ "$(date -u +%s)" -lt "$END" ]; do
  if [ -f "$OFF" ]; then say "trovato $OFF -- guardia disattivata, esco."; exit 0; fi
  hit=0
  for g in ${GPUS//,/ }; do
    uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v x="$g" '$1==x{print $2}')
    [ -z "$uuid" ] && continue
    for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
                 | awk -F', ' -v u="$uuid" '$1==u{print $2}'); do
      [ "$(ps -o user= -p "$pid" 2>/dev/null | xargs)" = "$ME" ] || continue
      # CHI era, prima di ucciderlo: senza questo si sa solo che qualcosa e' tornato, non
      # cosa -- ed e' esattamente l'informazione che mancava stamattina.
      args=$(ps -o args= -p "$pid" 2>/dev/null | cut -c1-140)
      ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
      pargs=$(ps -o args= -p "${ppid:-1}" 2>/dev/null | cut -c1-100)
      say "!! GPU$g occupata da noi (pid $pid) -- la libero"
      say "     job:   $args"
      say "     padre: [$ppid] $pargs"
      kill "$pid" 2>/dev/null && hit=$((hit+1))
    done
  done
  if [ "$hit" -gt 0 ]; then
    clean=0
  else
    clean=$((clean+1))
    [ $((clean % 60)) -eq 0 ] && say "ok: $((clean)) controlli consecutivi con GPU $GPUS libere"
  fi
  sleep 60
done
say "=== fine turno di guardia ==="
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | sed 's/^/  /'
