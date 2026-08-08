#!/usr/bin/env bash
# VERIFICA INDIPENDENTE del vincolo: dalle 09:00 di Madrid devono esserci 3 GPU libere su g4.
#
# Non si fida di quello che dice il coprifuoco. Aspetta le 09:10 (dieci minuti di margine
# perche' un processo ucciso puo' metterci qualche secondo a rilasciare il contesto CUDA),
# poi CAMPIONA `nvidia-smi` per un minuto intero, ogni 5 secondi.
#
# Un solo campione non basta: un job che sta partendo occupa la scheda per pochi secondi e poi
# fallisce, e un job che sta morendo tiene la memoria allocata finche' il driver non la
# rilascia. Una scheda si dichiara libera solo se e' risultata libera in TUTTI i campioni.
#
# Scrive l'esito in un file, cosi' resta agli atti anche se nessuno guarda.
#
#   bash scripts/zn_verify_g4_free.sh [HH:MM Madrid] [minuti] [n_gpu_richieste]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

AT=${1:-09:10}                 # ora di MADRID
MINUTI=${2:-1}
NEED=${3:-3}                   # quante schede devono risultare libere
G4=${G4:-leonardo@g4.etsisi.upm.es}
OUT="$REPO/evidence/g4_free_check_$(date -u +%Y%m%d).txt"
SOGLIA_MIB=200                 # sotto questa soglia la scheda e' "libera" (il driver tiene ~1-20 MiB)

mkdir -p "$REPO/evidence"
say() { echo "[$(TZ=Europe/Madrid date '+%F %H:%M:%S') Madrid] $*" | tee -a "$OUT"; }

# Istante assoluto in ora di Madrid, non confronto HHMM (vedi zn_ada_curfew.sh).
TARGET=$(TZ=Europe/Madrid date -d "today $AT" +%s)
[ "$TARGET" -le "$(date +%s)" ] && TARGET=$(TZ=Europe/Madrid date -d "tomorrow $AT" +%s)

say "=== VERIFICA GPU LIBERE SU g4 ==="
say "attendo le $AT Madrid ($(( (TARGET - $(date +%s)) / 60 )) min), poi campiono per $MINUTI min"
while [ "$(date +%s)" -lt "$TARGET" ]; do sleep 30; done

say "inizio campionamento (ogni 5 s per $MINUTI min, soglia ${SOGLIA_MIB} MiB)"
FINE=$(( $(date +%s) + MINUTI * 60 ))
N=0
declare -A OCCUPATA          # gpu -> quante volte e' risultata occupata
declare -A MAXMEM
GPULIST=""
while [ "$(date +%s)" -lt "$FINE" ]; do
  snap=$(timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$G4" \
         'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits' 2>/dev/null)
  [ -z "$snap" ] && { say "  campione saltato: g4 non raggiungibile"; sleep 5; continue; }
  N=$((N+1))
  while IFS=', ' read -r g mem util; do
    g=$(echo "$g" | xargs); mem=$(echo "$mem" | xargs)
    [[ "$GPULIST" == *" $g "* ]] || GPULIST="$GPULIST $g "
    [ "${MAXMEM[$g]:-0}" -lt "$mem" ] && MAXMEM[$g]=$mem
    [ "$mem" -gt "$SOGLIA_MIB" ] && OCCUPATA[$g]=$(( ${OCCUPATA[$g]:-0} + 1 ))
  done <<< "$snap"
  sleep 5
done

say ""
say "campioni raccolti: $N"
LIBERE=""; OCCUPATE=""
for g in $GPULIST; do
  occ=${OCCUPATA[$g]:-0}
  if [ "$occ" -eq 0 ]; then
    LIBERE="$LIBERE $g"; say "  GPU$g  LIBERA        (max ${MAXMEM[$g]:-0} MiB su $N campioni)"
  else
    OCCUPATE="$OCCUPATE $g"; say "  GPU$g  occupata      (in $occ/$N campioni, max ${MAXMEM[$g]:-0} MiB)"
  fi
done

NL=$(echo $LIBERE | wc -w)
say ""
if [ "$NL" -ge "$NEED" ]; then
  say "ESITO: OK — $NL GPU libere (${LIBERE# }), ne servivano $NEED"
  rc=0
else
  say "ESITO: !! VIOLATO — solo $NL GPU libere (${LIBERE:-nessuna}), ne servivano $NEED"
  say "       occupate:${OCCUPATE}"
  say "       processi nostri ancora sulle schede:"
  timeout 20 ssh -o BatchMode=yes "$G4" \
    'nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | while IFS=, read u p m; do
       g=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F", " -v x="$(echo $u|xargs)" "\$2==x{print \$1}")
       echo "         GPU$g pid$p $m $(ps -o user= -p $(echo $p|xargs) 2>/dev/null)"
     done' 2>/dev/null | tee -a "$OUT"
  rc=1
fi
say "referto scritto in $OUT"
exit $rc
