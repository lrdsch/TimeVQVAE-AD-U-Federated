#!/usr/bin/env bash
# Emette UNA riga di stato ogni PERIOD secondi, su stdout.
#
# PERCHE' NON UN CRON. Il cron della sessione scatta solo quando la REPL e' ferma, e in una
# sessione dove si conversa non e' mai ferma: armato alle 17:30 del 2026-08-01 e ricreato tre
# volte, ha sparato **zero** volte in dieci ore. Ogni riga che questo script scrive su stdout
# diventa invece una notifica, che arriva a prescindere da cosa sto facendo.
#
# Una riga sola per tick, densa: la notifica stessa deve bastare a capire se qualcosa e' rotto,
# senza dover andare a guardare. Se emettesse molte righe il monitor verrebbe silenziato.
set -u
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
PERIOD=${PERIOD:-900}
G4=leonardo@g4.etsisi.upm.es

# ⚠ CORRETTO 2026-08-02. Contava una cella come fatta appena esisteva UN report.json, cioe'
# appena il primo dei 5 client aveva finito: il conteggio anticipava la realta' di 1-3 celle
# e riportavo avanzamenti che non c'erano ancora. Una cella e' completa a 5 client, punto.
celle() { find "artifacts/runs/$1" -name report.json 2>/dev/null \
          | xargs -r -n1 dirname | xargs -r -n1 dirname \
          | sort | uniq -c | awk '$1 >= 5' | wc -l; }

prev=-1
while :; do
  tot=0
  for s in 001 011 222 229 014 043 083 086 170; do
    # cb128base = `local` e `centralized` a K=128, aggiunti il 2026-08-02: senza di loro il
    # blocco K=128 non aveva ne baseline ne tetto, e "la federazione batte local a K=128?"
    # era una domanda senza risposta. 30 celle per serie diventano 34, il totale 270 -> 306.
    for t in v1 enc cb128 proto_count cb128base; do tot=$(( tot + $(celle "ucr${s}_${t}") )); done
  done

  # job PADRE su g2: i worker del dataloader ereditano la riga di comando, quindi un pgrep
  # nudo conta 400 processi invece di 12. Si filtra chi ha il padre fuori dall'insieme.
  ps -eo pid,ppid,args | grep '[f]ederated_eval' | awk '{print $1}' > /tmp/_st_g2.txt
  g2j=$(ps -eo pid,ppid,args | grep '[f]ederated_eval' \
        | while read -r p pp _; do grep -qx "$pp" /tmp/_st_g2.txt || echo x; done | wc -l)
  g2load=$(uptime | sed -E 's/.*load average: ([0-9]+[.,][0-9]+).*/\1/')

  # ⚠ CORRETTO 2026-08-02 11:46. La versione precedente faceva UN `ps` PER OGNI PID per leggere
  # il ppid: con ~500 worker su una macchina a load 22 e 776 utenti sono 500 fork, e andava in
  # timeout da sola. Il tick delle 11:44 ha riportato "g4 ? job" facendomi credere che il ponte
  # fosse caduto, mentre g4 stava benissimo. Una sonda che mente e' peggio di nessuna sonda.
  # Ora: un solo `ps`, e il filtro sui padri lo fa awk in memoria.
  g4=$(timeout 45 ssh -o BatchMode=yes -o ConnectTimeout=10 "$G4" \
        'ps -eo pid,ppid,args -u leonardo 2>/dev/null | awk "/[f]ederated_ev/ {pid[\$1]=1; pp[\$1]=\$2}
             END {n=0; for (p in pid) if (!(pp[p] in pid)) n++; printf \"%d \", n}"
         uptime | sed -E "s/.*load average: ([0-9]+[.,][0-9]+).*/\1/"' 2>/dev/null) \
     || g4="? ?"
  g4j=${g4%% *}; g4load=${g4##* }

  # Collisione VERA = due processi con la STESSA out-dir, dataset, cluster E arm. Il rischio
  # esiste perche' piu' flussi scrivono lo stesso artifacts/runs e launch.sh salta una cella
  # solo se ne trova il report.json AL DISPATCH: una cella in corso non protegge.
  # ⚠ La chiave DEVE includere out-dir: confrontando solo dataset+cluster+arm ho segnalato due
  # falsi allarmi (`ucr086_v1` a K=64 e `ucr086_cb128` a K=128 sono lo stesso arm sulla stessa
  # serie, ma sono l'ablazione, e scrivono in cartelle diverse).
  coll=$(timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=10 "$G4" \
          'ps -eo pid,ppid,args -u leonardo 2>/dev/null | awk "
             /[f]ederated_ev/ {pid[\$1]=1; pp[\$1]=\$2; L[\$1]=\$0}
             END {for (p in pid) if (!(pp[p] in pid)) print L[p]}"' 2>/dev/null \
        | grep -oE '\-\-out-dir [^ ]+ --dataset [a-z0-9_]+ --cluster [a-z0-9_]+ --arms? [a-z0-9_,.]+' \
        | sort | uniq -d | wc -l)

  wd=$(tail -1 logs/bridge_watchdog.log 2>/dev/null | grep -oE '\-> .*' | cut -c4-)
  delta=$(( prev < 0 ? 0 : tot - prev )); prev=$tot

  alert=""
  [ "${g4j:-0}" = "0" ] && alert=" ⚠ NESSUN JOB SU g4"
  [ "${g2j:-0}" = "0" ] && alert="$alert ⚠ NESSUN JOB SU g2"
  [ "$wd" != "ok" ] && alert="$alert ⚠ PONTE: $wd"
  [ "${coll:-0}" -gt 0 ] 2>/dev/null && alert="$alert 🔴 COLLISIONE: $coll celle doppie"

  printf 'TICK %s | celle %s/306 (+%s) | g2 %s job load %s | g4 %s job load %s | ponte %s%s\n' \
    "$(date '+%H:%M')" "$tot" "$delta" "$g2j" "$g2load" "$g4j" "$g4load" "${wd:-?}" "$alert"
  sleep "$PERIOD"
done
