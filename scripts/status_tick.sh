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
# 300 s (era 900) dal 2026-08-02 19:30: la campagna e' in coda e gli eventi che contano
# — un flusso che finisce, uno che si svuota — capitano piu' fitti che a meta' corsa.
PERIOD=${PERIOD:-300}
G4=leonardo@g4.etsisi.upm.es
G2_SER="001 011 222 229"      # una serie, una macchina: queste stanno su g2
G4_SER="014 043 083 086 170"  # queste su g4
TAGS="v1 enc cb128 proto_count cb128base"   # 16+8+4+2+4 = 34 celle per serie

# ⚠ CORRETTO 2026-08-02. Contava una cella come fatta appena esisteva UN report.json, cioe'
# appena il primo dei 5 client aveva finito: il conteggio anticipava la realta' di 1-3 celle
# e riportavo avanzamenti che non c'erano ancora. Una cella e' completa a 5 client, punto.
celle() { find "artifacts/runs/$1" -name report.json 2>/dev/null \
          | xargs -r -n1 dirname | xargs -r -n1 dirname \
          | sort | uniq -c | awk '$1 >= 5' | wc -l; }

# Celle ancora da fare su un insieme di serie.
restanti() {
  local r=0 s g d
  for s in $1; do
    d=0
    for g in $TAGS; do d=$(( d + $(celle "ucr${s}_${g}") )); done
    r=$(( r + 34 - d ))
  done
  echo "$r"
}

prev=-1
while :; do
  r2=$(restanti "$G2_SER"); r4=$(restanti "$G4_SER")
  tot=$(( 306 - r2 - r4 ))

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
  g4=$(timeout 45 ssh -o BatchMode=yes -o ConnectTimeout=10 "$G4" \
        'ps -eo pid,ppid,args -u leonardo 2>/dev/null | awk "/[f]ederated_ev/ {pid[\$1]=1; pp[\$1]=\$2}
             END {n=0; for (p in pid) if (!(pp[p] in pid)) n++; printf \"%d \", n}"
         uptime | sed -E "s/.*load average: ([0-9]+[.,][0-9]+).*/\1/"' 2>/dev/null) \
     || g4="? ?"
  g4j=${g4%% *}; g4load=${g4##* }

  # Collisione VERA = due processi con la STESSA out-dir, dataset, cluster E arm.
  # ⚠ La chiave DEVE includere out-dir: confrontando solo dataset+cluster+arm ho segnalato due
  # falsi allarmi (`ucr086_v1` a K=64 e `ucr086_cb128` a K=128 sono lo stesso arm sulla stessa
  # serie, ma sono l'ablazione, e scrivono in cartelle diverse).
  coll=$(timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=10 "$G4" \
          'ps -eo pid,ppid,args -u leonardo 2>/dev/null | awk "
             /[f]ederated_ev/ {pid[\$1]=1; pp[\$1]=\$2; L[\$1]=\$0}
             END {for (p in pid) if (!(pp[p] in pid)) print L[p]}"' 2>/dev/null \
        | grep -oE '\-\-out-dir [^ ]+ --dataset [a-z0-9_]+ --cluster [a-z0-9_]+ --arms? [a-z0-9_,.]+' \
        | sort | uniq -d | wc -l)

  # AVANZAMENTO INTERNO. ⚠ AGGIUNTO 2026-08-02 21:16. Il conteggio celle sta fermo per 20-30
  # minuti di fila perche' una cella si chiude solo quando TUTTI e 5 i client hanno finito: il
  # "+0" ripetuto non distingue un lavoro che avanza da uno morto, ed e' esattamente la
  # confusione che oggi mi ha fatto gridare due volte a un blocco inesistente. Questi due numeri
  # guardano DENTRO i job:
  #   p4  = job al quinto client su cinque, cioe' a ridosso della chiusura di cella
  #   4/5 = celle con 4 report.json su 5, che chiudono entro il tick successivo
  # ⚠ I log stanno in logs/runs/, NON in artifacts/runs/ (li' non c'e' un solo .log): un find
  # sul percorso sbagliato torna vuoto e si legge identico a "tutto fermo".
  p4=$(for f in $(find logs/runs -name '*.log' -mmin -10 2>/dev/null); do
         grep -oE '_p[0-9]+\] ep ' "$f" 2>/dev/null | tail -1; done | grep -c '_p4\]')
  q45=$(find artifacts/runs -name report.json 2>/dev/null | xargs -r -n1 dirname \
        | xargs -r -n1 dirname | sort | uniq -c | awk '$1 == 4' | wc -l)

  wd=$(tail -1 logs/bridge_watchdog.log 2>/dev/null | grep -oE '\-> .*' | cut -c4-)
  delta=$(( prev < 0 ? 0 : tot - prev )); prev=$tot

  alert=""
  [ "${g4j:-0}" = "0" ] && [ "$r4" -gt 0 ] && alert=" ⚠ NESSUN JOB SU g4"
  [ "${g2j:-0}" = "0" ] && [ "$r2" -gt 0 ] && alert="$alert ⚠ NESSUN JOB SU g2"
  [ "$wd" != "ok" ] && alert="$alert ⚠ PONTE: $wd"
  [ "${coll:-0}" -gt 0 ] 2>/dev/null && alert="$alert 🔴 COLLISIONE: $coll celle doppie"

  # Capacita' sprecata. ⚠ CORRETTO 2026-08-02 19:40, un tick dopo averlo introdotto: la prima
  # versione confrontava i job di un nodo con le celle rimaste OVUNQUE, e ha gridato "g2 ha
  # capacita' libera" mentre g2 aveva 4 celle in tutto e tutte e 4 in esecuzione. Il residuo
  # era di g4, che g2 non puo' prendere (una serie, una macchina).
  # Condizione giusta: celle rimaste SU QUEL NODO oltre a quelle gia' in esecuzione li'.
  [ "$r4" -gt "${g4j:-0}" ] 2>/dev/null && [ "${g4j:-99}" -lt 12 ] 2>/dev/null \
    && alert="$alert 💤 g4: $g4j job ma $r4 celle da fare"
  [ "$r2" -gt "${g2j:-0}" ] 2>/dev/null && [ "${g2j:-99}" -lt 6 ] 2>/dev/null \
    && alert="$alert 💤 g2: $g2j job ma $r2 celle da fare"

  printf 'TICK %s | celle %s/306 (+%s) | in volo %s a p4, %s a 4/5 | g2 %s job load %s (%s da fare) | g4 %s job load %s (%s da fare) | ponte %s%s\n' \
    "$(date '+%H:%M')" "$tot" "$delta" "${p4:-?}" "${q45:-?}" "$g2j" "$g2load" "$r2" "$g4j" "$g4load" "$r4" "${wd:-?}" "$alert"
  sleep "$PERIOD"
done
