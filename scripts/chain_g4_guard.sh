#!/usr/bin/env bash
# Sorveglia `ucr_043` e ferma `chain_g4` nell'istante in cui arriva a 30/30.
#
# PERCHE' AUTOMATICO E NON A MANO. Dopo `ucr_043` la lista di `chain_g4` prosegue con ucr_086,
# ucr_170 e ucr_083 — che sono in corso in QUESTO momento su g086_g4, gpu5_g4 e gpu0_g4.
# `launch.sh` salta una cella solo se ne trova il report.json AL MOMENTO DEL DISPATCH, quindi
# una cella in corso non protegge: chain_g4 partirebbe sulle stesse celle, nella stessa
# cartella, mentre un altro processo ci sta scrivendo. Fra la fine di ucr_043 e il dispatch
# successivo passano pochi secondi: "me lo ricordo io" non e' un meccanismo.
#
# UCCIDERE NON PERDE NULLA: a 30/30 chain_g4 non ha piu' lavoro proprio, e tutto cio' che gli
# resta in lista e' coperto da flussi dedicati. Il rischio e' tutto dall'altra parte.
#
# ⚠ CORRETTO 2026-08-02 19:52 da una prova A VUOTO, prima che servisse. La versione precedente
# risolveva il pgid dal processo SCREEN, ma su g4 l'albero e':
#     421558 SCREEN            pgid=421558   <- lo screen, DA SOLO nel suo gruppo
#       421559 bash -lc        pgid=421559   <- il gruppo che contiene tutto il lavoro
#         421565 g4_chain.sh   pgid=421559
#           launch.sh -> federated_eval ...  pgid=421559
# `kill -9 -421558` avrebbe ammazzato solo l'involucro **lasciando la catena viva a
# dispatchare**, e avrebbe scritto "fatto" nel log. Ora il pgid si prende da `g4_chain.sh`,
# che sta nel gruppo giusto per costruzione.
set -u
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
LOG=logs/chain_g4_guard.log
G4=${G4:-leonardo@g4.etsisi.upm.es}
mkdir -p logs

celle() { find "artifacts/runs/$1" -name report.json 2>/dev/null \
          | xargs -r -n1 dirname | xargs -r -n1 dirname \
          | sort | uniq -c | awk '$1 >= 5' | wc -l; }

echo "[$(date '+%F %T')] guard avviata (pgid da g4_chain.sh): ferma chain_g4 a ucr_043 = 30/30" >> "$LOG"
while :; do
  n=$(( $(celle ucr043_v1) + $(celle ucr043_enc) + $(celle ucr043_cb128) + $(celle ucr043_proto_count) ))
  if [ "$n" -ge 30 ]; then
    echo "[$(date '+%F %T')] ucr_043 = $n/30 -> FERMO chain_g4" >> "$LOG"
    timeout 60 ssh -o BatchMode=yes "$G4" '
      pgid=$(ps -eo pgid,args --no-headers | grep "[g]4_chain.sh" | awk "{print \$1}" | head -1 | tr -d " ")
      echo "  pgid del lavoro: ${pgid:-NON TROVATO}"
      if [ -n "$pgid" ]; then
        echo "  processi nel gruppo prima: $(ps -eo pgid --no-headers | tr -d " " | grep -c "^${pgid}$")"
        kill -9 -"$pgid" 2>/dev/null
        sleep 3
        echo "  processi nel gruppo dopo:  $(ps -eo pgid --no-headers | tr -d " " | grep -c "^${pgid}$")"
      fi
      screen -S chain_g4 -X quit 2>/dev/null
      sleep 1
      echo "  chain_g4 ancora vivo? $(screen -ls | grep -c chain_g4)"
      echo "  screen rimasti:"; screen -ls | sed -n "2,\$p"' >> "$LOG" 2>&1
    echo "[$(date '+%F %T')] fatto. guard termina." >> "$LOG"
    exit 0
  fi
  sleep 30
done
