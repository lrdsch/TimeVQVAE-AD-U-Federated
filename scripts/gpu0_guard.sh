#!/usr/bin/env bash
# Ferma il ciclo di `g4_chain_gpu0.sh` appena ha dispatchato TUTTE e 8 le celle di `ucr083_enc`.
#
# PERCHE'. Dopo `enc` la lista di gpu0_g4 prosegue con `cb128` e `proto_count`, che dalle 22:47
# sono di due flussi dedicati (`g083tail` orfano e `g083proto`). `launch.sh` salta una cella solo
# se ne trova il report.json AL MOMENTO DEL DISPATCH: una cella in corso non protegge, quindi
# gpu0_g4 partirebbe sulle stesse cartelle mentre gli altri ci scrivono. I margini sono stretti
# — gpu0_g4 arriva a cb128 verso le 04:30, i flussi dedicati chiudono verso le 03:50 — e 40
# minuti non sono un meccanismo.
#
# CONDIZIONE. 4 log `ucr_split_w2p__*` dentro logs/runs/ucr083_enc/ significa che la seconda onda
# e' stata dispatchata, cioe' tutte e 8 le celle sono in volo o finite. Solo allora si uccide.
# Prima sarebbe una perdita secca: le 4 celle w2p non le lancerebbe piu' nessuno.
#
# COSA SOPRAVVIVE. Si uccide **solo** il pid del ciclo. I `federated_eval` gia' partiti restano
# vivi come orfani e finiscono: `launch.sh:352` li lancia con `nohup ... > file 2>&1 &`, cioe'
# redirezione diretta su file e non una pipe verso il padre. Verificato due volte stasera.
#
# ⚠ IL FILTRO. `awk "/[g]4_chain_gpu0.sh/"` prenderebbe lo **SCREEN**, la cui riga di comando
# contiene il nome dello script come argomento — errore fatto due volte oggi. Si seleziona il
# processo la cui riga di comando *inizia* con `bash scripts/...`. E mai `kill -9 -<pgid>`: il
# gruppo contiene anche i job in volo.
set -u
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
LOG=logs/gpu0_guard.log
G4=${G4:-leonardo@g4.etsisi.upm.es}
mkdir -p logs

echo "[$(date '+%F %T')] guard avviata: fermo gpu0_g4 quando ucr083_enc ha dispatchato le 8 celle" >> "$LOG"
while :; do
  n=$(ls logs/runs/ucr083_enc/ucr_split_w2p__*.log 2>/dev/null | wc -l)
  vivo=$(timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=10 "$G4" \
          'ps -eo pid,args -u leonardo | awk "\$2==\"bash\" && \$3==\"scripts/g4_chain_gpu0.sh\" {print \$1}"' 2>/dev/null)
  if [ -z "$vivo" ]; then
    echo "[$(date '+%F %T')] gpu0_g4 non c'e' piu' (w2p=$n): niente da fare, guard termina." >> "$LOG"
    exit 0
  fi
  if [ "$n" -ge 4 ]; then
    echo "[$(date '+%F %T')] ucr083_enc ha 4 log w2p -> fermo il ciclo (pid $vivo)" >> "$LOG"
    timeout 30 ssh -o BatchMode=yes "$G4" "kill $vivo" >> "$LOG" 2>&1
    sleep 5
    timeout 30 ssh -o BatchMode=yes "$G4" \
      'echo "  ciclo ancora vivo? $(ps -eo pid,args -u leonardo | awk "\$2==\"bash\" && \$3==\"scripts/g4_chain_gpu0.sh\"" | wc -l)"
       echo "  job enc padri vivi: $(ps -eo pid,ppid,args -u leonardo | awk "/[f]ederated_ev/ {pid[\$1]=1; pp[\$1]=\$2; L[\$1]=\$0} END {for (p in pid) if (!(pp[p] in pid)) print L[p]}" | grep -c ucr083_enc)"' >> "$LOG" 2>&1
    echo "[$(date '+%F %T')] fatto. guard termina." >> "$LOG"
    exit 0
  fi
  sleep 120
done
