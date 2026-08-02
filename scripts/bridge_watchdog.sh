#!/usr/bin/env bash
# Sentinella del ponte g2<->g4. Scrive una riga ogni ciclo in logs/bridge_watchdog.log.
#
# PERCHE'. Il 2026-08-01 il ponte e' andato in deadlock alle 15:54 e me ne sono accorto alle
# 17:06, quando l'utente ha chiesto un recap: 1h20m di wall clock e ~5 h di training in volo
# bruciate senza un solo messaggio d'errore. Il relay nuovo (_passive_relay.py) muore invece di
# bloccarsi, ma un mount che sparisce lascia comunque i job fermi: serve qualcuno che GUARDI.
#
# Cosa controlla, e perche' proprio questo:
#  * il relay e' vivo (processo presente)      -- se muore, il mount e' andato
#  * il mount risponde entro 15 s              -- `timeout` apposta: un FUSE bloccato NON
#                                                 ritorna errore, si pianta, ed e' esattamente
#                                                 il modo in cui l'ultimo guasto si e' nascosto
#  * i log dei run su g4 avanzano              -- il test che conta davvero: un mount che
#                                                 risponde a `ls` mentre i job sono fermi e'
#                                                 comunque un guasto
#
# NON riavvia niente da sola. Un riavvio automatico del ponte sotto job vivi rischia di
# lasciare mezze scritture in artifacts/, e la diagnosi va fatta guardando: la sentinella
# grida, la riparazione e' a mano.
set -u
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
LOG=logs/bridge_watchdog.log
PERIOD=${PERIOD:-300}
mkdir -p logs

# I tag che girano su g4: i loro log sono la prova di vita che conta.
newest_g4_log_age() {
  local newest=0 t
  for f in logs/runs/*/[a-z]*.log; do
    case "$f" in *ucr222*|*ucr229*|*gpu1probe*) continue;; esac   # questi sono di g2
    [ -f "$f" ] || continue
    t=$(stat -c %Y "$f" 2>/dev/null) || continue
    [ "$t" -gt "$newest" ] && newest=$t
  done
  [ "$newest" -eq 0 ] && { echo -1; return; }
  echo $(( $(date +%s) - newest ))
}

while :; do
  ts=$(date '+%F %T')
  nrelay=$(pgrep -c -f "[_]passive_relay.py" || true)
  if timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=10 leonardo@g4.etsisi.upm.es \
       "timeout 10 ls /home/leonardo/PhD/TimeVQVAE-AD-U-Federated/scripts >/dev/null" 2>/dev/null; then
    mnt=ok
  else
    mnt=BLOCCATO
  fi
  age=$(newest_g4_log_age)
  njob=$(timeout 20 ssh -o BatchMode=yes leonardo@g4.etsisi.upm.es \
           "pgrep -c -u leonardo -f '[f]ederated_ev' 2>/dev/null || echo 0" 2>/dev/null || echo "?")

  verdict=ok
  [ "$nrelay" -lt 2 ] && verdict="ALLARME: relay $nrelay/2"
  [ "$mnt" = BLOCCATO ] && verdict="ALLARME: mount non risponde"
  # 1800 s = 30 min. Un round su g4 sta sotto i 7 min e lo stage 2 scrive ogni poche epoche,
  # quindi mezz'ora di silenzio con job vivi non e' lentezza, e' un blocco.
  [ "$age" -gt 1800 ] 2>/dev/null && [ "${njob:-0}" != "0" ] && verdict="ALLARME: log fermi da ${age}s con $njob job vivi"

  printf '[%s] relay=%s/2 mount=%s log_piu_recente=%ss job_g4=%s -> %s\n' \
    "$ts" "$nrelay" "$mnt" "$age" "$njob" "$verdict" >> "$LOG"
  sleep "$PERIOD"
done
