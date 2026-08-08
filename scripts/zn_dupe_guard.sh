#!/usr/bin/env bash
# SENTINELLA ANTI-DOPPIO-DISPATCH, su g2 e g4 insieme.
#
# ⚠ IL RISCHIO CHE CHIUDE. `launch.sh` salta una cella solo se l'out-json esiste GIA' al
# momento del dispatch. Due dispatcher su HOST DIVERSI non si vedono e non si aspettano:
# se prendono la stessa (tag, serie, arm) scrivono nella STESSA cartella di checkpoint
# — `artifacts/runs/<tag>/ckpt/<ds>/<serie>/seed0/<arm>/` — sul filesystem condiviso.
# Il risultato non e' un crash: e' un checkpoint misto fra due run, cioe' un numero
# sbagliato senza nessun errore. E' il modo caratteristico in cui questo repo si rompe.
#
# Casi reali oggi (2026-08-05, audit delle 18:50), tutti cross-host e non serializzati:
#   zn_enc/ucr_222/{fedproto,fedprox}     g2 li sta facendo · zn_chain.sh g4 li ha in coda
#   zn_main/ucr_222/{federated,cb_only,cb_only_ema}   g4 li sta facendo · zn_g2_finish li ha in coda
# Le contese su ucr_170 invece sono innocue: `zn_g4_after_chain.sh` aspetta esplicitamente
# che `zn_chain.sh g4` esca, quindi i due sono serializzati per costruzione.
#
# REGOLA: se la stessa cella ha due processi vivi, si uccide il PIU' GIOVANE — ha meno
# lavoro dentro — e si rimuove il suo albero di checkpoint parziale, che altrimenti farebbe
# ripartire un resume da uno stato scritto a quattro mani.
#
#   bash scripts/zn_dupe_guard.sh [secondi fra i controlli]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
EVERY=${1:-45}
G4=${G4:-leonardo@g4.etsisi.upm.es}
LOG="$REPO/evidence/dupe_guard_$(date -u +%Y%m%d).log"
mkdir -p "$REPO/evidence"
say() { echo "[$(TZ=Europe/Madrid date '+%F %H:%M:%S') Madrid][dupe] $*" | tee -a "$LOG"; }

say "=== SENTINELLA ANTI-DOPPIO-DISPATCH attiva (ogni ${EVERY}s, g2+g4) ==="

# Elenca "tag|serie|arm|host|pid|etime_secondi" per ogni job vivo.
snapshot() {
  for H in g2 g4; do
    if [ "$H" = g2 ]; then out=$(ps -eo pid,etimes,args --no-headers 2>/dev/null)
    else out=$(timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=10 "$G4" 'ps -eo pid,etimes,args --no-headers' 2>/dev/null); fi
    printf '%s\n' "$out" | awk -v h="$H" '
      /federated_eval\.py/ {
        cl=""; arm=""; tag=""
        for (i=1;i<=NF;i++) {
          if ($i=="--cluster") cl=$(i+1)
          else if ($i=="--arms") arm=$(i+1)
          else if ($i=="--out-json") { n=split($(i+1),p,"/"); for(j=1;j<n;j++) if (p[j]=="runs") tag=p[j+1] }
        }
        if (cl!="" && arm!="" && tag!="") print tag"|"cl"|"arm"|"h"|"$1"|"$2
      }'
  done
}

while :; do
  snap=$(snapshot)
  # chiavi presenti piu' di una volta, contando solo i PROCESSI PADRE (i worker del
  # DataLoader ereditano la riga di comando: senza questo, ogni cella sembrerebbe duplicata
  # una quarantina di volte).
  dupes=$(printf '%s\n' "$snap" | awk -F'|' '{k=$1"|"$2"|"$3; if (!($5 in seen)) {seen[$5]=1}} {print}' \
          | awk -F'|' '{k=$1"|"$2"|"$3; hosts[k]=hosts[k]" "$4} END {for (k in hosts) {n=split(hosts[k],a," "); u=""; for(i=1;i<=n;i++) if (index(u,a[i])==0) u=u" "a[i]; if (split(u,b," ")>1) print k"|"u}}')
  if [ -n "$dupes" ]; then
    while IFS='|' read -r tag cl arm hosts; do
      [ -z "$tag" ] && continue
      say "!! DOPPIO DISPATCH: $tag / $cl / $arm  su host:$hosts"
      # il piu' giovane = etime minore; una riga per host, prendo il pid padre (etime max per host)
      young_h=""; young_pid=""; young_et=999999
      while IFS='|' read -r t c a h p e; do
        [ "$t|$c|$a" = "$tag|$cl|$arm" ] || continue
        if [ "$e" -lt "$young_et" ]; then young_et=$e; young_h=$h; young_pid=$p; fi
      done <<< "$(printf '%s\n' "$snap")"
      [ -z "$young_pid" ] && continue
      say "   uccido il piu' giovane: host=$young_h pid=$young_pid (vivo da ${young_et}s)"
      if [ "$young_h" = g2 ]; then kill "$young_pid" 2>/dev/null
      else timeout 20 ssh -o BatchMode=yes "$G4" "kill $young_pid" 2>/dev/null; fi
      sleep 5
      ck="artifacts/runs/$tag/ckpt/ucr_split_w2p/$cl/seed0/$arm"
      if [ -d "$ck" ] && [ ! -f "artifacts/runs/$tag/ucr_split_w2p/${cl}__${arm}.json" ]; then
        say "   ⚠ checkpoint parziale scritto a quattro mani: NON lo cancello da solo."
        say "     controllare a mano: $ck  (l'altro processo ci sta ancora scrivendo)"
      fi
    done <<< "$dupes"
  fi
  sleep "$EVERY"
done
