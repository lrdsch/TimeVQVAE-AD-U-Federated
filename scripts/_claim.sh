#!/usr/bin/env bash
# PRESA IN CARICO ATOMICA DI UNA CELLA — da sorgere dentro launch.sh.
#
# ⚠ NON ANCORA ATTIVO. Va inserito in `launch.sh` quando NESSUNA istanza di `launch.sh` e'
# in esecuzione: bash legge gli script a offset di byte, quindi modificarlo mentre gira
# sposta le righe non ancora lette e da li' in poi esegue spazzatura. Vedi in fondo la patch.
#
# ── IL PROBLEMA ─────────────────────────────────────────────────────────────────
# `launch.sh` decide se dispatchare con `[[ -f "$out" ]]`: controlla e poi agisce, senza
# niente in mezzo. Due dispatcher su HOST DIVERSI non si vedono, guardano nello stesso
# istante, vedono entrambi "libera" e partono. Scrivono nella stessa cartella di checkpoint
# sul filesystem condiviso, e il risultato non e' un crash: e' un checkpoint misto fra due
# run, cioe' un numero sbagliato senza nessun errore.
#
# Tutte le compensazioni di questa campagna — liste di serie disgiunte scritte a mano in sei
# script, `zn_dupe_guard.py`, i waiter che si aspettano a vicenda — esistono per quella riga.
#
# ── PERCHE' `set -o noclobber` E NON UN LOCKFILE ────────────────────────────────
# `noclobber` fa aprire con O_EXCL: o il file non c'e' e lo crei, o fallisci. E' l'unica
# primitiva atomica disponibile qui, e MISURATA su questo mount: 200 celle contese fra g2 e
# g4 partendo nello stesso secondo -> 200 vincite, ZERO doppie (2026-08-05).
# `flock` non e' un'opzione: su sshfs/FUSE il lock non attraversa gli host.
#
# ── SCADENZA ────────────────────────────────────────────────────────────────────
# Un claim il cui dispatcher e' morto bloccherebbe la cella per sempre: l'orfana silenziosa
# e' peggio del doppione, perche' non la vede nessuno. Quindi il dispatcher fa da BATTITO:
# rinfresca l'mtime dei claim che tiene a ogni giro di reap (~20 s). Un claim piu' vecchio di
# CLAIM_STALE_MIN e senza out-json e' di un dispatcher morto, e si puo' rilevare.
set -uo pipefail
CLAIM_STALE_MIN="${CLAIM_STALE_MIN:-10}"

# claim_take <out_json> -> 0 se la cella e' NOSTRA, 1 se e' di un altro processo vivo
claim_take() {
  local out="$1" c="$1.claim"
  if ( set -o noclobber; printf '%s\t%s\t%s\n' "$(hostname)" "$$" "$(date -u +%FT%TZ)" > "$c" ) 2>/dev/null; then
    return 0
  fi
  # esiste gia': e' di un vivo o di un morto?
  local age_min
  age_min=$(( ( $(date +%s) - $(stat -c %Y "$c" 2>/dev/null || echo 0) ) / 60 ))
  if [[ ! -f "$out" && $age_min -ge $CLAIM_STALE_MIN ]]; then
    echo "  claim SCADUTO su $(basename "$out") (fermo da ${age_min} min, $(cat "$c" 2>/dev/null | tr '\t' ' ')): lo riprendo"
    rm -f "$c"
    ( set -o noclobber; printf '%s\t%s\t%s\n' "$(hostname)" "$$" "$(date -u +%FT%TZ)" > "$c" ) 2>/dev/null && return 0
  fi
  return 1
}

# claim_beat <out_json...> — rinfresca l'mtime dei claim ancora in lavorazione
claim_beat() { local o; for o in "$@"; do [[ -n "$o" ]] && touch "$o.claim" 2>/dev/null; done; return 0; }

# claim_release <out_json> — a cella chiusa (o fallita): il claim non serve piu'
claim_release() { rm -f "$1.claim" 2>/dev/null; return 0; }

# ─────────────────────────────────────────────────────────────────────────────────
# PATCH DA APPLICARE A launch.sh (tre punti, ~6 righe). Solo a dispatcher fermi.
#
#  1) dopo la definizione di PY/REPO, in cima:
#         source "$REPO/scripts/_claim.sh"
#
#  2) nel ciclo di dispatch, DOVE OGGI c'e':
#         if [[ -f "$out" ]]; then NSKIP=$((NSKIP+1)); continue; fi
#     mettere:
#         if [[ -f "$out" ]]; then NSKIP=$((NSKIP+1)); continue; fi
#         if ! claim_take "$out"; then
#           say "SKIP  $name — presa da un altro dispatcher"; NSKIP=$((NSKIP+1)); continue
#         fi
#
#  3) in `reap()`, dove si registra DONE:
#         claim_release "${SLOT_OUT[$k]}"
#     e a ogni giro di reap, prima del `sleep 20`:
#         claim_beat "${SLOT_OUT[@]}"
#     (serve una mappa SLOT_OUT[slot]=$out accanto a SLOT_NAME, tre righe)
#
# Con questo, `zn_dupe_guard.py` diventa superfluo e le liste di serie disgiunte scritte a
# mano smettono di essere l'unica difesa: restano un'ottimizzazione, non una condizione di
# correttezza.
