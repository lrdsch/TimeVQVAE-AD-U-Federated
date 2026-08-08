#!/usr/bin/env bash
# Stampa gli indici delle GPU su cui possiamo lanciare, cioe' quelle SENZA processi di altri
# utenti. Vale su qualunque host: `bash scripts/free_gpus.sh` (locale) oppure
# `bash scripts/free_gpus.sh leonardo@g4.etsisi.upm.es`.
#
# ⚠ PERCHE' NON BASTA GUARDARE LA MEMORIA. Il 2026-08-05 ho scelto la GPU0 di g2 perche' era
# la meno carica e ci ho messo sopra una sonda: c'era gia' `ssanchez` con 1,5 GiB. Nello stesso
# giorno ho fatto l'errore opposto su g4, dove le schede 1/2/3 sembravano di altri (84-87% di
# utilizzo) ed erano tutte nostre. La memoria libera e l'utilizzo non dicono NIENTE su di chi
# sia una scheda: l'unica domanda giusta e' "chi possiede i processi che ci girano sopra".
#
# ⚠ IL DAEMON MPS NON CONTA. Su g2 `nvidia-cuda-mps-server` e' nostro e vive per giorni: se lo
# contassimo come occupazione, ogni scheda risulterebbe presa per sempre.
#
# Uscita: una riga con gli indici liberi separati da spazio (vuota se non ce n'e' nessuna).
# Con -v, una tabella leggibile su stderr.
set -uo pipefail

HOST=${1:-}
VERBOSE=0
[ "${1:-}" = "-v" ] && { VERBOSE=1; HOST=${2:-}; }
[ "${2:-}" = "-v" ] && VERBOSE=1

REMOTE='
me=$(id -un)
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader | tr -d " " > /tmp/.fg.$$
while IFS=, read -r idx uuid mem; do
  others=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | tr -d " " \
    | awk -F, -v u="$uuid" "\$1==u{print \$2}" \
    | while read -r p; do
        # il daemon MPS e nostro e non e un job: escluderlo o ogni scheda risulta occupata
        case "$(ps -o comm= -p "$p" 2>/dev/null | tr -d " ")" in nvidia-cuda-mps*) continue;; esac
        u=$(ps -o user= -p "$p" 2>/dev/null | tr -d " ")
        [ -n "$u" ] && [ "$u" != "$me" ] && echo "$u"
      done | sort -u | paste -sd,)
  echo "$idx|$mem|$others"
done < /tmp/.fg.$$
rm -f /tmp/.fg.$$'

if [ -z "$HOST" ]; then
  CENSUS=$(bash -c "$REMOTE")
else
  CENSUS=$(timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" "$REMOTE") \
    || { echo "non riesco a interrogare $HOST" >&2; exit 1; }
fi

# ⛔ RISERVATE. Su g4, GPU0 e GPU5 non si toccano nemmeno se completamente libere: sono le
# schede che l'utente cede ad altri (regola 2026-08-06, vedi scripts/_g4_gpu.sh e CLAUDE.md).
# Senza questo filtro lo script le annuncerebbe come disponibili — ed e' proprio lui che si
# consulta prima di ogni lancio, quindi la regola va applicata QUI e non solo al guardiano.
case "$HOST" in *g4*) RES="${G4_RESERVED_GPUS:-0 5}" ;; *) RES="" ;; esac
[ -n "${G4_ALLOW_RESERVED:-}" ] && RES=""

is_res() { for r in $RES; do [ "$r" = "$1" ] && return 0; done; return 1; }

if [ "$VERBOSE" = 1 ]; then
  printf 'GPU | mem | altri utenti\n' >&2
  while IFS='|' read -r idx mem others; do
    [ -z "$idx" ] && continue
    if is_res "$idx"; then note="⛔ RISERVATA (non nostra, anche se vuota)"
    elif [ -z "$others" ]; then note="-- libera per noi --"
    else note="$others"; fi
    printf '%3s | %8s | %s\n' "$idx" "$mem" "$note" >&2
  done <<<"$CENSUS"
fi

OUT=""
while IFS='|' read -r idx mem others; do
  [ -z "$idx" ] && continue
  is_res "$idx" && continue
  [ -z "$others" ] && OUT="$OUT$idx "
done <<<"$CENSUS"
echo "$OUT"
