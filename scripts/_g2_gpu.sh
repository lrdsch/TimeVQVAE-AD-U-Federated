#!/usr/bin/env bash
# QUALE GPU DI g2 POSSIAMO USARE — la regola, come codice.
#
#   export LAUNCH_GPUS="$(bash scripts/_g2_gpu.sh)"      # in ogni lanciatore su g2
#   bash scripts/_g2_gpu.sh --spiega                     # stampa anche il perche'
#
# ── LA REGOLA (utente, 2026-08-06) ──────────────────────────────────────────────
#   altri su 0 GPU  ->  usiamo ENTRAMBE
#   altri su 1 GPU  ->  usiamo l'ALTRA, e una sola
#   altri su 2 GPU  ->  ne prendiamo UNA sola, quella dove danno meno fastidio,
#                       senza impedirgli il lavoro
#
# In nessun caso occupiamo entrambe le schede quando qualcun altro sta lavorando.
#
# ── PERCHE' UN FILE E NON UNA NOTA ──────────────────────────────────────────────
# Il 2026-08-06 alle 03:35 ho letto `nvidia-smi --query-compute-apps` come «3 job per GPU,
# sono nostri, il tetto e' 6-7, quindi ci sono 6 slot liberi» e ho lanciato 5 job sopra a
# ssanchez. Quei 3+3 erano 2 suoi + il nostro `nvidia-cuda-mps-server` su GPU0, e 1 suo +
# mps-server + 1 nostro su GPU1: di NOSTRO girava un job solo.
#
# Due trappole in una riga di output:
#   · il conteggio dei processi per GPU **non dice di chi sono** — serve `ps -o user=`;
#   · `nvidia-cuda-mps-server` compare nella lista e **non e' un job**.
#
# Su g4 il controllo del proprietario l'ho sempre fatto (li' la regola e' piu' dura: mai una
# scheda con processi di altri). Su g2 non l'avevo mai fatto perche' la davo per nostra.
# Questo file esiste perche' la prossima volta non dipenda dal fatto che me ne ricordi.
set -uo pipefail
SPIEGA=0; [ "${1:-}" = "--spiega" ] && SPIEGA=1
ME="$(whoami)"

nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader > /tmp/_g2_apps.$$ 2>/dev/null || : > /tmp/_g2_apps.$$
nvidia-smi --query-gpu=index,uuid --format=csv,noheader > /tmp/_g2_idx.$$

declare -A ALTRI
IDX=()
while IFS=, read -r i u; do
  i="$(echo "$i" | tr -d ' ')"; u="$(echo "$u" | tr -d ' ')"
  IDX+=("$i"); ALTRI[$i]=0
  for p in $(grep "$u" /tmp/_g2_apps.$$ | cut -d, -f2 | tr -d ' '); do
    # il server MPS e' NOSTRO ma non e' un job: contarlo falsa il conto in entrambi i sensi
    [ "$(ps -o comm= -p "$p" 2>/dev/null)" = "nvidia-cuda-mps" ] && continue
    o="$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')"
    [ -n "$o" ] && [ "$o" != "$ME" ] && ALTRI[$i]=$(( ALTRI[$i] + 1 ))
  done
done < /tmp/_g2_idx.$$
rm -f /tmp/_g2_apps.$$ /tmp/_g2_idx.$$

OCCUPATE=(); LIBERE=()
for i in "${IDX[@]}"; do
  [ "${ALTRI[$i]}" -gt 0 ] && OCCUPATE+=("$i") || LIBERE+=("$i")
done

case "${#OCCUPATE[@]}" in
  0) SCELTA="${IDX[*]}"; MOTIVO="nessun altro utente sulle schede -> le usiamo entrambe" ;;
  1) SCELTA="${LIBERE[0]}"; MOTIVO="altri su GPU${OCCUPATE[0]} -> prendiamo l'altra, e una sola" ;;
  *) # tutte occupate: si sceglie quella con MENO processi altrui
     best="${OCCUPATE[0]}"
     for i in "${OCCUPATE[@]}"; do [ "${ALTRI[$i]}" -lt "${ALTRI[$best]}" ] && best="$i"; done
     SCELTA="$best"
     MOTIVO="altri su tutte -> ne condividiamo UNA sola, GPU$best (la meno carica: ${ALTRI[$best]} processi altrui)" ;;
esac

if [ "$SPIEGA" = 1 ]; then
  for i in "${IDX[@]}"; do echo "  GPU$i: ${ALTRI[$i]} processi di ALTRI utenti" >&2; done
  echo "  -> $MOTIVO" >&2
fi
echo "$SCELTA"
