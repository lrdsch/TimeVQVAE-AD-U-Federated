#!/usr/bin/env bash
# GPU0 e GPU5 DI g4 SONO RISERVATE — guardiano, da chiamare prima di ogni lancio su g4.
#
#   bash scripts/_g4_gpu.sh "$LAUNCH_GPUS"     # esce 4 se la lista tocca 0 o 5
#   bash scripts/_g4_gpu.sh --libere           # stampa le schede che POSSIAMO usare
#
# ── LA REGOLA (utente, 2026-08-06) ──────────────────────────────────────────────
# **Su g4 non si usano GPU0 e GPU5 se non me lo dici espressamente.** Sono le due schede che
# l'utente cede ad altri. Non e' una questione di carico: anche vuote, non sono nostre da
# prendere.
#
# Restano utilizzabili **GPU1, GPU2, GPU3** (le tre 3090) e **GPU4** (Ada).
#
# ── DEROGA ESPLICITA ────────────────────────────────────────────────────────────
# Solo su indicazione diretta dell'utente, e va resa visibile nel comando:
#     G4_ALLOW_RESERVED=1 LAUNCH_GPUS="0 5" bash scripts/<lanciatore>.sh ...
# La variabile e' scomoda apposta: deve essere impossibile attivarla per distrazione.
#
# ── PERCHE' UN GUARDIANO E NON UNA NOTA ─────────────────────────────────────────
# Il 2026-08-06 alle 13:11 l'utente ha chiesto «abbiamo in programma di usarle o posso
# cederle?» e io, invece di rispondere, ho lanciato una cella su GPU0. Nessun danno (2 minuti,
# nessun out-json, checkpoint parziale rimosso), ma il punto e' che la scheda me la sono
# presa mentre lui stava decidendo di darla via.
#
# La lezione del progetto e' gia' scritta: un vincolo modellato sui PRODUTTORI noti — «questo
# script non tocca GPU0» — cade appena qualcuno lancia qualcosa a mano. Va modellata la
# PROPRIETA', e va controllata da qualcosa che gira. Vedi il coprifuoco Ada del 2026-08-05,
# che scrisse «OK» mentre il vincolo era violato da tre ore e mezza.
#
# ⚠️ `scripts/launch.sh` NON ha questo controllo dentro: al 2026-08-06 girava in 7 istanze e
# bash rilegge gli script dall'offset di byte, quindi modificarlo mentre gira gli fa eseguire
# spazzatura. PATCH DA APPLICARE A DISPATCHER FERMI, subito dopo la lettura di LAUNCH_GPUS:
#
#     if [[ "$(hostname -s)" == g4* ]]; then
#       bash "$REPO/scripts/_g4_gpu.sh" "$LAUNCH_GPUS" || exit 4
#     fi
#
# Finche' non c'e', la protezione e' che ogni lanciatore lo chiami da se'.
set -uo pipefail
RISERVATE="${G4_RESERVED_GPUS:-0 5}"

if [ "${1:-}" = "--libere" ]; then
  out=""
  for g in $(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' '); do
    riservata=0
    for r in $RISERVATE; do [ "$g" = "$r" ] && riservata=1; done
    [ "$riservata" = 0 ] && out="$out $g"
  done
  echo "${out# }"
  exit 0
fi

CHIESTE="${1:?uso: _g4_gpu.sh \"<lista GPU>\" | --libere}"
if [ "${G4_ALLOW_RESERVED:-0}" = "1" ]; then
  echo "[g4-gpu] ⚠ DEROGA ESPLICITA attiva (G4_ALLOW_RESERVED=1): consentite anche [$RISERVATE]" >&2
  exit 0
fi

viol=""
for g in $CHIESTE; do
  for r in $RISERVATE; do [ "$g" = "$r" ] && viol="$viol $g"; done
done
if [ -n "$viol" ]; then
  echo "[g4-gpu] ⛔ RIFIUTO: la lista chiede GPU[$viol], che sono RISERVATE su g4." >&2
  echo "[g4-gpu]    Usabili: $(bash "$0" --libere)" >&2
  echo "[g4-gpu]    Se l'utente l'ha detto espressamente: G4_ALLOW_RESERVED=1" >&2
  exit 4
fi
exit 0
