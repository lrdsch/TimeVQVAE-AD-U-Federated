#!/usr/bin/env bash
# DUE RAMI AGGIUNTIVI sulla coorte `ucr2p_10`. Interventi 1 e 2 del piano di riparazione
# (memoria fed-fix-plan-2026-08-05).
#
#   ramo N — tag `zn_norev` — arm `federated_cb_only_ema_norevive`
#     EMA del server ACCESA, rianimazione dei codici morti SPENTA. Discrimina il MECCANISMO
#     dell'evacuazione del dizionario: se il cricchetto è la causa (un codice muore ->
#     rianimato come duplicato di un centroide GIÀ FUSO -> ricattura un misto cross-client
#     -> rivola nel vuoto), spegnere la rianimazione ferma l'evacuazione. Se il dizionario
#     si evacua lo stesso, la causa è nel MERGE.
#     Contrasto appaiato: `norevive − cb_only_ema`.
#
#   ramo A1 — tag `zn_a1` — arm `federated_enc_fedavg --fed-enc-bn shared`
#     Il default `buffers_local` condivide gamma/beta ma tiene LOCALI le running stats: in
#     valutazione ogni client usa una CHIMERA (pesi del consenso, statistiche sue), che la
#     sonda D2b ha misurato PEGGIORE dell'encoder straniero intero (accordo 0,10-0,13). Con
#     `shared` le statistiche sono messe in comune per legge della varianza totale.
#     Contrasto appaiato: `zn_a1 − zn_enc` sullo stesso arm ⇒ isola il SOLO regime BN.
#     Gate funzionale: `scripts/zn_a1_gate.sh`, accordo di token = 100%.
#
# ⚠ RAMO N SOLO DOVE LA RIANIMAZIONE SCATTA DAVVERO (deciso 2026-08-05 alle 14:20).
# `_server_merge` esegue `if revive and n_dead > 0:` e consuma l'RNG SOLO lì dentro. Su una
# serie che non ha mai un codice morto, `revive=True` e `revive=False` danno run
# BIT-IDENTICHE: la cella non è un risultato nullo misurato, è un pareggio per costruzione.
# Misurato sui log di `cb_only_ema`: la rianimazione scatta su 011 (5), 014 (2), 043 (1),
# 222 (3), 229 (10) e MAI su 001, 082, 083, 086, 170 (dead_max = 0,0%).
# Quindi il ramo N gira sulle 5 informative + `ucr_001` come CONTROLLO DI DETERMINISMO: se
# torna bit-identica alla sua gemella, è anche la prova che la pipeline è deterministica.
# Le altre quattro sono tagliate: costavano ~20 h-GPU per uno 0,000 già noto, e in un test
# appaiato a 10 serie sarebbero stati 5 pareggi garantiti che tolgono potenza.
#
# ⚠ LA LISTA DI SERIE È PER RAMO, NON PER HOST. I contrasti sono DENTRO la serie, quindi
# ogni cella nuova deve girare sull'architettura del SUO partner -- e i due rami hanno
# partner diversi. Su `ucr_222`/`ucr_229` il partner di N (`cb_only_ema`) è girato su
# g4-3090 mentre quello di A1 (`enc_fedavg`) è girato su g2: la stessa serie va su host
# DIVERSI a seconda del ramo. Una lista sola per host lo sbagliava.
#
#   bash scripts/zn_branches.sh g2
#   bash scripts/zn_branches.sh g4
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

HOST="${1:?serve g2 o g4}"
case "$HOST" in
  g2) export LAUNCH_GPUS="1"; export SLOTS_PER_GPU="${SLOTS_PER_GPU:-6}"
      # ⚠ RIDOTTA il 2026-08-05 alle 17:10: ucr_001/011/229 sono passate al lancio
      # urgente su Ada (`zn_a1_rush.sh`). Se restassero qui, questo waiter e quel
      # dispatcher — che gira su g4 e quindi non si vedono a vicenda — potrebbero
      # prendere la STESSA cella e scriverla nella stessa cartella di checkpoint.
      SER_A1="ucr_222"
      SER_NOREV="ucr_001,ucr_011"                  # cb_only_ema di queste 2 è girato su g2
      ORDER="a1 norev"          # g2 fa A1 per primo: dà presto la risposta del gate
      ;;
  g4) export LAUNCH_GPUS="1 2 3"  # SOLO le 3090: 0/4/5 sono Ada, restituite alle 09:00
      export SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"
      export NO_MPS=1             # MAI un daemon MPS su un host con 20+ utenti
      SER_A1="ucr_014,ucr_043,ucr_082,ucr_083,ucr_086,ucr_170"
      SER_NOREV="ucr_014,ucr_043,ucr_222,ucr_229"  # cb_only_ema di queste 4 è su g4-3090
      ORDER="norev a1"          # g4 (host più grande) fa per primo il ramo di priorità 1
      ;;
  *) echo "host sconosciuto '$HOST' (g2|g4)" >&2; exit 2;;
esac
export FEDVQ_AMP=fp16            # come tutta la campagna: due architetture, una aritmetica

ZN='--window-normalization zscore'
say() { echo "[$(date -u '+%F %T') UTC | $(TZ=Europe/Madrid date '+%H:%M') Madrid][rami/$HOST] $*"; }

# ── attesa: mai due dispatcher sullo stesso host ─────────────────────────────────
# `launch.sh` ha un pool di slot suo e non sa degli altri: due in parallelo raddoppiano il
# carico per scheda, e su g4 è misurato che oltre 3 job per 3090 il throughput PEGGIORA
# (0,93× da 9 a 15 job).
#
# ⚠ Il filtro guarda il TERZO campo di `ps -eo pid,args` (lo script), non `pgrep -f`: il
# pattern matcherebbe anche lo SCREEN che ospita la catena -- e, mettendoci dentro anche
# `zn_branches.sh`, questo processo aspetterebbe SÉ STESSO per sempre.
# `zn_ada_curfew*.sh`, `zn_ada_guard.sh` e `zn_verify_g4_free.sh` sono ESCLUSI apposta: non
# dispatchano niente e vivono a lungo, aspettarli vorrebbe dire non partire mai.
DISPATCHERS='zn_chain[.]sh|zn_g4_ada[.]sh|zn_g4_pickup[.]sh|zn_g2_finish[.]sh|zn_g4_fill[.]sh|launch[.]sh'
alive() { ps -eo pid,args --no-headers \
          | awk -v re="$DISPATCHERS" '$2=="bash" && $3 ~ re {n++} END{exit !n}'; }

say "A1: $SER_A1"
say "N : $SER_NOREV"
say "gpu: $LAUNCH_GPUS · slot/gpu: $SLOTS_PER_GPU · ordine: $ORDER"
if alive; then
  say "aspetto che i dispatcher di $HOST finiscano (controllo ogni 3 min)"
  while alive; do sleep 180; done
fi
say "host libero, parto."

run_norev() {
  [ -z "$SER_NOREV" ] && { say "ramo N: niente da fare su $HOST"; return; }
  say "=== ramo N: cb_only_ema_norevive su $SER_NOREV ==="
  LAUNCH_ONLY_CLUSTERS="$SER_NOREV" bash scripts/launch.sh --cohort ucr2p_10 --tag zn_norev \
    --arms federated_cb_only_ema_norevive --extra "$ZN"
  say "=== ramo N finito (rc=$?) ==="
}

run_a1() {
  # NIENTE --fedproto-agg qui: il suo unico proprietario è federated_enc_fedproto, e
  # federated_eval esce con errore se nessun arm richiesto legge un flag passato (è la
  # protezione contro le run che DICHIARANO un trattamento mai applicato). Tutto il resto
  # è identico a `zn_enc`, così l'unica differenza col partner è il regime BN.
  [ -z "$SER_A1" ] && { say "ramo A1: niente da fare su $HOST"; return; }
  say "=== ramo A1: enc_fedavg con BN condivisa su $SER_A1 ==="
  LAUNCH_ONLY_CLUSTERS="$SER_A1" bash scripts/launch.sh --cohort ucr2p_10 --tag zn_a1 \
    --arms federated_enc_fedavg \
    --extra "$ZN --fed-enc-cb suffstat --fed-enc-prior local --fed-enc-bn shared"
  say "=== ramo A1 finito (rc=$?) ==="
}

for step in $ORDER; do
  case "$step" in norev) run_norev;; a1) run_a1;; esac
done
say "RAMI $HOST COMPLETI"
