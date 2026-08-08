#!/usr/bin/env bash
# I DUE GATE DELLO STAGE-2 (PROJECT_S2_GATES.md §2) — 2026-08-08.
#
#   bash scripts/zn_s2_gates.sh fedtokcp   # GATE 0: prior centrale su tokenizer A2 congelato
#   bash scripts/zn_s2_gates.sh ctrl       # GATE 1: ctrl strumentato (resume + oracolo fisso)
#   bash scripts/zn_s2_gates.sh smoke      # smoke 3-round di ctrl su ucr_014, tag zn_smoke
#
# PERCHE'. L'audit del 2026-08-07 ha bocciato la premessa «ottimizzare meglio il prior ⇒
# detection migliore» (rho nulla-o-perversa su 16 arm x 6 serie). Prima di comprare varianti
# (tau/FedAdam/FedProx) si MISURA: (0) il tetto di ogni upgrade stage-2 — un prior centrale
# sui token pooled del tokenizer di A2, lecito come diagnostico perche' gli encoder di A2
# finiscono IDENTICI fra client (il codice lo VERIFICA, non lo assume); (1) il ctrl che
# logga cio' che serve a decidere: detect lungo la traiettoria (snapshot), best-vs-last,
# aggregation_penalty (il numero che decide FedProx).
#
# ⚠️ UN TAG PER VARIANTE: l'out-json e' cieco ai knob (cohort.py:256 + skip di launch.sh).
# ⚠️ ctrl usa --resume-from PER SERIE (path diverso) ⇒ un launch.sh per serie, 1 job l'uno.
# ⚠️ GPU: l'utente il 2026-08-08 ha detto «usa tutte le gpu di g4 se ti serve» ⇒ deroga
#    esplicita su 0/5, MA resta la regola dura: mai una scheda con processi di ALTRI utenti.
#    Il set si calcola qui sotto scheda per scheda; 0/5 entrano solo se VUOTE, e in quel
#    caso G4_ALLOW_RESERVED=1 e' visibile nel comando come da protocollo.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
ROLE="${1:?fedtokcp | ctrl | smoke}"
PROBE=(ucr_011 ucr_014 ucr_043 ucr_170)
A2K='--window-normalization zscore --fed-enc-cb suffstat --fed-enc-prior partial --fed-enc-bn shared'
A2ROOT="$REPO/artifacts/runs/zn_a2/ckpt"
say() { echo "[$(date -u '+%F %T') UTC | $(TZ=Europe/Madrid date '+%H:%M') Madrid][s2-gates/$ROLE] $*"; }

"$PY" scripts/zn_owner.py --check >/dev/null || { say "!! partizione ROTTA — non parto"; exit 2; }

# ── GPU utilizzabili: tutte quelle SENZA processi di altri utenti ────────────────────────
FREE=()
RESERVED_IN_USE=0
while IFS=, read -r i u; do
  i="$(echo "$i" | tr -d ' ')"; u="$(echo "$u" | tr -d ' ')"
  others=0
  for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
             | awk -F', ' -v x="$u" '$1==x{print $2}'); do
    o="$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')"
    [ -n "$o" ] && [ "$o" != "$(whoami)" ] && others=$((others+1))
  done
  if [ "$others" -gt 0 ]; then
    case "$i" in 0|5) RESERVED_IN_USE=1;; esac
    say "GPU$i: $others processi di ALTRI — esclusa"
  else
    FREE+=("$i")
  fi
done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
[ "${#FREE[@]}" -gt 0 ] || { say "!! nessuna GPU libera"; exit 3; }
# LAUNCH_GPUS pre-impostato vince (serve a spartire le schede fra ctrl e fedtokcp, che
# girano in parallelo e NON si vedono a vicenda: ogni launch.sh conta solo i propri slot).
export LAUNCH_GPUS="${LAUNCH_GPUS:-${FREE[*]}}"
# 0/5 nel set ⇒ serve la deroga esplicita (data dall'utente il 2026-08-08), visibile qui:
case " ${FREE[*]} " in *" 0 "*|*" 5 "*) export G4_ALLOW_RESERVED=1
  say "deroga utente 2026-08-08: G4_ALLOW_RESERVED=1 (0/5 vuote incluse)";; esac
bash "$REPO/scripts/_g4_gpu.sh" "$LAUNCH_GPUS" || exit 4
say "GPU prese: [$LAUNCH_GPUS]"

export NO_MPS=1 FEDVQ_AMP=fp16 SLOTS_PER_GPU="${SLOTS_PER_GPU:-1}"

case "$ROLE" in
  smoke)
    # 3 round di stage-2, snapshot a ogni round, penalty on: valida resume+oracolo+strumenti
    # in ~10 min. Tag zn_smoke: FUORI dal manifesto apposta, va cancellato dopo.
    export S1_ROUNDS=0 S2_ROUNDS=3 PATIENCE=6
    LAUNCH_ONLY_CLUSTERS=ucr_014 bash scripts/launch.sh --cohort ucr2p_10 --tag zn_smoke \
      --arms federated_enc_fedavg \
      --extra "$A2K --fed-s2-val fixed --fed-s2-agg-penalty --fed-s2-snapshot-every 1 --resume-from $A2ROOT/ucr_split_w2p/ucr_014/seed0/federated_enc_fedavg_bn-shared_prior-partial"
    ;;
  ctrl)
    # GATE 1. Stage-1 ripreso da zn_a2 (0 round addizionali), stage-2 FedAvg puro, oracolo
    # fisso, snapshot ogni 4 round (~8-10 per run), penalty per round. Un lancio per serie.
    export S1_ROUNDS=0 S2_ROUNDS=300 PATIENCE=6
    # UNA GPU per serie, assegnata QUI: i 4 launch.sh non si vedono a vicenda (ognuno conta
    # solo i propri slot), quindi senza assegnazione esplicita si impilerebbero tutti sulla
    # prima scheda del set.
    GSET=($LAUNCH_GPUS)
    i=0
    for s in "${PROBE[@]}"; do
      g="${GSET[$(( i % ${#GSET[@]} ))]}"; i=$((i+1))
      say "=== ctrl $s -> GPU$g ==="
      LAUNCH_GPUS="$g" LAUNCH_ONLY_CLUSTERS="$s" bash scripts/launch.sh --cohort ucr2p_10 \
        --tag zn_a2s2_ctrl --arms federated_enc_fedavg \
        --extra "$A2K --fed-s2-val fixed --fed-s2-agg-penalty --fed-s2-snapshot-every 4 --resume-from $A2ROOT/ucr_split_w2p/$s/seed0/federated_enc_fedavg_bn-shared_prior-partial" &
      sleep 8   # sfalsa i dispatcher: stesso tag, serie diverse — out-json distinti
    done
    wait
    ;;
  fedtokcp)
    # GATE 0. Un solo launch.sh: --fedtokcp-stage1-root e' costante fra le serie.
    # L'identita' degli encoder fra client e' VERIFICATA dal codice (SystemExit se falsa).
    export S1_ROUNDS=300 S2_ROUNDS=300 PATIENCE=6   # S1 ignorato dall'arm (nessun training s1)
    LAUNCH_ONLY_CLUSTERS="$(IFS=,; echo "${PROBE[*]}")" bash scripts/launch.sh \
      --cohort ucr2p_10 --tag zn_fedtokcp --arms fedtok_centralprior \
      --extra "--window-normalization zscore --fed-s2-val fixed --fedtokcp-stage1-root $A2ROOT"
    ;;
  *) echo "ruolo sconosciuto '$ROLE'" >&2; exit 2;;
esac
say "=== $ROLE finito (rc=$?) ==="
