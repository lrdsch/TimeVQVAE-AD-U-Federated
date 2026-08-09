#!/usr/bin/env bash
# COMPOSIZIONE cbfa × tau64 — 2026-08-09.
#
#   bash scripts/zn_cbfa_tau64.sh
#
# I due upgrade migliori a oggi non erano mai stati combinati, e sono ORTOGONALI:
#   cbfa   -> stage 1: --fed-enc-cb suffstat → fedavg  (il dizionario si MUOVE in locale
#             via EMA e il server lo media, invece di restare congelato con M-step pooled)
#   tau64  -> stage 2: --tau-steps 64        (ritmo di sincronizzazione del prior)
#
# La cella è STAGE-2-ONLY: lo stage 1 lo prende a prestito da zn_170_cbfa con --resume-from,
# esattamente come zn_a2s2_tau64 lo prende da zn_a2. Costa circa un terzo di una run piena.
# Il null-test del resume regge già (zn_a2s2_ctrl/014 0,945 contro A2 0,946).
#
# ⚠️ S2_ROUNDS VA SEMPRE ESPLICITATO: con S1_ROUNDS=0 il default di launch.sh è
#    S2_ROUNDS=S1_ROUNDS=0 — un run che finisce prima di cominciare.
# ⚠️ --fed-s2-patience 29: con τ=64 sono ≈1856 step di pazienza, cioè gli stessi di A2 e di
#    zn_a2s2_tau64. Cambiarla romperebbe il confronto appaiato con tau64.
# ⚠️ g2: SOLO GPU1. Su GPU0 lavora ssanchez e non si tocca nemmeno se si svuota.
#    Il collo di bottiglia vero di g2 è la CPU (16 core) ⇒ una cella per volta, in serie.
# ⚠️ ucr_043 parte solo quando zn_170_cbfa/043 ha finito: il suo stage-1 non esiste ancora.
#    ucr_014 non c'è: cbfa non ha stage-1 là e la serie è satura (A2 0,939 / tau64 0,941).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
TAG=zn_cbfa_tau64
SRC="$REPO/artifacts/runs/zn_170_cbfa/ckpt/ucr_split_w2p"
SUB=federated_enc_fedavg_bn-shared_cb-fedavg_prior-partial
LOG="$REPO/evidence/zn_cbfa_tau64_20260809.log"

say() { echo "[$(date -u '+%F %T') UTC | $(TZ=Europe/Madrid date '+%H:%M') Madrid][cbfa×tau64] $*" | tee -a "$LOG"; }

"$PY" scripts/zn_owner.py --check >/dev/null || { say "!! partizione ROTTA — non parto"; exit 2; }

# g2: GPU1 dev'essere libera da processi di ALTRI utenti. Mai GPU0 (ssanchez).
gpu1_mine_only() {
  local u1 p o
  u1="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==1{print $2}')"
  for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
             | awk -F', ' -v x="$u1" '$1==x{print $2}'); do
    o="$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')"
    [ -n "$o" ] && [ "$o" != "$(whoami)" ] && { say "!! GPU1 ha processi di $o — mi fermo"; return 1; }
  done
  return 0
}

export NO_MPS=1 FEDVQ_AMP=fp16 SLOTS_PER_GPU=1
export CUDA_MPS_PIPE_DIRECTORY="$PWD/.no_mps_pipe_inesistente"
KNOBS='--window-normalization zscore --fed-enc-cb fedavg --fed-enc-prior partial --fed-enc-bn shared --fed-s2-val fixed --tau-steps 64 --fed-s2-patience 29'

run_cell() {  # serie
  local s="$1"
  local out="$REPO/artifacts/runs/$TAG/ucr_split_w2p/${s}__federated_enc_fedavg.json"
  [ -f "$out" ] && { say "$s già a disco — salto"; return 0; }
  gpu1_mine_only || return 1
  say ">>> $TAG / $s  (stage-2 only, resume da zn_170_cbfa)"
  S1_ROUNDS=0 S2_ROUNDS=300 PATIENCE=6 LAUNCH_GPUS=1 LAUNCH_ONLY_CLUSTERS="$s" \
    bash scripts/launch.sh --cohort ucr2p_10 --tag "$TAG" --arms federated_enc_fedavg \
      --extra "$KNOBS --resume-from $SRC/$s/seed0/$SUB" >>"$LOG" 2>&1
  say "<<< $TAG / $s  rc=$?"
}

# 1) le due serie il cui stage-1 cbfa è già a disco
for s in ucr_011 ucr_170; do
  [ -d "$SRC/$s/seed0/$SUB" ] || { say "!! manca lo stage-1 cbfa di $s — salto"; continue; }
  run_cell "$s" || exit 3
done

# 2) ucr_043: aspetta che zn_170_cbfa/043 chiuda (stage-1 ancora in corso al lancio)
DEP="$REPO/artifacts/runs/zn_170_cbfa/ucr_split_w2p/ucr_043__federated_enc_fedavg.json"
if [ ! -f "$DEP" ]; then
  say "attendo zn_170_cbfa/ucr_043 (stage-1 in corso)…"
  for _ in $(seq 1 240); do            # max 4 ore
    [ -f "$DEP" ] && break
    pgrep -f 'cluster ucr_043 --arms federated_enc_fedavg' >/dev/null 2>&1 || {
      sleep 20; [ -f "$DEP" ] || { say "!! zn_170_cbfa/043 è morto senza json — 043 salta"; exit 4; }; }
    sleep 60
  done
fi
[ -d "$SRC/ucr_043/seed0/$SUB" ] && run_cell ucr_043 || say "!! stage-1 cbfa di 043 assente — 043 salta"

say "=== catena finita ==="
