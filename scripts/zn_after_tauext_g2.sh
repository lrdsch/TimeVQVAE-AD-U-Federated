#!/usr/bin/env bash
# CONTROLLO ritmo-vs-budget (zn_a2s2_lep) — parte DA SOLO quando la catena tauext libera
# la corsia su GPU1. 2026-08-09, g2-only (g4 giù).
#
# Perché: tau64 fa 2,2-3,5× più passi locali di A2 prima del best. Questo tag pareggia la
# STEP-PATIENCE di tau64 (≈1856 passi) alzando LOCAL_EPOCHS per serie, senza toccare τ:
# il sync resta a fine round, quindi diventa PIÙ RARO — direzione opposta a tau64.
#   lep ≈ tau64            ⇒ il merito era budget/patience, non il ritmo
#   lep ≈ A2, tau64 sopra  ⇒ la leva è il sync frequente     (predizione: questa)
#
# ⚠️ nota pre-registrata: il confronto pulito di lep è contro zn_a2s2_ctrl (0,870/0,746/0,082),
#    non contro zn_a2 nudo — ctrl porta già --fed-s2-val fixed come questo tag.
# ⚠️ g2: SOLO GPU1, una cella per volta, mai GPU0 (ssanchez).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
TAG=zn_a2s2_lep
A2ROOT="$REPO/artifacts/runs/zn_a2/ckpt/ucr_split_w2p"
SUB=federated_enc_fedavg_bn-shared_prior-partial
LOG="$REPO/evidence/zn_a2s2_lep_20260809.log"
A2K='--window-normalization zscore --fed-enc-cb suffstat --fed-enc-prior partial --fed-enc-bn shared'

say() { echo "[$(date -u '+%F %T') UTC][lep-g2] $*" | tee -a "$LOG"; }

"$PY" scripts/zn_owner.py --check >/dev/null || { say "!! partizione ROTTA — non parto"; exit 2; }

say "attendo la fine della catena tauext (corsia unica su GPU1)…"
for _ in $(seq 1 1440); do         # max 24 ore (il primo waiter e' morto a 12h: sotto load
  pgrep -f "scripts/zn_tauext_g2.sh" >/dev/null 2>&1 || break   # 18 tauext ha superato le 12)
  sleep 60
done
pgrep -f "scripts/zn_tauext_g2.sh" >/dev/null 2>&1 && { say "!! tauext gira ancora dopo 24h — esco"; exit 3; }
say "tauext chiusa: $(find artifacts/runs/zn_a2s2_tau64/ucr_split_w2p -name '*.json' | wc -l)/10 celle tau64 a disco. Parto."

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

# E per serie: 6 round di patience × E epoche × nb batch ≈ 1856 passi (la step-patience di tau64)
run_cell() {  # serie epoche
  local s="$1" E="$2"
  local out="$REPO/artifacts/runs/$TAG/ucr_split_w2p/${s}__federated_enc_fedavg.json"
  [ -f "$out" ] && { say "$s già a disco — salto"; return 0; }
  gpu1_mine_only || exit 4
  say ">>> $TAG / $s (LOCAL_EPOCHS=$E, stage-2 only)"
  S1_ROUNDS=0 S2_ROUNDS=300 PATIENCE=6 LOCAL_EPOCHS="$E" LAUNCH_GPUS=1 LAUNCH_ONLY_CLUSTERS="$s" \
    bash scripts/launch.sh --cohort ucr2p_10 --tag "$TAG" --arms federated_enc_fedavg \
      --extra "$A2K --fed-s2-val fixed --resume-from $A2ROOT/$s/seed0/$SUB" >>"$LOG" 2>&1
  say "<<< $TAG / $s rc=$?"
}

run_cell ucr_011 24     # 6×24×13 = 1872
run_cell ucr_170 16     # 6×16×19 = 1824
run_cell ucr_043 31     # 6×31×10 = 1860
say "=== lep finito ==="
