#!/usr/bin/env bash
# TAUEXT su g2 — 2026-08-09. g4 è giù: le 6 serie non-probe di zn_a2s2_tau64 girano qui.
#
# È la parte tau64-only del ruolo `tauext` di scripts/zn_s2_wave.sh (scritto per g4, mai
# girato): completa la riga τ=64 a n=10, che è ciò che serve per proporla come config
# raccomandata. tau16/t64adam restano fuori: con una GPU sola si fa prima il flagship.
#
# Celle STAGE-2-ONLY: resume dallo stage-1 di zn_a2 (stesso pattern di zn_a2s2_tau64 sulle
# 4 probe). Ownership: zn_a2s2_tau64 possiede già tutte e 10 le serie (check passato).
#
# ⚠️ g2: SOLO GPU1, mai GPU0 (ssanchez). Una cella per volta: sulla GPU1 girano anche le
#    2 corsie c50 e il muro è la CPU (16 core). launch.sh è idempotente: rilanciare è gratis.
# ⚠️ S2_ROUNDS esplicito (con S1_ROUNDS=0 il default sarebbe 0 = run vuota).
# ⚠️ patience 29 eval come le 4 probe: cambiarla romperebbe l'appaiamento interno del tag.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
TAG=zn_a2s2_tau64
A2ROOT="$REPO/artifacts/runs/zn_a2/ckpt/ucr_split_w2p"
SUB=federated_enc_fedavg_bn-shared_prior-partial
LOG="$REPO/evidence/zn_tauext_g2_20260809.log"
A2K='--window-normalization zscore --fed-enc-cb suffstat --fed-enc-prior partial --fed-enc-bn shared'

say() { echo "[$(date -u '+%F %T') UTC][tauext-g2] $*" | tee -a "$LOG"; }

"$PY" scripts/zn_owner.py --check >/dev/null || { say "!! partizione ROTTA — non parto"; exit 2; }

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

# ordine: finestra crescente (le pesanti per ultime, così i primi numeri arrivano presto)
for s in ucr_222 ucr_229 ucr_001 ucr_086 ucr_082 ucr_083; do
  out="$REPO/artifacts/runs/$TAG/ucr_split_w2p/${s}__federated_enc_fedavg.json"
  [ -f "$out" ] && { say "$s già a disco — salto"; continue; }
  [ -d "$A2ROOT/$s/seed0/$SUB" ] || { say "!! manca lo stage-1 A2 di $s — salto"; continue; }
  gpu1_mine_only || exit 3
  say ">>> $TAG / $s (stage-2 only, τ=64)"
  S1_ROUNDS=0 S2_ROUNDS=300 PATIENCE=6 LAUNCH_GPUS=1 LAUNCH_ONLY_CLUSTERS="$s" \
    bash scripts/launch.sh --cohort ucr2p_10 --tag "$TAG" --arms federated_enc_fedavg \
      --extra "$A2K --fed-s2-val fixed --tau-steps 64 --fed-s2-patience 29 --resume-from $A2ROOT/$s/seed0/$SUB" >>"$LOG" 2>&1
  say "<<< $TAG / $s rc=$?"
done
say "=== tauext g2 finito ==="
