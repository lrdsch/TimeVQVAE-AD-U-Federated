#!/usr/bin/env bash
# WAVE τ + ESTENSIONE ctrl — 2026-08-08, lanciata perché i GATE sono VERDI.
#
#   bash scripts/zn_s2_wave.sh ctrlext   # ctrl sulle 6 serie non-probe (n=10 completo)
#   bash scripts/zn_s2_wave.sh tau       # 3 tag × 4 serie probe: tau64, tau16, t64adam
#
# I GATE (zn_fedtokcp, tetto misurato — AUPRC recuperabile via stage-2):
#     ucr_011 +0.12 · ucr_043 +0.22 · ucr_170 +0.11 · ucr_014 +0.00
# e il null-test del resume regge (ctrl/014 0.945 vs A2 0.946; ctrl/043 +0.07, nel rumore).
#
# SCALA τ, pre-registrata: τ=64 (costo ~A2) → τ=16 → τ=1 SOLO se 64→16 e' monotono su
# val-fissa E AUPRC. FedAdam slr 0.03 entra come controllo composto (t64adam), non come
# scommessa. Niente agg-penalty qui (la porta il ctrl); niente snapshot (idem).
#
# ⚠️ S2_ROUNDS VA SEMPRE ESPLICITATO: con S1_ROUNDS=0 il default di launch.sh e'
#    S2_ROUNDS=S1_ROUNDS=0 — un run che finisce prima di cominciare.
# ⚠️ un tag per variante (out-json cieco ai knob) · resume PER SERIE (path diverso)
#    ⇒ un launch.sh per (tag, serie), GPU assegnata QUI round-robin (i dispatcher non si
#    vedono fra loro).
# ⚠️ patience: τ=64 → --fed-s2-patience 29 eval (≈1856 step, come A2); τ=16 con
#    --fed-s2-val-every 2 → 57 eval (≈1824 step). Tetti: τ=64→300 round; τ=16→1200.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
ROLE="${1:?ctrlext | tau}"
PROBE=(ucr_011 ucr_014 ucr_043 ucr_170)
REST=(ucr_001 ucr_082 ucr_083 ucr_086 ucr_222 ucr_229)
A2K='--window-normalization zscore --fed-enc-cb suffstat --fed-enc-prior partial --fed-enc-bn shared'
A2ROOT="$REPO/artifacts/runs/zn_a2/ckpt"
say() { echo "[$(date -u '+%F %T') UTC | $(TZ=Europe/Madrid date '+%H:%M') Madrid][s2-wave/$ROLE] $*"; }

"$PY" scripts/zn_owner.py --check >/dev/null || { say "!! partizione ROTTA — non parto"; exit 2; }

# GPU senza processi di ALTRI utenti (deroga utente 2026-08-08 per 0/5 se vuote).
FREE=()
while IFS=, read -r i u; do
  i="$(echo "$i" | tr -d ' ')"; u="$(echo "$u" | tr -d ' ')"; others=0
  for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
             | awk -F', ' -v x="$u" '$1==x{print $2}'); do
    o="$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')"
    [ -n "$o" ] && [ "$o" != "$(whoami)" ] && others=$((others+1))
  done
  [ "$others" -eq 0 ] && FREE+=("$i") || say "GPU$i: $others processi di ALTRI — esclusa"
done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
[ "${#FREE[@]}" -gt 0 ] || { say "!! nessuna GPU disponibile"; exit 3; }
case " ${FREE[*]} " in *" 0 "*|*" 5 "*) export G4_ALLOW_RESERVED=1
  say "deroga utente 2026-08-08: G4_ALLOW_RESERVED=1";; esac
bash "$REPO/scripts/_g4_gpu.sh" "${FREE[*]}" || exit 4
say "GPU disponibili: [${FREE[*]}]"
export NO_MPS=1 FEDVQ_AMP=fp16 SLOTS_PER_GPU=1

# ⚠️ NIENTE command substitution per il round-robin: `g=$(next_gpu)` eseguiva l'increment
# in una SUBSHELL e gidx restava 0 nel padre — il primo lancio ha messo TUTTI i job su
# GPU0. L'indice si tocca solo nel processo corrente.
gidx=0

launch_cell() {  # tag serie s2_rounds extra_knobs
  local tag="$1" s="$2" rounds="$3" knobs="$4"
  local g="${FREE[$(( gidx % ${#FREE[@]} ))]}"; gidx=$((gidx+1))
  say "=== $tag $s -> GPU$g (S2_ROUNDS=$rounds) ==="
  S1_ROUNDS=0 S2_ROUNDS="$rounds" PATIENCE=6 LAUNCH_GPUS="$g" LAUNCH_ONLY_CLUSTERS="$s" \
    bash scripts/launch.sh --cohort ucr2p_10 --tag "$tag" --arms federated_enc_fedavg \
    --extra "$A2K --fed-s2-val fixed $knobs --resume-from $A2ROOT/ucr_split_w2p/$s/seed0/federated_enc_fedavg_bn-shared_prior-partial" &
  sleep 6
}

case "$ROLE" in
  ctrlext)
    for s in "${REST[@]}"; do
      launch_cell zn_a2s2_ctrl "$s" 300 "--fed-s2-agg-penalty --fed-s2-snapshot-every 4"
    done
    wait ;;
  tau)
    for s in "${PROBE[@]}"; do launch_cell zn_a2s2_tau64 "$s" 300 "--tau-steps 64 --fed-s2-patience 29"; done
    for s in "${PROBE[@]}"; do launch_cell zn_a2s2_tau16 "$s" 1200 "--tau-steps 16 --fed-s2-val-every 2 --fed-s2-patience 57"; done
    for s in "${PROBE[@]}"; do launch_cell zn_a2s2_t64adam "$s" 300 "--tau-steps 64 --fed-s2-patience 29 --server-opt fedadam --server-lr 0.03"; done
    wait ;;
  tauext)
    # Le 6 serie non-probe dei 3 tag τ (n=10 per le righe di completezza del paper).
    # PARTE A SCIA: aspetta che i job in volo scendano sotto la soglia, per non rallentare
    # il probe — che e' quello che decide la scala τ. Poi 18 celle, round-robin.
    # ⚠️ ucr_222 e' l'outlier di budget (51 900 passi al best): tetti dedicati, o a τ=16
    #    tronca per costruzione (51900/16 = 3244 > 1200).
    while :; do
      # CELLE distinte, non processi: ogni job federated_eval spawna decine di worker
      # DataLoader che condividono la cmdline del padre — contarli dava 974 "job".
      NRUN=$(ps -eo args -u "$(whoami)" \
             | grep -oE "runs/zn_a2s2_(tau64|tau16|t64adam|ctrl)/ucr_split_w2p/ucr_[0-9]+__[a-z_]+" \
             | sort -u | wc -l)
      [ "$NRUN" -lt 6 ] && break
      say "in coda: $NRUN celle ancora in volo, riprovo fra 10 min"; sleep 600
    done
    for s in "${REST[@]}"; do
      r64=300; r16=1200
      [ "$s" = "ucr_222" ] && r64=1200 && r16=4500
      launch_cell zn_a2s2_tau64 "$s" "$r64" "--tau-steps 64 --fed-s2-patience 29"
      launch_cell zn_a2s2_tau16 "$s" "$r16" "--tau-steps 16 --fed-s2-val-every 2 --fed-s2-patience 57"
      launch_cell zn_a2s2_t64adam "$s" "$r64" "--tau-steps 64 --fed-s2-patience 29 --server-opt fedadam --server-lr 0.03"
    done
    wait ;;
  *) echo "ruolo sconosciuto '$ROLE'" >&2; exit 2;;
esac
say "=== $ROLE finito (rc=$?) ==="
