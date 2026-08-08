#!/usr/bin/env bash
# I 4 FIX DI ucr_170 — 2026-08-08. Meccanismo misurato (memoria
# ucr170-and-floor-series-mechanisms): UN solo blocco FP [25366:25463], motivo RARO il cui
# unico parente di train vive nello shard 10% di p1 ⇒ la media/merge pesati sul conteggio
# ERODONO la competenza di minoranza. Più una DETONAZIONE separata a r21-22 (enc_drift 9.46).
#
#   bash scripts/zn_170_fixes.sh smoke   # valida i percorsi NUOVI (ctfp, ur, k128) in ~15 min
#   bash scripts/zn_170_fixes.sh run     # le 4 celle vere, una GPU l'una
#
# Le 4 celle (tutte: cluster ucr_170, seed 0, z-norm, oracolo fisso):
#   zn_170_ctfp  arm centraltok_fedprior     tokenizer CENTRALE + prior federato A2.
#                DIAGNOSTICO: completa il 2×2 {A2 0.086, fedtokcp 0.202, centralized 0.878}.
#   zn_170_ur    A2 + --fed-enc-cb union_recluster   il fix che mira al meccanismo:
#                centroidi per-client → union → farthest-point + Lloyd pesato (k-FED).
#   zn_170_sched A2 + --fed-enc-sched cosine         contro la detonazione r21-22.
#   zn_170_k128  A2 + --codebook-size 128            capacità: i motivi rari tengono codici.
#
# PREDIZIONI (pre-registrate qui): ctfp ≈ centralized ⇒ colpa SOLO stage-1; ur > ctrl(0.086)
# se l'erosione è la causa; sched da solo ~ctrl; k128 > ctrl se il collo è la capacità.
# Confronto: contro zn_a2 (0.086) e ctrl; il rumore a serie singola è ±0.14 (AUPRC).
#
# ⚠️ S2_ROUNDS SEMPRE ESPLICITO · un tag per variante · niente resume (lo stage-1 È il fix;
#    solo ctfp carica lo stage-1 centrale di zn_main via --fedtokcp-stage1-root).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
ROLE="${1:?smoke | run}"
ZK='--window-normalization zscore --fed-s2-val fixed'
A2K="$ZK --fed-enc-cb suffstat --fed-enc-prior partial --fed-enc-bn shared"
CTFP_ROOT="$REPO/artifacts/runs/zn_main/ckpt"
say() { echo "[$(date -u '+%F %T') UTC | $(TZ=Europe/Madrid date '+%H:%M') Madrid][170-fixes/$ROLE] $*"; }

"$PY" scripts/zn_owner.py --check >/dev/null || { say "!! partizione ROTTA — non parto"; exit 2; }

# GPU senza processi di ALTRI utenti (deroga utente 2026-08-08 per 0/5 se vuote),
# ordinate per numero di NOSTRI job (le meno cariche prima: la wave τ è ancora in volo).
FREE=()
while IFS=, read -r i u; do
  i="$(echo "$i" | tr -d ' ')"; u="$(echo "$u" | tr -d ' ')"; others=0; ours=0
  for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
             | awk -F', ' -v x="$u" '$1==x{print $2}'); do
    o="$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')"
    if [ -n "$o" ] && [ "$o" != "$(whoami)" ]; then others=$((others+1)); else ours=$((ours+1)); fi
  done
  [ "$others" -eq 0 ] && FREE+=("$ours $i") || say "GPU$i: $others processi di ALTRI — esclusa"
done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
[ "${#FREE[@]}" -gt 0 ] || { say "!! nessuna GPU disponibile"; exit 3; }
mapfile -t FREE < <(printf '%s\n' "${FREE[@]}" | sort -n | awk '{print $2}')
case " ${FREE[*]} " in *" 0 "*|*" 5 "*) export G4_ALLOW_RESERVED=1
  say "deroga utente 2026-08-08: G4_ALLOW_RESERVED=1 (0/5 vuote incluse)";; esac
bash "$REPO/scripts/_g4_gpu.sh" "${FREE[*]}" || exit 4
say "GPU (meno cariche prima): [${FREE[*]}]"
export NO_MPS=1 FEDVQ_AMP=fp16 SLOTS_PER_GPU=1

gidx=0
launch_cell() {  # tag arm s1_rounds s2_rounds extra  (⚠️ niente subshell sull'indice)
  local tag="$1" arm="$2" s1="$3" s2="$4" extra="$5"
  local g="${FREE[$(( gidx % ${#FREE[@]} ))]}"; gidx=$((gidx+1))
  say "=== $tag ($arm) -> GPU$g (S1=$s1 S2=$s2) ==="
  S1_ROUNDS="$s1" S2_ROUNDS="$s2" PATIENCE=6 LAUNCH_GPUS="$g" LAUNCH_ONLY_CLUSTERS=ucr_170 \
    bash scripts/launch.sh --cohort ucr2p_10 --tag "$tag" --arms "$arm" --extra "$extra" &
  sleep 6
}

case "$ROLE" in
  smoke)
    # Solo i percorsi NUOVI o a rischio-shape: ctfp (arm nuovo), ur (merge nuovo),
    # k128 (vocabolario prior segue K?). sched è un LR schedule già esistente: no smoke.
    for t in zn_smoke zn_smoke_k; do
      if [ -d "$REPO/artifacts/runs/$t" ]; then
        say "pulisco il vecchio $t"
        rm -rf "$REPO/artifacts/runs/$t" "$REPO/logs/runs/$t"
      fi
    done
    launch_cell zn_smoke centraltok_fedprior 0 3 "$ZK --fedtokcp-stage1-root $CTFP_ROOT"
    wait   # tag unico ⇒ sequenziale: lo skip di launch.sh ragiona per (tag, serie, arm)
    launch_cell zn_smoke federated_enc_fedavg 3 2 "$ZK --fed-enc-cb union_recluster --fed-enc-prior partial --fed-enc-bn shared"
    wait
    launch_cell zn_smoke_k federated_enc_fedavg 2 2 "$A2K --codebook-size 128"
    wait
    say "— verifiche smoke —"
    ok=1
    grep -l "central tokenizer identity verified" "$REPO"/logs/runs/zn_smoke/*.log >/dev/null 2>&1 \
      && say "  ✓ ctfp: identità tokenizer centrale verificata" \
      || { say "  ✗ ctfp: banner identità MANCANTE"; ok=0; }
    grep -l "UNION-RECLUSTER" "$REPO"/logs/runs/zn_smoke/*.log >/dev/null 2>&1 \
      && say "  ✓ ur: banner union-recluster presente" \
      || { say "  ✗ ur: banner MANCANTE"; ok=0; }
    if ls -d "$REPO"/artifacts/runs/zn_smoke/ckpt/ucr_split_w2p/ucr_170/seed0/*cb-union_recluster* >/dev/null 2>&1; then
      say "  ✓ ur: arm dir cb-union_recluster"
    else say "  ✗ ur: arm dir MANCANTE"; ok=0; fi
    if ls -d "$REPO"/artifacts/runs/zn_smoke_k/ckpt/ucr_split_w2p/ucr_170/seed0/*_K128 >/dev/null 2>&1; then
      say "  ✓ k128: arm dir con bit K128 (vocabolario prior segue K)"
    else say "  ✗ k128: arm dir K128 MANCANTE"; ok=0; fi
    FAILN=$(grep -c "FAILED" "$REPO"/logs/runs/zn_smoke*/_orchestrator.log 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
    [ "${FAILN:-0}" -eq 0 ] && say "  ✓ nessun job FAILED" || { say "  ✗ $FAILN job FAILED"; ok=0; }
    [ "$ok" -eq 1 ] && say "=== SMOKE PASS ===" || { say "=== SMOKE FAIL ==="; exit 5; }
    ;;
  run)
    launch_cell zn_170_ctfp  centraltok_fedprior  300 300 "$ZK --fedtokcp-stage1-root $CTFP_ROOT"
    launch_cell zn_170_ur    federated_enc_fedavg 300 300 "$ZK --fed-enc-cb union_recluster --fed-enc-prior partial --fed-enc-bn shared"
    # ⚠️ ORIZZONTE COSINE = S1_ROUNDS (audit 2026-08-08): con 300 round di orizzonte e lo
    # stop di patience al ginocchio (~r16-23 su 170), la LR sarebbe ancora ~99% del picco —
    # un near-null etichettato come ablazione di schedule. 40 round ⇒ alla finestra della
    # detonazione (r21) la LR è già scesa a ~50%: il trattamento ESISTE. 40 ≈ 2.5× il best
    # round osservato di A2 (r16); se il TRUNCATED alarm scatta, si rilancia con 60.
    launch_cell zn_170_sched federated_enc_fedavg 40  300 "$A2K --fed-enc-sched cosine"
    launch_cell zn_170_k128  federated_enc_fedavg 300 300 "$A2K --codebook-size 128"
    wait ;;
  *) echo "ruolo sconosciuto '$ROLE'" >&2; exit 2;;
esac
say "=== $ROLE finito ==="
