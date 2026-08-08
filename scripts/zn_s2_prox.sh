#!/usr/bin/env bash
# FedProx-SUL-PRIOR (riga di completezza) — 2026-08-08.
#
#   bash scripts/zn_s2_prox.sh smoke   # 3 round su ucr_014, tag zn_smoke, verifica automatica
#   bash scripts/zn_s2_prox.sh queue   # ASPETTA la fine di tau/tauext/ctrl, poi 20 celle
#
# PERCHE' GIRA PUR ESSENDO ATTESO ~ctrl: e' la riga «avete provato FedProx sul prior?» del
# paper, DICHIARATA come completezza — l'agg_penalty misurata dal ctrl e' NEGATIVA a fine
# training (la media dei body e' un bonus, non un danno), quindi la predizione pre-registrata
# e' pmu ≈ ctrl. Se invece vincesse, tanto meglio: entra in tabella principale.
#
# DISEGNO (documentation/A2_STAGE2_UPGRADE_DESIGN.md §3+§7):
#   * SOLO forma decoupled: w ← w + lr·μ·(w^t − w) dopo lo step, fuori dal precondizionatore
#     di AdamW (la loss-form a questi μ e' un no-op silenzioso). Diagnostico: prox_pull_frac.
#   * μ ∈ {1, 3}: pull ≈13% / ≈34% del drift per round a lr=1e-3. μ=10 CANCELLATO (§7:
#     94,6% di pull = un τ-piccolo travestito).
#   * regime EPOCH, ricetta identica al ctrl (resume A2 + oracolo fisso): il contrasto
#     pmu−ctrl isola il solo termine prossimale.
#
# CODA A SCIA: parte quando (a) NESSUNA cella tau/ctrl e' in volo E (b) nessun processo
# zn_s2_wave.sh e' vivo — la (b) evita la collisione con tauext, che a sua volta parte a
# scia quando le celle scendono sotto 6: senza (b) potremmo fare fuoco mentre lui sta per
# lanciarne 18.
#
# ⚠️ S2_ROUNDS SEMPRE ESPLICITO (con S1_ROUNDS=0 il default di launch.sh sarebbe 0).
# ⚠️ un tag per variante (out-json cieco ai knob) · resume PER SERIE ⇒ un launch.sh per
#    (tag, serie), GPU assegnata qui round-robin.
# ⚠️ zn_smoke e' FUORI dal manifesto e va ripulito PRIMA di rilanciare: lo skip di launch.sh
#    ragiona per (tag, serie, arm-name) e il vecchio smoke del ctrl occuperebbe la cella.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
ROLE="${1:?smoke | queue}"
ALL=(ucr_001 ucr_011 ucr_014 ucr_043 ucr_082 ucr_083 ucr_086 ucr_170 ucr_222 ucr_229)
A2K='--window-normalization zscore --fed-enc-cb suffstat --fed-enc-prior partial --fed-enc-bn shared'
A2ROOT="$REPO/artifacts/runs/zn_a2/ckpt"
say() { echo "[$(date -u '+%F %T') UTC | $(TZ=Europe/Madrid date '+%H:%M') Madrid][s2-prox/$ROLE] $*"; }

"$PY" scripts/zn_owner.py --check >/dev/null || { say "!! partizione ROTTA — non parto"; exit 2; }

# GPU senza processi di ALTRI utenti (deroga utente 2026-08-08 per 0/5 se vuote).
# In `queue` si chiama DOPO l'attesa: il censimento fatto ore prima sarebbe stantio.
compute_free() {
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
    say "deroga utente 2026-08-08: G4_ALLOW_RESERVED=1 (0/5 vuote incluse)";; esac
  bash "$REPO/scripts/_g4_gpu.sh" "${FREE[*]}" || exit 4
  say "GPU disponibili: [${FREE[*]}]"
}
export NO_MPS=1 FEDVQ_AMP=fp16 SLOTS_PER_GPU=1

gidx=0
launch_cell() {  # tag serie mu
  local tag="$1" s="$2" mu="$3"
  # ⚠️ niente command substitution: l'indice si tocca solo nel processo corrente
  # (il bug del primo lancio della wave: g=$(next_gpu) incrementava in una subshell).
  local g="${FREE[$(( gidx % ${#FREE[@]} ))]}"; gidx=$((gidx+1))
  say "=== $tag $s mu=$mu -> GPU$g (S2_ROUNDS=300) ==="
  S1_ROUNDS=0 S2_ROUNDS=300 PATIENCE=6 LAUNCH_GPUS="$g" LAUNCH_ONLY_CLUSTERS="$s" \
    bash scripts/launch.sh --cohort ucr2p_10 --tag "$tag" --arms federated_enc_fedavg \
    --extra "$A2K --fed-s2-val fixed --fed-prior-prox-mu $mu --fed-prior-prox-form decoupled --resume-from $A2ROOT/ucr_split_w2p/$s/seed0/federated_enc_fedavg_bn-shared_prior-partial" &
  sleep 6
}

case "$ROLE" in
  smoke)
    # Pulizia del vecchio zn_smoke (tag usa-e-getta, fuori dal manifesto per contratto).
    if [ -d "$REPO/artifacts/runs/zn_smoke" ]; then
      say "pulisco il vecchio zn_smoke (lo skip per arm-name occuperebbe la cella)"
      rm -rf "$REPO/artifacts/runs/zn_smoke" "$REPO/logs/runs/zn_smoke"
    fi
    compute_free
    g="${FREE[0]}"
    say "=== smoke: 3 round, ucr_014, mu=1 decoupled -> GPU$g ==="
    S1_ROUNDS=0 S2_ROUNDS=3 PATIENCE=6 LAUNCH_GPUS="$g" LAUNCH_ONLY_CLUSTERS=ucr_014 \
      bash scripts/launch.sh --cohort ucr2p_10 --tag zn_smoke --arms federated_enc_fedavg \
      --extra "$A2K --fed-s2-val fixed --fed-prior-prox-mu 1 --fed-prior-prox-form decoupled --resume-from $A2ROOT/ucr_split_w2p/ucr_014/seed0/federated_enc_fedavg_bn-shared_prior-partial"
    rc=$?
    say "launch.sh rc=$rc — verifiche:"
    # il log del JOB, non l'_orchestrator.log (che matcherebbe *.log per primo)
    LOG="$(grep -l "FedProx-on-prior" "$REPO"/logs/runs/zn_smoke/*.log 2>/dev/null | head -1)"
    ok=1
    if [ -n "$LOG" ] && grep -q "FedProx-on-prior: mu=1" "$LOG"; then
      say "  ✓ banner FedProx-on-prior presente"
      grep -m1 "decoupled contraction" "$LOG" | sed 's/^/    /'
      grep -m2 "prox_dist" "$LOG" | sed 's/^/    /'
    else say "  ✗ banner MANCANTE (log: ${LOG:-nessuno})"; ok=0; fi
    D="$REPO/artifacts/runs/zn_smoke/ckpt/ucr_split_w2p/ucr_014/seed0"
    if ls -d "$D"/*pmu1_dec* >/dev/null 2>&1; then
      say "  ✓ arm dir con bit pmu1_dec: $(basename "$(ls -d "$D"/*pmu1_dec* | head -1)")"
    else say "  ✗ arm dir pmu1_dec MANCANTE in $D"; ok=0; fi
    FH="$(ls "$D"/*pmu1_dec*/fed_history.json 2>/dev/null | head -1)"
    if [ -n "$FH" ] && "$PY" -c "
import json,sys,math
h=json.load(open('$FH')); s2=h.get('stage2',[])
rows=[r for r in s2 if 'prox_pull_frac' in r and math.isfinite(r.get('prox_dist',float('nan')))]
sys.exit(0 if len(rows)==len(s2)>0 else 1)"; then
      say "  ✓ fed_history.stage2 ha prox_dist/prox_pull_frac su ogni round"
    else say "  ✗ telemetria prox assente da fed_history ($FH)"; ok=0; fi
    [ "$ok" -eq 1 ] && say "=== SMOKE PASS ===" || { say "=== SMOKE FAIL ==="; exit 5; }
    ;;
  queue)
    say "in attesa della fine di tau/tauext/ctrl (celle=0 E nessun zn_s2_wave.sh vivo)…"
    while :; do
      NRUN=$(ps -eo args -u "$(whoami)" \
             | grep -oE "runs/zn_a2s2_(tau64|tau16|t64adam|ctrl)/ucr_split_w2p/ucr_[0-9]+__[a-z_]+" \
             | sort -u | wc -l)
      NWAVE=$(pgrep -u "$(whoami)" -f "zn_s2_wave\.sh" | wc -l)
      [ "$NRUN" -eq 0 ] && [ "$NWAVE" -eq 0 ] && break
      say "in coda: $NRUN celle in volo, $NWAVE processi wave vivi — riprovo fra 15 min"
      sleep 900
    done
    say "scia libera: parte la wave prox (2 tag × 6 serie REST = 12 celle)"
    # ⚠️ RE-ROUTE 2026-08-08 (morte di g4 alle 03:14): le 4 serie PROBE sono passate a
    # zn_g2_requeue.sh su g2 — qui restano SOLO le 6 REST, come da zn_ownership.json.
    REST=(ucr_001 ucr_082 ucr_083 ucr_086 ucr_222 ucr_229)
    compute_free
    for s in "${REST[@]}"; do
      launch_cell zn_a2s2_pmu1 "$s" 1
      launch_cell zn_a2s2_pmu3 "$s" 3
    done
    wait ;;
  *) echo "ruolo sconosciuto '$ROLE'" >&2; exit 2;;
esac
say "=== $ROLE finito ==="
