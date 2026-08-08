#!/usr/bin/env bash
# RE-ROUTE SU g2 — 2026-08-08. g4 è morto alle 03:14 UTC (kernel wedge: ARP sì, ICMP/ssh
# no); questa coda porta su g2 la TESTA del backlog, ordinata per valore scientifico.
# Ownership: zn_ownership.json (owner zn_g2_requeue.sh); a g4 restano tauext + prox REST.
#
#   bash scripts/zn_g2_requeue.sh all     # smoke (3 percorsi nuovi) -> coda a 2 corsie
#   bash scripts/zn_g2_requeue.sh smoke   # solo gli smoke
#   bash scripts/zn_g2_requeue.sh queue   # solo la coda (se lo smoke è già passato)
#
# VINCOLI g2 (indicazione utente 2026-08-08 + regole 2026-08-06):
#  * SOLO GPU1 — ssanchez lavora su GPU0: anche se la scheda "sopporterebbe", non si tocca.
#  * LANES celle in parallelo (default 3 — misurato 2026-08-08: 1 cella ≈ 1,3 core, e 3 è
#    il ginocchio di throughput per scheda; oltre non si guadagna, misurato 0,93× su g4).
#  * il nvidia-cuda-mps-server nostro (vivo da 9gg) NON si uccide: il job di ssanchez
#    potrebbe passarci attraverso. I NOSTRI job lo EVITANO puntando CUDA_MPS_PIPE_DIRECTORY
#    a una dir inesistente (senza daemon raggiungibile, CUDA crea contesti diretti).
#  * se un ALTRO utente compare su GPU1: la corsia ASPETTA (non si salta su GPU0).
#
# ORDINE (misto, valore prima — motivazioni in zn_ownership.json):
#   A: fix-170  ctfp -> ur -> sched(orizzonte 40) -> k128
#   B: prox probe  pmu1/011, pmu1/043, pmu3/011, pmu3/043, pmu1/014, pmu1/170, pmu3/014, pmu3/170
#   C: code ctrl   082, 222
#
# ⚠️ CONFONDENTE DICHIARATO: queste celle girano su g2 (RTX 8000, sm_75) mentre le baseline
# di confronto sono nate su g4 — confondente misurato innocuo (soglie di rumore 2026-07-31),
# ma su ucr_170 va citato accanto a ogni contrasto.
# ⚠️ S2_ROUNDS sempre esplicito · un tag per variante · fp16 (bf16 su sm_75 è una pessimizzazione).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
ROLE="${1:?all | smoke | queue}"
ZK='--window-normalization zscore --fed-s2-val fixed'
A2K="$ZK --fed-enc-cb suffstat --fed-enc-prior partial --fed-enc-bn shared"
A2ROOT="$REPO/artifacts/runs/zn_a2/ckpt"
CTFP_ROOT="$REPO/artifacts/runs/zn_main/ckpt"
QUEUE="$REPO/evidence/zn_g2_queue_20260808.txt"
say() { echo "[$(date -u '+%F %T') UTC][g2-requeue/$ROLE] $*"; }

"$PY" scripts/zn_owner.py --check >/dev/null || { say "!! partizione ROTTA — non parto"; exit 2; }
[[ "$(hostname -s)" == g2* ]] || { say "!! questo script gira su g2 (hostname: $(hostname -s))"; exit 2; }

export NO_MPS=1 FEDVQ_AMP=fp16 SLOTS_PER_GPU=1
# Evita il server MPS residente (vedi intestazione): pipe dir inesistente ⇒ contesti diretti.
export CUDA_MPS_PIPE_DIRECTORY="$REPO/.no_mps_pipe_inesistente"

# GPU1 deve essere libera da processi di ALTRI utenti (il nostro mps-server non conta:
# 26 MiB, nessun kernel — e comunque non è un job). Ritorna 0 se libera.
gpu1_libera() {
  local u1 p o
  u1=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1=="1"{print $2}')
  for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
             | awk -F', ' -v x="$u1" '$1==x{print $2}'); do
    o="$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')"
    [ -n "$o" ] && [ "$o" != "$(whoami)" ] && return 1
  done
  return 0
}
aspetta_gpu1() {
  while ! gpu1_libera; do
    say "GPU1 ha processi di ALTRI utenti — aspetto 5 min (non si salta su GPU0)"
    sleep 300
  done
}

run_cell() {  # tag series arm s1 s2 extra — FOREGROUND (la corsia è la concorrenza)
  local tag="$1" series="$2" arm="$3" s1="$4" s2="$5" extra="$6"
  aspetta_gpu1
  say ">>> $tag / $series ($arm) S1=$s1 S2=$s2"
  S1_ROUNDS="$s1" S2_ROUNDS="$s2" PATIENCE=6 LAUNCH_GPUS=1 LAUNCH_ONLY_CLUSTERS="$series" \
    bash scripts/launch.sh --cohort ucr2p_10 --tag "$tag" --arms "$arm" --extra "$extra"
  say "<<< $tag / $series rc=$?"
}

case "$ROLE" in
  smoke|all)
    for t in zn_smoke zn_smoke_k; do
      [ -d "$REPO/artifacts/runs/$t" ] && { say "pulisco il vecchio $t"; rm -rf "$REPO/artifacts/runs/$t" "$REPO/logs/runs/$t"; }
    done
    run_cell zn_smoke   ucr_170 centraltok_fedprior  0 3 "$ZK --fedtokcp-stage1-root $CTFP_ROOT"
    run_cell zn_smoke   ucr_170 federated_enc_fedavg 3 2 "$ZK --fed-enc-cb union_recluster --fed-enc-prior partial --fed-enc-bn shared"
    run_cell zn_smoke_k ucr_170 federated_enc_fedavg 2 2 "$A2K --codebook-size 128"
    say "— verifiche smoke —"; ok=1
    grep -l "central tokenizer identity verified" "$REPO"/logs/runs/zn_smoke/*.log >/dev/null 2>&1 \
      && say "  ✓ ctfp: identità tokenizer centrale verificata" || { say "  ✗ ctfp: banner identità MANCANTE"; ok=0; }
    grep -l "UNION-RECLUSTER" "$REPO"/logs/runs/zn_smoke/*.log >/dev/null 2>&1 \
      && say "  ✓ ur: banner union-recluster" || { say "  ✗ ur: banner MANCANTE"; ok=0; }
    ls -d "$REPO"/artifacts/runs/zn_smoke/ckpt/ucr_split_w2p/ucr_170/seed0/*cb-union_recluster* >/dev/null 2>&1 \
      && say "  ✓ ur: arm dir cb-union_recluster" || { say "  ✗ ur: arm dir MANCANTE"; ok=0; }
    ls -d "$REPO"/artifacts/runs/zn_smoke_k/ckpt/ucr_split_w2p/ucr_170/seed0/*_K128 >/dev/null 2>&1 \
      && say "  ✓ k128: arm dir K128" || { say "  ✗ k128: arm dir K128 MANCANTE"; ok=0; }
    if [ "$ok" -ne 1 ]; then say "=== SMOKE FAIL — la coda NON parte ==="; exit 5; fi
    say "=== SMOKE PASS ==="
    [ "$ROLE" = "smoke" ] && { say "=== smoke finito ==="; exit 0; }
    ;;&
  queue|all)
    # La coda vive su file: una riga = una cella; le corsie fanno pop atomico via flock.
    # launch.sh salta da solo le celle già a disco (out-json), quindi rilanciare è idempotente.
    cat > "$QUEUE" <<EOQ
zn_170_ctfp|ucr_170|centraltok_fedprior|300|300|$ZK --fedtokcp-stage1-root $CTFP_ROOT
zn_170_ur|ucr_170|federated_enc_fedavg|300|300|$ZK --fed-enc-cb union_recluster --fed-enc-prior partial --fed-enc-bn shared
zn_170_sched|ucr_170|federated_enc_fedavg|40|300|$A2K --fed-enc-sched cosine
zn_170_k128|ucr_170|federated_enc_fedavg|300|300|$A2K --codebook-size 128
zn_a2s2_pmu1|ucr_011|federated_enc_fedavg|0|300|$A2K --fed-prior-prox-mu 1 --fed-prior-prox-form decoupled --resume-from $A2ROOT/ucr_split_w2p/ucr_011/seed0/federated_enc_fedavg_bn-shared_prior-partial
zn_a2s2_pmu1|ucr_043|federated_enc_fedavg|0|300|$A2K --fed-prior-prox-mu 1 --fed-prior-prox-form decoupled --resume-from $A2ROOT/ucr_split_w2p/ucr_043/seed0/federated_enc_fedavg_bn-shared_prior-partial
zn_a2s2_pmu3|ucr_011|federated_enc_fedavg|0|300|$A2K --fed-prior-prox-mu 3 --fed-prior-prox-form decoupled --resume-from $A2ROOT/ucr_split_w2p/ucr_011/seed0/federated_enc_fedavg_bn-shared_prior-partial
zn_a2s2_pmu3|ucr_043|federated_enc_fedavg|0|300|$A2K --fed-prior-prox-mu 3 --fed-prior-prox-form decoupled --resume-from $A2ROOT/ucr_split_w2p/ucr_043/seed0/federated_enc_fedavg_bn-shared_prior-partial
zn_a2s2_pmu1|ucr_014|federated_enc_fedavg|0|300|$A2K --fed-prior-prox-mu 1 --fed-prior-prox-form decoupled --resume-from $A2ROOT/ucr_split_w2p/ucr_014/seed0/federated_enc_fedavg_bn-shared_prior-partial
zn_a2s2_pmu1|ucr_170|federated_enc_fedavg|0|300|$A2K --fed-prior-prox-mu 1 --fed-prior-prox-form decoupled --resume-from $A2ROOT/ucr_split_w2p/ucr_170/seed0/federated_enc_fedavg_bn-shared_prior-partial
zn_a2s2_pmu3|ucr_014|federated_enc_fedavg|0|300|$A2K --fed-prior-prox-mu 3 --fed-prior-prox-form decoupled --resume-from $A2ROOT/ucr_split_w2p/ucr_014/seed0/federated_enc_fedavg_bn-shared_prior-partial
zn_a2s2_pmu3|ucr_170|federated_enc_fedavg|0|300|$A2K --fed-prior-prox-mu 3 --fed-prior-prox-form decoupled --resume-from $A2ROOT/ucr_split_w2p/ucr_170/seed0/federated_enc_fedavg_bn-shared_prior-partial
zn_a2s2_ctrl|ucr_082|federated_enc_fedavg|0|300|$A2K --fed-s2-agg-penalty --fed-s2-snapshot-every 4 --resume-from $A2ROOT/ucr_split_w2p/ucr_082/seed0/federated_enc_fedavg_bn-shared_prior-partial
zn_a2s2_ctrl|ucr_222|federated_enc_fedavg|0|300|$A2K --fed-s2-agg-penalty --fed-s2-snapshot-every 4 --resume-from $A2ROOT/ucr_split_w2p/ucr_222/seed0/federated_enc_fedavg_bn-shared_prior-partial
EOQ
    say "coda scritta: $(wc -l < "$QUEUE") celle -> $QUEUE"
    pop_cell() {
      ( flock -x 200
        local line; line=$(head -n1 "$QUEUE" 2>/dev/null)
        [ -n "$line" ] && sed -i '1d' "$QUEUE"
        echo "$line"
      ) 200>"$QUEUE.lock"
    }
    corsia() {
      local id="$1" line
      while :; do
        line="$(pop_cell)"; [ -z "$line" ] && break
        IFS='|' read -r tag series arm s1 s2 extra <<<"$line"
        say "[corsia $id] prendo $tag/$series"
        run_cell "$tag" "$series" "$arm" "$s1" "$s2" "$extra"
      done
      say "[corsia $id] coda vuota, esco"
    }
    LANES="${LANES:-3}"
    say "parto con $LANES corsie"
    for _l in $(seq 1 "$LANES"); do corsia "$_l" & sleep 90; done
    # lo sleep 90 sfalsa le corsie: tre stage-1 che partono nello stesso istante
    # picchiano insieme su disco/CPU per la costruzione dei loader
    wait
    say "=== CODA COMPLETA — restano a g4: tauext (18) + prox REST (12) ==="
    ;;
  *) echo "ruolo sconosciuto '$ROLE'" >&2; exit 2;;
esac
say "=== $ROLE finito ==="
