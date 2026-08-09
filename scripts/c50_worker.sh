#!/usr/bin/env bash
# c50_worker.sh — corsie della campagna C50 su UNA GPU di g2.
#
#   bash scripts/c50_worker.sh <GPU> <N_CORSIE>      # es.: 0 3   oppure   1 2
#
# DEROGA GPU (istruzione diretta dell'utente, 2026-08-09): la campagna c50 usa ANCHE
# GPU0 mentre ssanchez ci lavora — coabitazione voluta, non svista. Per tutto ciò che
# non è c50 resta valida la regola standard di scripts/_g2_gpu.sh.
#
# CONTROLLI (mai uccidere una cella a metà):
#   touch evidence/c50_STOP           → ogni corsia finisce la cella in corso ed esce
#   touch evidence/c50_STOP_gpu0      → solo le corsie della GPU0 (idem _gpu1)
#   editare evidence/c50_queue_20260809.txt  → togliere/riordinare le celle future
#                                       (il pop è atomico via flock: editare è sicuro)
#   ri-routare:       touch evidence/c50_STOP_gpu0 && bash scripts/c50_worker.sh 1 3
#   più parallelo:    rilanciare il worker sulla stessa GPU (le corsie si sommano)
#   stato:            bash scripts/c50_status.sh
#
# Riga di coda: TAG|SERIE|ARM|S1|S2|EXTRA → una cella launch.sh. launch.sh è idempotente
# (out-json esistente ⇒ cella saltata), quindi rimettere righe in coda è sempre gratis.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
GPU="${1:?gpu id (0|1)}"; LANES="${2:-2}"
QUEUE="$REPO/evidence/c50_queue_20260809.txt"
LOGD="$REPO/logs/c50"; mkdir -p "$LOGD"
[ -f "$QUEUE" ] || { echo "coda mancante: $QUEUE" >&2; exit 2; }

export NO_MPS=1 FEDVQ_AMP=fp16 SLOTS_PER_GPU=1
# Evita il server MPS residente (pipe dir inesistente ⇒ contesti CUDA diretti).
export CUDA_MPS_PIPE_DIRECTORY="$REPO/.no_mps_pipe_inesistente"

say() { echo "[$(date -u '+%F %T') UTC][c50/gpu$GPU] $*"; }

pop_cell() {
  ( flock -x 200
    line=$(head -n1 "$QUEUE" 2>/dev/null)
    [ -n "$line" ] && sed -i '1d' "$QUEUE"
    echo "$line"
  ) 200>"$QUEUE.lock"
}

corsia() {
  local id="$1" line tag series arm s1 s2 extra
  while :; do
    [ -f "$REPO/evidence/c50_STOP" ]         && { say "[corsia $id] STOP globale — esco"; break; }
    [ -f "$REPO/evidence/c50_STOP_gpu$GPU" ] && { say "[corsia $id] STOP gpu$GPU — esco"; break; }
    line="$(pop_cell)"; [ -z "$line" ] && { say "[corsia $id] coda vuota — esco"; break; }
    IFS='|' read -r tag series arm s1 s2 extra <<<"$line"
    say "[corsia $id] >>> $tag / $series / $arm"
    S1_ROUNDS="$s1" S2_ROUNDS="$s2" PATIENCE=6 LAUNCH_GPUS="$GPU" LAUNCH_ONLY_CLUSTERS="$series" \
      bash scripts/launch.sh --cohort c50 --tag "$tag" --arms "$arm" --extra "$extra" \
      >>"$LOGD/gpu${GPU}_corsia${id}.log" 2>&1
    say "[corsia $id] <<< $tag / $series rc=$?"
  done
}

say "worker su GPU$GPU: $LANES corsie (coda: $(wc -l <"$QUEUE") celle)"
for _l in $(seq 1 "$LANES"); do
  corsia "$_l" &
  echo $! > "$LOGD/gpu${GPU}_corsia${_l}.pid"
  sleep 90   # sfalsa i picchi CPU della costruzione dei loader
done
wait
say "worker GPU$GPU: tutte le corsie chiuse"
