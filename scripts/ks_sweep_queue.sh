#!/usr/bin/env bash
# CODA dello sweep ks — esegue la griglia pre-registrata in
# documentation/PREREG_KS_SWEEP_2026-08-09.md (§3 e §Appendice A).
#
#   bash scripts/ks_sweep_queue.sh cpu  2      # device, worker paralleli
#   bash scripts/ks_sweep_queue.sh cuda 1      # quando GPU1 si libera
#
# RIPARTIBILE: salta ogni (cella, variante) che ha gia' il suo rescore.json, quindi si puo'
# uccidere e rilanciare, e si puo' passare da cpu a cuda a meta' strada.
# ⚠️ Passando da cpu a cuda i residui fp cambiano (max|Δ| 4,1e-02 misurato): il confronto
#    appaiato NON deve attraversare due device. Per questo ogni processo ri-punteggia
#    `paper` INSIEME ai ks nuovi — dentro lo stesso processo, stesso device. Il device
#    finisce in rescore.json e il lettore rifiuta i confronti misti.
# ⚠️ g2: mai GPU0 (ssanchez). Il guardiano e' dentro rescore_ks.py, che rifiuta di partire
#    se CUDA_VISIBLE_DEVICES e' vuoto o contiene 0. Qui si usa scripts/_g2_gpu.sh.
# ⚠️ Il vincolo di g2 e' la CPU (16 core), non la scheda: nice 10, 2 thread per processo.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
DEV="${1:-cpu}"
NPROC="${2:-1}"
# ⚠️ `--threads N` limita SOLO gli intra-op di torch. BLAS, numpy e sklearn aprono i loro
# pool e ignorano quel flag: al primo lancio (2 worker) siamo saliti a 1057% di CPU contro
# il 212% di ssanchez — l'inverso della regola «restare leggeri su g2». Con questi tappi un
# worker costa ~2 core. Il training da solo ne prende gia' ~7 (163 processi).
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
       NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
OUT="$REPO/evidence/ks_sweep"
LOG="$REPO/evidence/ks_sweep_20260809.log"
mkdir -p "$OUT"

say() { echo "[$(date -u '+%F %T') UTC][ks-sweep] $*" | tee -a "$LOG"; }

if [ "$DEV" = "cuda" ]; then
  G="$(bash scripts/_g2_gpu.sh)" || { say "!! guardiano g2 nega la GPU"; exit 4; }
  case " $G " in *" 0 "*) say "!! il guardiano ha restituito GPU0 — mi fermo"; exit 4;; esac
  export CUDA_VISIBLE_DEVICES="$G"
  say "device=cuda su GPU $G"
else
  say "device=cpu, $NPROC worker x 2 thread"
fi

# Ordine: la serie di SCOPERTA per prima (controllo §2bis), poi per costo crescente,
# ucr_222 per ultima (8030 finestre = 35% del costo totale della griglia).
SERIES=(ucr_170 ucr_043 ucr_011 ucr_001 ucr_014 ucr_082 ucr_086 ucr_083 ucr_229 ucr_222)

arm_dir() {  # tag serie -> cartella d'arm
  case "$1" in
    local)       echo "artifacts/runs/zn_main/ckpt/ucr_split_w2p/$2/seed0/local" ;;
    centralized) echo "artifacts/runs/zn_main/ckpt/ucr_split_w2p/$2/seed0/centralized" ;;
    a2)          echo "artifacts/runs/zn_a2/ckpt/ucr_split_w2p/$2/seed0/federated_enc_fedavg_bn-shared_prior-partial" ;;
  esac
}

# Una riga di lavoro = "arm serie client ks_list flag"
JOBS="$(mktemp)"; trap 'rm -f "$JOBS"' EXIT

# ── STAGE 0 — controllo §2bis (separa 'meno contesto' da 'meno smoothing') ────
# Solo sulla serie di scoperta, solo l'arm primario: e' un controllo di MECCANISMO,
# non entra nella statistica delle 9 serie di prova.
for c in 0 1 2 3 4; do
  echo "a2 ucr_170 ucr_170_p$c paper --center-only" >> "$JOBS"
done

# ── STAGE A — confermativo: paper vs ks=1 sui 3 arm, tutte le serie ───────────
for s in "${SERIES[@]}"; do
  for arm in a2 local centralized; do
    for c in 0 1 2 3 4; do echo "$arm $s ${s}_p$c paper,1 -" >> "$JOBS"; done
  done
done

# ── STAGE B — scala di meccanismo: solo arm primario ─────────────────────────
for s in "${SERIES[@]}"; do
  for c in 0 1 2 3 4; do echo "a2 $s ${s}_p$c 3,5,9,13 -" >> "$JOBS"; done
done

say "$(wc -l < "$JOBS") righe in coda (stage 0 + A + B)"

run_one() {
  read -r arm s client kslist flag <<< "$1"
  local d; d="$(arm_dir "$arm" "$s")/$client"
  [ -f "$d/stage2.ckpt" ] || { say "manca $d/stage2.ckpt — salto"; return 0; }
  local tgt="$OUT/$arm/$s"
  # gia' fatto? (il primo ks della lista basta come sentinella: rescore_ks scrive in ordine)
  local first="${kslist%%,*}"; local sfx=""; [ "$flag" = "--center-only" ] && sfx="_c"
  local probe="$tgt/${client}__${first}${sfx}"
  [ "$first" != "paper" ] && probe="$tgt/${client}__ks${first}${sfx}"
  [ -f "$probe/rescore.json" ] && return 0
  mkdir -p "$tgt"
  local extra=(); [ "$flag" = "--center-only" ] && extra=(--center-only)
  nice -n 10 "$PY" scripts/rescore_ks.py --client-dir "$d" --ks "$kslist" \
      --out "$tgt" --device "$DEV" --threads 2 "${extra[@]}" >>"$LOG" 2>&1 \
    || say "!! FALLITA: $arm/$s/$client ks=$kslist $flag"
}
export -f run_one arm_dir say
export PY DEV OUT LOG REPO

if [ "$NPROC" -gt 1 ] && [ "$DEV" = "cpu" ]; then
  xargs -a "$JOBS" -d '\n' -I{} -P "$NPROC" bash -c 'run_one "$@"' _ {}
else
  while IFS= read -r line; do run_one "$line"; done < "$JOBS"
fi

say "=== coda finita: $(find "$OUT" -name rescore.json | wc -l) risultati in $OUT ==="
