#!/usr/bin/env bash
# run_ucrsplit_w2p.sh — the SAME 6 arms as run_ucrsplit.sh, but with the PAPER'S WINDOW.
#
# TimeVQVAE-AD (arXiv 2311.12550v5) Algorithm 1 derives the rolling window from the series'
# period -- "Define a period length P of x*", then "x in R^T, T = 2P". It is NOT a fixed
# hyperparameter. config.py pins window_length=128 for every dataset (the period override
# was removed for wsd_fed, where 2P=2880 would have cut 31 clients to 18). On UCR that
# deletes the paper's protocol, and the damage is measured:
#
#   cycles seen by our window (128/period), median over the archive : 0.77
#   cycles the paper prescribes                                     : 2.00
#   series (of 250) where 128 is the right value                    : ZERO
#   too narrow (2P > 128): 192/250 (77%)   too wide (2P < 128): 58/250 (23%)
#
# This script is the A/B twin of run_ucrsplit.sh: identical arms, identical
# hyperparameters, identical scheduler, ONE variable changed. See
# documentation/UCR_SPLIT_OPEN_QUESTIONS.md point 1 and documentation/UCR_ALL_SERIES_WINDOWS.md.
#
#   1 local          2 centralized     3 federated_enc_fedavg
#   4 federated_enc_fedprox            5 federated_enc_fedproto
#   6 federated_enc_commoninit  (the null -- arms 3-5 are uninterpretable without it)
#
# ── THE TWO THINGS THAT MAKE THIS A CLEAN A/B ────────────────────────────────────────
#
# (a) --metrics-tolerance 64 is passed ALWAYS and is not optional. The VUS/PATE buffer and
#     the top-k hit radius default to window_length//2, so without the pin a 2P run would
#     be judged with a yardstick up to 1514 samples wide while the 128 run uses 64 -- the
#     window would move both the model AND the metric, and nothing could be attributed.
#
# (b) A separate RESDIR/CKPT root. federated_eval keys checkpoints on --out-dir, not on
#     config.run_name, so pointing this at artifacts/ucrsplit would OVERWRITE the 128 run's
#     checkpoints with incompatible-shape ones. (config.run_name now also carries a `_w<N>`
#     suffix -- that protects the run.py entry points, not this one.)
#
# ── SCOPE ────────────────────────────────────────────────────────────────────────────
# 76 of the 82 clusters. A larger window eats the short clients: 6 clusters would leave
# their 10% client with fewer than MIN_WIN=32 stride-1 windows (four of them go NEGATIVE,
# i.e. no window at all). They are listed at startup and skipped -- never silently.
#
#   DRYRUN=1 bash scripts/run_ucrsplit_w2p.sh              # print the plan, run nothing
#   SUBSET=8 DRYRUN=1 bash scripts/run_ucrsplit_w2p.sh     # the cheap 8-cluster probe first
#   setsid nohup bash scripts/run_ucrsplit_w2p.sh > logs/ucrsplit_w2p_boot.log 2>&1 </dev/null &
#
# ⚠ Do NOT launch this while run_ucrsplit.sh is still going unless you mean to halve both
#   runs' throughput -- they share the same 14 slots and the same 16-core CPU wall.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PY:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10}"
export PIPELINE_PYTHON="$PY"

# MPS on the PRIVATE pipe dir. Without this the workers attach to the compiled-in default
# /tmp/nvidia-mps, which is created 0777 with a 0666 control socket: any user on the box can
# `quit_server` it, and on 2026-07-28 a server death there wedged 14 jobs on a socket poll
# for 9.9 h with no error and no log. Env only -- this never starts or stops the daemon
# (`bash scripts/mps_ctl.sh up` does that, and must be run first).
MPS_CTL_ENV_ONLY=1 source "$REPO/scripts/mps_ctl.sh"

# ── THE 2x2: window x codebook ───────────────────────────────────────────────────────
# The paper differs from our config in TWO places, and they must be separable:
#
#            cb=64 (our default)          cb=128 (the paper's)
#   W=128    A  run_ucrsplit.sh  ✅        C  WINDOW_MODE=fixed CODEBOOK=128
#   W=2P     B  (this, default)            D  CODEBOOK=128        <- full replication
#
# Cell A is the already-running baseline and keeps its own pristine script. B isolates the
# window, C isolates the codebook, D is the paper. Running D alone would move both knobs
# and attribute nothing -- C exists so the codebook term can be read off on its own.
WINDOW_MODE="${WINDOW_MODE:-2p}"     # 2p = per-cluster 2*period ; fixed = leave config's 128
CODEBOOK="${CODEBOOK:-0}"            # 0 = config default (64) ; 128 = the paper's
case "$WINDOW_MODE" in 2p|fixed) ;; *) echo "WINDOW_MODE must be 2p|fixed"; exit 1 ;; esac
TAG="w${WINDOW_MODE}_cb$([[ "$CODEBOOK" == 0 ]] && echo 64 || echo "$CODEBOOK")"

LOGDIR="$REPO/logs/ucrsplit_$TAG"; RESDIR="$REPO/artifacts/ucrsplit_$TAG"
CKPT="$RESDIR/ckpt"
mkdir -p "$LOGDIR" "$RESDIR"
ORCH="$LOGDIR/_orchestrator.log"

GPUS=(0 1); SLOTS_PER_GPU="${SLOTS_PER_GPU:-7}"      # measured optimum under MPS; see the notes
SEED="${SEED:-0}"
# Seed suffix. EMPTY at seed 0 so existing artifacts keep their names and the skip-if-exists
# guard still recognises them; non-empty otherwise. Without this the filename carries no seed,
# so `SEED=1 bash <this script>` matches the seed-0 jsons and SKIPS every job — a silent no-op
# that looks like a completed replication.
SFX=""; [[ "$SEED" != "0" ]] && SFX="_s$SEED"
S1_ROUNDS="${S1_ROUNDS:-300}"; LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"; PATIENCE="${PATIENCE:-6}"
BATCH="${BATCH:-64}"
FEDPROX_MU="${FEDPROX_MU:-0.1}"; FEDPROX_FORM="${FEDPROX_FORM:-loss}"
FEDPROTO_LAM="${FEDPROTO_LAM:-0.1}"; FEDPROTO_AGG="${FEDPROTO_AGG:-count}"
TOL="${TOL:-64}"                                     # PINNED. see (a) above.
MIN_WIN="${MIN_WIN:-32}"                             # min stride-1 windows for the 10% client at 2P
CLUSTER_FILE="${CLUSTER_FILE:-$REPO/scripts/ucr_split_clusters.txt}"
SUBSET="${SUBSET:-0}"                                # >0 = keep only the N cheapest clusters

stamp() { date +'%F %T'; }
say() { echo "[$(stamp)] $*" | tee -a "$ORCH"; }

[[ -f "$CLUSTER_FILE" ]] || { echo "missing $CLUSTER_FILE"; exit 1; }

# ── per-cluster window = 2*period, eligibility, and LPT cost order ───────────────────
# Emits "cluster window cost" per line, most expensive first. Everything the shell needs
# to know about periods is computed here, once, from the archive's own period table.
mapfile -t PLAN < <("$PY" - "$CLUSTER_FILE" "$MIN_WIN" "$SUBSET" "$WINDOW_MODE" <<'PYEOF'
import csv, json, math, sys
from collections import defaultdict
cluster_file, min_win, subset = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
window_mode = sys.argv[4]
per = list(csv.DictReader(open("preprocessing/UCR_anomaly_dataset_periods.csv")))
k0 = list(per[0])[0]
P = {f"ucr_{int(r[k0]):03d}": int(r["period"]) for r in per if r["period"].strip()}
md = json.load(open("data/raw/ucr_split/metadata.json"))
SH = defaultdict(dict)
TEST = {}
for e in md["entities"]:
    SH[e["series"]][e["partition"]] = e["train_length"]
    TEST[e["series"]] = e["test_length"]
keep = [l.strip() for l in open(cluster_file) if l.strip()]
BATCH, LE, R, SPS, DPS, S2F = 64, 10, 38, 15.0, 60.0, 6000
rows, skipped = [], []
for c in keep:
    # WINDOW_MODE=fixed keeps config's 128 -> every cluster is eligible (that is cell A's
    # scope, and cell C must match it exactly or the codebook term is confounded with a
    # different cluster set).
    W = 2 * P[c] if window_mode == "2p" else 128
    wins = [SH[c][i] - W + 1 for i in range(5)]
    if window_mode == "2p" and min(wins) < min_win:
        skipped.append((c, W, min(wins)))
        continue
    stride = max(1, round(0.1 * W))
    spe = sum(math.ceil(max(0, w) / BATCH) for w in wins)
    det = 5 * max(0, (TEST[c] - W) // stride + 1)
    cost = (R * LE * spe + 5 * S2F) / SPS / 3600 + det / DPS / 3600
    rows.append((c, W, cost))
rows.sort(key=lambda r: -r[2])
if subset > 0:
    rows = sorted(rows, key=lambda r: r[2])[:subset]      # cheapest N for a quick probe
    rows.sort(key=lambda r: -r[2])
for c, W, _, in [(a, b, 0) for a, b, _ in skipped]:
    pass
print(f"#SKIPPED {len(skipped)} " + " ".join(f"{c}:2P={W}:minwin={m}" for c, W, m in skipped))
print(f"#TOTALCOST {sum(r[2] for r in rows):.1f}")
for c, W, cost in rows:
    print(f"{c} {W} {cost:.3f}")
PYEOF
)

SKIPLINE=""; TOTALCOST=""
JOBS=(); CLUSTERS=()
declare -A WIN
for line in "${PLAN[@]}"; do
  case "$line" in
    "#SKIPPED "*) SKIPLINE="${line#\#SKIPPED }" ;;
    "#TOTALCOST "*) TOTALCOST="${line#\#TOTALCOST }" ;;
    *) read -r c w _cost <<<"$line"; CLUSTERS+=("$c"); WIN["$c"]="$w" ;;
  esac
done
[[ ${#CLUSTERS[@]} -gt 0 ]] || { echo "no eligible clusters"; exit 1; }

COMMON="--protocol converged --fed-enc-prior local --s1-rounds $S1_ROUNDS \
--local-epochs $LOCAL_EPOCHS --fed-patience-rounds $PATIENCE --seeds $SEED --batch $BATCH \
--metrics-tolerance $TOL"
# 0 is federated_eval's "leave the config default" sentinel, so only pass it when set.
[[ "$CODEBOOK" != 0 ]] && COMMON="$COMMON --codebook-size $CODEBOOK"

declare -A ARM=(
  [local]="local|"
  [centralized]="centralized|"
  [fedavg]="federated_enc_fedavg|"
  [fedprox]="federated_enc_fedprox|--fedprox-mu $FEDPROX_MU --fedprox-form $FEDPROX_FORM"
  [fedproto]="federated_enc_fedproto|--fedproto-weight $FEDPROTO_LAM --fedproto-agg $FEDPROTO_AGG"
  [commoninit]="federated_enc_commoninit_cbshared|"
  # ── cb-LOCAL twins: ADDITIONS, never replacements ──────────────────────────────────
  # With the default shared codebook the arms federate TWO things at once: the VQ
  # dictionary (via the Prop.1 sufficient-statistic merge, which agrees with the pooled
  # centroid to ~1.5e-7 in fp32 — one ulp, i.e. zero aggregation loss up to rounding; the
  # 1e-12..1e-14 figures elsewhere in the repo are float64 and belong to the closed-form
  # floor heads, not to this merge) and the encoder (via weight averaging). Only the cb-local twins
  # isolate the encoder term, and they keep the quantizer the `local` baseline actually
  # got, so the pairing is honest. There is no fedproto twin: its classes ARE the shared
  # codebook indices (federated.py refuses it, by design).
  [commoninit_cblocal]="federated_enc_commoninit_cblocal|"
  [fedavg_cblocal]="federated_enc_fedavg_cblocal|"
  [fedprox_cblocal]="federated_enc_fedprox_cblocal|--fedprox-mu $FEDPROX_MU --fedprox-form $FEDPROX_FORM"
  # ── codebook-only rungs ────────────────────────────────────────────────────────────
  # cb_only federates the DICTIONARY and nothing else: encoder/decoder/prior all local, and
  # fed_encoder stays "off" so each client draws its OWN encoder init. That last part is what
  # separates it from commoninit_cbshared, which broadcasts one init at the same codebook
  # regime — the pair is the cleanest single-axis read on what a shared starting point buys.
  # cb_only_ema adds the server-side EMA over rounds (gamma=0.8); on wsd it was the first
  # federated arm to beat `local`, so it belongs in any table that claims federation fails.
  [cb_only]="federated_cb_only|"
  [cb_only_ema]="federated_cb_only_ema|"
)
# Default = the 6 headline columns. Add mechanism columns without touching the defaults:
#   ARMS="local centralized fedavg fedprox fedproto commoninit fedavg_cblocal commoninit_cblocal"
# Resumable + additive: a re-run with a longer ARMS list only fills in the missing jsons.
ARM_ORDER=(${ARMS:-local centralized fedavg fedprox fedproto commoninit})
for a in "${ARM_ORDER[@]}"; do
  [[ -n "${ARM[$a]:-}" ]] || { echo "unknown arm '$a'. known: ${!ARM[*]}"; exit 1; }
done

mk_cmd() {  # $1 cluster  $2 arm-tag
  local cl="$1" tag="$2"; local spec="${ARM[$tag]}"
  local id="${spec%%|*}" extra="${spec#*|}" wflag=""
  # In fixed mode do NOT pass --window-length at all: cell C must be byte-identical to
  # cell A's command except for the codebook, and passing "128" explicitly would still
  # differ from A's command line in a way that is easy to misread later.
  [[ "$WINDOW_MODE" == "2p" ]] && wflag="--window-length ${WIN[$cl]}"
  echo "$PY -u pipeline/federated_eval.py --dataset ucr_split --cluster $cl \
--arms $id $COMMON $wflag --out-dir $CKPT $extra \
--out-json $RESDIR/${cl}__${tag}${SFX}.json"
}

# wave 0 = all 6 arms on the 3 cheapest clusters (fast failure signal), then LPT descending
SEEDW=("${CLUSTERS[@]: -3}")
is_seed() { local c; for c in "${SEEDW[@]}"; do [[ "$c" == "$1" ]] && return 0; done; return 1; }
add() { JOBS+=("$1|$RESDIR/$2__$3${SFX}.json|$(mk_cmd "$2" "$3")"); }   # SFX: see above
for cl in "${SEEDW[@]}"; do for a in "${ARM_ORDER[@]}"; do add "${cl}_${a}" "$cl" "$a"; done; done
for cl in "${CLUSTERS[@]}"; do
  is_seed "$cl" && continue
  for a in "${ARM_ORDER[@]}"; do add "${cl}_${a}" "$cl" "$a"; done
done

say "=== ucrsplit [$TAG]: ${#JOBS[@]} jobs (${#CLUSTERS[@]} clusters x ${#ARM_ORDER[@]} arms), seed=$SEED ==="
say "    arms: ${ARM_ORDER[*]}"
if [[ "$WINDOW_MODE" == "2p" ]]; then
  say "    window: PER-CLUSTER 2*period (paper Algorithm 1)   tolerance PINNED at $TOL (not window//2)"
else
  say "    window: config default 128 (unchanged)             tolerance PINNED at $TOL"
fi
say "    codebook: $([[ "$CODEBOOK" == 0 ]] && echo '64 (config default)' || echo "$CODEBOOK (paper)")"
say "    convergence: --s1-rounds $S1_ROUNDS --fed-patience-rounds $PATIENCE --local-epochs $LOCAL_EPOCHS --batch $BATCH"
say "    fedprox=${FEDPROX_FORM}/mu${FEDPROX_MU}  fedproto=${FEDPROTO_AGG}/lam${FEDPROTO_LAM}"
say "    SKIPPED (10% client would have < $MIN_WIN windows at 2P): $SKIPLINE"
say "    predicted ~${TOTALCOST} slot-h/arm-set -> ~$(awk -v c="$TOTALCOST" -v s=$((${#GPUS[@]}*SLOTS_PER_GPU)) 'BEGIN{printf "%.1f", c*6/s/24}') days at $((${#GPUS[@]}*SLOTS_PER_GPU)) slots"
[[ "$SUBSET" -gt 0 ]] && say "    SUBSET=$SUBSET -> only the $SUBSET cheapest clusters (probe mode)"

if [[ -n "${DRYRUN:-}" ]]; then
  n=0
  for j in "${JOBS[@]}"; do
    IFS='|' read -r name out cmd <<<"$j"; n=$((n+1))
    [[ -f "$out" ]] && tag="[skip:exists]" || tag="[run]"
    if [[ $n -le 12 || -n "${DRYRUN_ALL:-}" ]]; then printf '  %3d  %-28s %-14s\n' "$n" "$name" "$tag"; fi
    [[ $n -eq 13 && -z "${DRYRUN_ALL:-}" ]] && echo "  ... (DRYRUN_ALL=1 to list all)"
  done
  echo "  ($n jobs; unset DRYRUN to run)"
  echo "  sample: $(IFS='|' read -r _ _ c <<<"${JOBS[0]}"; echo "$c" | tr -s ' ')"
  exit 0
fi

declare -A SLOT_PID SLOT_NAME; SLOT_KEYS=()
for g in "${GPUS[@]}"; do for ((i=0;i<SLOTS_PER_GPU;i++)); do SLOT_PID["$g:$i"]=""; SLOT_NAME["$g:$i"]=""; SLOT_KEYS+=("$g:$i"); done; done
free_slot() { local k; for k in "${SLOT_KEYS[@]}"; do [[ -z "${SLOT_PID[$k]}" ]] && { echo "$k"; return; }; done; }
NFAIL=0
reap() { local k p rc; for k in "${SLOT_KEYS[@]}"; do p="${SLOT_PID[$k]}"; [[ -z "$p" ]] && continue
  if ! kill -0 "$p" 2>/dev/null; then wait "$p" 2>/dev/null; rc=$?
    say "DONE  ${SLOT_NAME[$k]}  (slot $k, rc=$rc)"
    [[ $rc -ne 0 ]] && { NFAIL=$((NFAIL+1)); say "  !! FAILED: $LOGDIR/${SLOT_NAME[$k]}.log"; }
    SLOT_PID[$k]=""; SLOT_NAME[$k]=""; fi; done; }

for job in "${JOBS[@]}"; do
  IFS='|' read -r name out cmd <<<"$job"
  if [[ -f "$out" ]]; then say "SKIP  $name (exists)"; continue; fi
  while :; do reap; slot="$(free_slot)"; [[ -n "$slot" ]] && break; sleep 20; done
  gpu="${slot%%:*}"
  say "START $name (W=${WIN[${name%_*}]}) -> GPU$gpu (slot $slot)"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    nohup bash -c "$cmd" > "$LOGDIR/$name.log" 2>&1 &
  SLOT_PID[$slot]=$!; SLOT_NAME[$slot]="$name"; sleep 3
done
say "all dispatched; waiting"
while :; do reap; r=0; for k in "${SLOT_KEYS[@]}"; do p="${SLOT_PID[$k]}"; [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null && r=$((r+1)); done; [[ $r -eq 0 ]] && break; sleep 20; done

say "=== convergence audit: any TRUNCATED row is NOT reportable (hard rule 2026-07-24) ==="
"$PY" - "$CKPT" <<'PYEOF' 2>&1 | tee -a "$ORCH"
import json, sys, glob, os
files = sorted(glob.glob(os.path.join(sys.argv[1], "**", "fed_history.json"), recursive=True))
print(f"  scanned {len(files)} fed_history.json" + ("  -- NONE FOUND" if not files else ""))
trunc = 0
for f in files:
    h = json.load(open(f)).get("stage1", [])
    if not h: continue
    if h[-1].get("truncated"):
        trunc += 1
        who = "/".join(os.path.relpath(f, sys.argv[1]).split(os.sep)[0:3])
        print(f"  {who:48s} rounds->{h[-1].get('round')}  TRUNCATED, NOT CONVERGED")
print(f"  {trunc} truncated / {len(files)} runs")
PYEOF
say "=== done ($NFAIL failed). jsons in $RESDIR ==="
say "    A/B vs the fixed-128 run: compare $RESDIR/*.json against $REPO/artifacts/ucrsplit/*.json"
