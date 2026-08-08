#!/usr/bin/env bash
# launch.sh — THE launcher. One entry point, any dataset, any cluster subset, any arms.
#
# Everything that decides WHICH DATA a run covers lives in a cohort file (cohorts/<name>.json,
# built by scripts/cohort.py). The launcher never picks clusters, windows or tolerances -- it
# reads them. That is the whole point: two runs naming the same cohort are matched by
# construction, and an ablation is the SAME cohort with different knobs.
#
#   bash scripts/launch.sh --cohort full --arms local,centralized --tag main
#   bash scripts/launch.sh --cohort full --arms local --tag abl_cb128 --extra "--codebook-size 128"
#   bash scripts/launch.sh --cohort probe --arms local --tag t1 --dry
#   bash scripts/launch.sh --cohort full --engine floor --heads ma_c,ar,pca --tag floor1
#
# Resumable: a job whose out-json already exists is skipped, so re-running after an
# interruption costs only what is missing. Kill safely with
#   pkill -f "[l]aunch.sh" && pkill -f "[f]ederated_eval.py"
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PY:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10}"
export PIPELINE_PYTHON="$PY"
# Private MPS pipe dir. Never the compiled-in /tmp/nvidia-mps (0777, 0666 control socket):
# a server death there wedged 14 jobs for 9.9 h with no error. See scripts/mps_ctl.sh.
MPS_CTL_ENV_ONLY=1 source "$REPO/scripts/mps_ctl.sh"

# ── PRECISIONE FISSATA (2026-08-04) ─────────────────────────────────────────────
# `_amp_dtype()` legge FEDVQ_AMP e nessuno script lo esportava, quindi vinceva sempre
# `auto` -- che sceglie in base alla compute capability: fp16 sotto Ampere, bf16 da
# Ampere in su. Cioe' fp16 su g2 (Quadro RTX 8000, sm_75) e bf16 su g4 (Ada 8.9 /
# 3090 8.6): la precisione di training dipendeva da QUALE HOST prendeva il job.
# Nell'archivio pre-cutoff: 195 celle bf16, 150 fp16.
#
# Non e' igiene, e' portante. L'ablazione `abl_fp32` su ucr_001 (centralized, W=2P)
# sposta AUPRC 0,646 -> 0,418 e top-1 1,00 -> 0,40 -- piu' grande di OGNI effetto che
# lo studio vuole misurare (contributo (A) = -0,112, centralized>local = +0,212).
#
# fp16 e non bf16: su Turing bf16 e' EMULATO. Misurato su questo nodo, matmul 4096^3:
# fp32 22,9 ms / fp16 3,4 ms / bf16 40,2 ms. I tensor core fp16 esistono anche su
# Ampere e Ada, quindi fp16 e' l'unica precisione veloce su ENTRAMBI gli host.
# Le parti delicate restano fp32 comunque: l'accumulo delle statistiche sufficienti
# del VQ ha l'autocast disabilitato dentro il quantizer (Prop.1 resta esatta) e
# eval/metriche sono fp32.
export FEDVQ_AMP="${FEDVQ_AMP:-fp16}"

COHORT=""; ARMS="local,centralized"; TAG=""; ENGINE="deep"; EXTRA=""; HEADS="ma_c"
MODES="local"; DRY=0
# Override with LAUNCH_GPUS="4 5" when driving another host through the sshfs mount.
#
# Senza override, su g2 il set NON e' piu' cablato a "0 1": lo CALCOLA scripts/_g2_gpu.sh,
# perche' g2 e' condivisa con ssanchez e la regola (utente, 2026-08-06) e':
#     altri su 0 GPU -> usiamo ENTRAMBE
#     altri su 1 GPU -> usiamo l'ALTRA, e una sola
#     altri su 2 GPU -> ne prendiamo UNA sola, la meno carica, senza impedirgli il lavoro
# In nessun caso occupiamo entrambe le schede se qualcun altro sta lavorando. Calcolato e non
# ricordato apposta: il 2026-08-06 avevo contato i processi di ssanchez come nostri e gli sono
# finito sopra con 5 job. Stessa filosofia della z-norm pinnata nella coorte.
if [[ -n "${LAUNCH_GPUS:-}" ]]; then
  read -r -a GPUS <<< "$LAUNCH_GPUS"
elif [[ "$(hostname -s)" == "g2" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  read -r -a GPUS <<< "$(bash "$REPO/scripts/_g2_gpu.sh")"
  echo "[vincoli g2] schede prese: ${GPUS[*]}"
  bash "$REPO/scripts/_g2_gpu.sh" --spiega >/dev/null
else
  read -r -a GPUS <<< "0 1"
fi
# ⛔ g4: GPU0 e GPU5 sono RISERVATE (utente, 2026-08-06) — anche vuote, non sono nostre.
# Patch applicata il 2026-08-07 A DISPATCHER FERMI (bash rilegge gli script a offset di
# byte: mai editare questo file con istanze vive). Deroga: G4_ALLOW_RESERVED=1, visibile
# nel comando. Il default "0 1" del ramo sopra su g4 viene cosi' RIFIUTATO, non corretto
# in silenzio: chi lancia deve scegliere schede lecite (o usare `_g4_gpu.sh --libere`).
if [[ "$(hostname -s)" == aida-g4* || "$(hostname -s)" == g4* ]]; then
  bash "$REPO/scripts/_g4_gpu.sh" "${GPUS[*]}" || exit 4
fi
SLOTS_PER_GPU="${SLOTS_PER_GPU:-7}"                  # measured optimum under MPS
PROTOCOL="${PROTOCOL:-converged}"
S1_ROUNDS="${S1_ROUNDS:-300}"; LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"
PATIENCE="${PATIENCE:-6}"; BATCH="${BATCH:-64}"
# --s2-rounds was NEVER passed, so it silently stayed at the argparse default of 2: the three
# arms with a shared prior body (federated, federated_shared, federated_fedavg_cb_sharedprior)
# would have trained that body for 2 rounds while stage 1 ran 300, and RUN.json would not even
# have recorded it. Mirrors S1_ROUNDS because stage 2 now honours --fed-patience-rounds too,
# so this is a CEILING that stops where it flattens, not a budget that is spent in full.
S2_ROUNDS="${S2_ROUNDS:-$S1_ROUNDS}"
# `--arms paper` expands to the reporting table (federated_eval.PAPER_ARMS), so the launch
# command cannot drift from the table the paper prints.
PAPER_ARMS_CSV="$("$PY" -c "
import ast,sys
t=ast.parse(open('pipeline/federated_eval.py').read())
g={n.targets[0].id:n.value for n in t.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
print(','.join(e.value for e in g['PAPER_ARMS'].elts))")"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cohort) COHORT="$2"; shift 2;;
    --arms)   ARMS="$2";   shift 2;;
    --tag)    TAG="$2";    shift 2;;
    --engine) ENGINE="$2"; shift 2;;
    --extra)  EXTRA="$2";  shift 2;;
    --heads)  HEADS="$2";  shift 2;;
    --modes)  MODES="$2";  shift 2;;
    --dry)    DRY=1;       shift;;
    *) echo "unknown flag $1"; exit 2;;
  esac
done
[[ -n "$COHORT" && -n "$TAG" ]] || { echo "need --cohort and --tag"; exit 2; }
[[ -f "cohorts/$COHORT.json" ]] || { echo "no cohorts/$COHORT.json (scripts/cohort.py new)"; exit 2; }
[[ "$ARMS" == "paper" ]] && ARMS="$PAPER_ARMS_CSV"
# An unknown --engine used to fall through to the deep dispatcher while SKIPPING the arm-name
# validation below (which is gated on `== "deep"`): a typo like `--engine dep` launched 428
# real jobs with unchecked arm names. Every engine value must be spelled out here.
ENGINES=(deep floor)
printf '%s\n' "${ENGINES[@]}" | grep -qxF -- "$ENGINE" || {
  echo "unknown --engine '$ENGINE'; known: ${ENGINES[*]}"; exit 2; }

RUNDIR="$REPO/artifacts/runs/$TAG"; LOGDIR="$REPO/logs/runs/$TAG"
# A --dry must leave NOTHING on disk. It used to mkdir both trees and tee the orchestrator
# log into logs/runs/<tag>/, so dry-running a tag left an empty artifacts/runs/<tag>/ that is
# indistinguishable from a real run whose jobs all failed -- the same confusion the RUN.json
# dry-guard below was added to fix on 2026-07-30, one level up.
# Remembered so a later refusal (the disk preflight) can undo a tree IT created without ever
# touching a tag that was already on disk from a real run. RUN.json is rewritten unconditionally
# a few lines down -- BEFORE the job list exists and so before the preflight can decide -- so a
# refused re-run against an existing tag would otherwise have already clobbered that tag's
# manifest. Snapshot it; the refusal path puts it back, the dispatch path drops the copy.
TAG_IS_NEW=0; [[ -e "$RUNDIR" ]] || TAG_IS_NEW=1
RUNJSON_BAK=""
if [[ $DRY -eq 0 && -f "$RUNDIR/RUN.json" ]]; then
  RUNJSON_BAK="$(mktemp)"; cp "$RUNDIR/RUN.json" "$RUNJSON_BAK"
fi
[[ $DRY -eq 0 ]] && mkdir -p "$RUNDIR" "$LOGDIR"
ORCH="$LOGDIR/_orchestrator.log"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
# L'HOST nel prefisso: `logs/runs/<tag>/_orchestrator.log` sta sul filesystem CONDIVISO, e
# g2 e g4 ci scrivono dentro entrambi. Senza il nome dell'host le righe dei due si mescolano
# e "GPU1" non dice quale scheda sia.
HOST_TAG="$(hostname)"
say() { if [[ $DRY -eq 1 ]]; then echo "[$(stamp)][$HOST_TAG] $*"
        else echo "[$(stamp)][$HOST_TAG] $*" | tee -a "$ORCH"; fi; }

FP=$("$PY" -c "import json;print(json.load(open('cohorts/$COHORT.json'))['fingerprint'])")
# Seeds are pinned in the cohort; the launcher used to hardcode `--seeds 0` and silently
# ignore them. federated_eval loops seeds internally and pools the per-client reports into
# ONE json, so passing the list through is the whole fix -- no extra jobs, no merge step.
SEEDS=$("$PY" -c "import json;print(','.join(str(s) for s in json.load(open('cohorts/$COHORT.json'))['seeds']))")

# ── --extra may not touch what the cohort certifies ──────────────────────────────────
# $EXTRA is appended LAST to the job command and argparse keeps the LAST occurrence, so
# `--extra "--metrics-tolerance 64"` silently REPLACES the tolerance the cohort pinned while
# RUN.json goes on declaring that cohort's fingerprint. The result is a run that claims to be
# matched to every other tag with the same fingerprint and is not -- the unpinned-axis failure
# the whole cohort system exists to prevent, now wearing a certificate. Refuse it here: an
# ablation on a pinned axis IS a different cohort, not the same one with a knob turned.
# `--clusters` is here too because the floor engine spells the same pinned axis in the plural
# (floor_eval.py --clusters), and it is passed BEFORE $EXTRA there as well.
PINNED_FLAGS=(--window-length --metrics-tolerance --seeds --dataset --cluster --clusters
              --out-dir --cohort)
if [[ -n "$EXTRA" ]]; then
  read -ra _EXTRA_TOK <<<"$EXTRA"
  for _tok in "${_EXTRA_TOK[@]}"; do
    _flag="${_tok%%=*}"                    # matches both `--flag val` and `--flag=val`
    if printf '%s\n' "${PINNED_FLAGS[@]}" | grep -qxF -- "$_flag"; then
      echo "refusing --extra: '$_flag' is PINNED by cohort '$COHORT' ($FP)."
      echo "  Those values (window, tolerance, seeds, dataset, cluster set, out-dir) are what"
      echo "  the cohort_fingerprint certifies. Overriding one on the command line produces a"
      echo "  run whose RUN.json claims comparability it does not have."
      echo "  The right way is a NEW cohort:"
      echo "    $PY scripts/cohort.py new <newname> --datasets ... --clusters ... [--seeds ...]"
      echo "    bash scripts/launch.sh --cohort <newname> --arms $ARMS --tag <newtag>"
      echo "  (pinned: ${PINNED_FLAGS[*]})"
      exit 2
    fi
  done
fi

# Arm names are validated HERE, once, against the registry -- not 438 times inside jobs that
# each pay a data-loading pass before raising. `--arms paper` is already expanded above.
if [[ "$ENGINE" == "deep" ]]; then
  "$PY" - "$ARMS" <<'PYEOF' || exit 2
import ast, sys
t = ast.parse(open("pipeline/federated_eval.py").read())
g = {n.targets[0].id: n.value for n in t.body
     if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
known = {e.value for k in ("PAPER_ARMS", "OTHER_ARMS") for e in g[k].elts}
bad = [a for a in sys.argv[1].split(",") if a.strip() and a.strip() not in known]
if bad:
    print(f"unknown arm(s) {bad}\nknown arms:\n  " + "\n  ".join(sorted(known)))
    raise SystemExit(2)
PYEOF
fi

# MPS: the pipe dir is exported above, but a dead daemon means every job silently falls back
# to time-slicing (or wedges on a socket poll, as on 2026-07-28). Bring it up and say so.
# NO_MPS=1 skips it entirely -- mandatory on a SHARED host (g4 has 20+ users): starting an
# MPS daemon there would route other people's contexts through our server.
if [[ -n "${NO_MPS:-}" ]]; then
  say "MPS: disabilitato (NO_MPS=1) -- host condiviso, time-slicing nativo"
elif [[ "$ENGINE" == "deep" && $DRY -eq 0 ]]; then
  if pgrep -u "$(id -u)" -f 'nvidia-cuda-mps-control -d' >/dev/null 2>&1; then
    say "MPS: daemon already up on $CUDA_MPS_PIPE_DIRECTORY"
  else
    say "MPS: no daemon on $CUDA_MPS_PIPE_DIRECTORY -- starting one"
    bash "$REPO/scripts/mps_ctl.sh" up 2>&1 | tee -a "$ORCH" | tail -3
  fi
fi

# ── the run manifest: what this tag IS, written before anything runs ─────────────────
# Without this a directory of jsons is uninterpretable six months later: you cannot tell
# which cohort, which protocol or which knobs produced it. Every comparison this repo has
# had to retract failed on exactly that.
# --dry must NOT mint one: it would claim a run at the full budget that never happened, and
# a later reader has no way to tell an orphan manifest from a run whose jobs all failed.
# (Caught 2026-07-30: a --dry left artifacts/runs/encdrift_v1/RUN.json declaring s1_rounds=300
# and zero jobs.) The dry path prints the joblines and exits further down.
# TVQ_SMOKE collapses every converged budget (config.apply_env_overrides), so a smoke run's
# numbers are meaningless -- but it produces a RUN.json with a VALID cohort_fingerprint and a
# note that asserts comparability. Without this field a smoke run is indistinguishable from a
# real one at the manifest level, which is precisely the failure the cohort system exists to
# prevent. Caught 2026-07-30 by a smoke of the encdrift launch. The note goes red too.
if [[ "${TVQ_SMOKE:-}" == "1" ]]; then
  SMOKE_FLAG=true
  SMOKE_NOTE="!! TVQ_SMOKE=1 -- ALL BUDGETS COLLAPSED. These numbers are MEANINGLESS and this run is comparable to NOTHING, same fingerprint or not. Code-path exercise only."
  say "!! TVQ_SMOKE=1 -- this run is NOT reportable; RUN.json is flagged."
else
  SMOKE_FLAG=false
  SMOKE_NOTE="Comparable to any other tag with the same cohort_fingerprint. An ablation is the same cohort with different extra_flags."
fi
# Serialised by json.dumps, not by shell interpolation: a tag or an --extra carrying a quote
# or a backslash used to emit a file that is not JSON at all, and RUN.json is the ONE artifact
# that says what a tag is -- an unparseable one is worse than none, because the directory
# still looks documented. Values travel as environment variables so nothing is re-quoted.
if [[ $DRY -eq 0 ]]; then
  RJ_TAG="$TAG" RJ_SMOKE="$SMOKE_FLAG" RJ_COHORT="$COHORT" RJ_FP="$FP" RJ_ENGINE="$ENGINE" \
  RJ_ARMS="$ARMS" RJ_HEADS="$HEADS" RJ_MODES="$MODES" RJ_EXTRA="$EXTRA" \
  RJ_PROTOCOL="$PROTOCOL" RJ_S1="$S1_ROUNDS" RJ_S2="$S2_ROUNDS" RJ_EPOCHS="$LOCAL_EPOCHS" \
  RJ_PATIENCE="$PATIENCE" RJ_BATCH="$BATCH" RJ_SEEDS="$SEEDS" RJ_STARTED="$(stamp)" \
  RJ_NOTE="$SMOKE_NOTE" RJ_OUT="$RUNDIR/RUN.json" "$PY" - <<'PYEOF' || exit 2
import json, os
e = os.environ
json.dump({
    "tag": e["RJ_TAG"],
    "smoke": e["RJ_SMOKE"] == "true",
    "cohort": e["RJ_COHORT"],
    "cohort_fingerprint": e["RJ_FP"],
    "engine": e["RJ_ENGINE"],
    "arms": e["RJ_ARMS"],
    "heads": e["RJ_HEADS"],
    "modes": e["RJ_MODES"],
    "extra_flags": e["RJ_EXTRA"],
    "protocol": e["RJ_PROTOCOL"],
    "s1_rounds": int(e["RJ_S1"]),
    "s2_rounds": int(e["RJ_S2"]),
    "local_epochs": int(e["RJ_EPOCHS"]),
    "fed_patience_rounds": int(e["RJ_PATIENCE"]),
    "batch": int(e["RJ_BATCH"]),
    "seeds": e["RJ_SEEDS"],
    "started": e["RJ_STARTED"],
    "note": e["RJ_NOTE"],
}, open(e["RJ_OUT"], "w"), indent=2)
PYEOF
fi

# ── FLOOR engine: CPU only, no slots, one process per dataset (config is per-dataset) ──
if [[ "$ENGINE" == "floor" ]]; then
  say "=== floor [$TAG] cohort=$COHORT ($FP) heads=$HEADS modes=$MODES ==="
  "$PY" - "$COHORT" <<'PYEOF' | while read -r ds cls; do
import json, sys
c = json.load(open(f"cohorts/{sys.argv[1]}.json"))
for ds, v in c["datasets"].items():
    print(ds, ",".join(v["clusters"]))
PYEOF
    n=$(awk -F, '{print NF}' <<<"$cls")
    [[ $DRY -eq 1 ]] && { say "DRY floor $ds: $n cluster (${cls:0:60}$([[ ${#cls} -gt 60 ]] && echo ' ...'))"; continue; }
    say "floor -> $ds"
    "$PY" scripts/floor_eval.py --dataset "$ds" --clusters "$cls" \
        --heads "$HEADS" --modes "$MODES" --jobs "${FLOOR_JOBS:-4}" \
        --out-dir "$RUNDIR/floor" $EXTRA 2>&1 | tee -a "$LOGDIR/floor_$ds.log" | tail -3
  done
  say "=== floor done -> $RUNDIR/floor ==="
  exit 0
fi

# ── DEEP engine ──────────────────────────────────────────────────────────────────────
mapfile -t JOBLINES < <("$PY" scripts/cohort.py jobs "$COHORT" --arms "$ARMS" --tag "$TAG")
# ── LAUNCH_ONLY_CLUSTERS: dividere UNA coorte fra due host ──────────────────────
# `launch.sh` non ha arbitraggio cross-host: lanciarlo su g2 e g4 con lo stesso tag
# fa dispatchare a entrambi la stessa lista (l'idempotenza piu' sotto non salva --
# a disco vuoto l'out-json non esiste ancora per nessuno dei due) e due processi
# finirebbero per scrivere lo stesso stage1.ckpt.
#
# La divisione va fatta per SERIE, non per arm: g2 e g4 hanno architetture diverse
# (sm_75 vs sm_86) e con cudnn.benchmark=True i kernel scelti a tempo cambiano la
# traiettoria di training. Dividendo per arm il confondente hardware finisce DENTRO
# un contrasto appaiato; dividendo per serie resta FRA serie, dove i test appaiati
# non lo vedono. Stessa logica del warning in scripts/run_on_g4.sh.
#
# Il filtro sta QUI e non nel ciclo di dispatch, cosi' --dry mostra quello che
# girera' davvero invece della coorte intera.
#
# Una sola coorte -> UN solo cohort_fingerprint per entrambe le meta'. Due coorti
# con liste di cluster diverse darebbero fingerprint diversi, e le due meta' si
# dichiarerebbero non confrontabili.
#
#   g2:  LAUNCH_ONLY_CLUSTERS="ucr_001,ucr_011"      bash scripts/launch.sh ...
#   g4:  LAUNCH_ONLY_CLUSTERS="ucr_014,ucr_043,..."  bash scripts/launch.sh ...
if [[ -n "${LAUNCH_ONLY_CLUSTERS:-}" ]]; then
  _nall=${#JOBLINES[@]}; _keep=()
  for _l in "${JOBLINES[@]}"; do
    _cl=$(awk '{print $2}' <<<"$_l")
    [[ ",${LAUNCH_ONLY_CLUSTERS}," == *",$_cl,"* ]] && _keep+=("$_l")
  done
  JOBLINES=("${_keep[@]}")
  say "!! LAUNCH_ONLY_CLUSTERS=$LAUNCH_ONLY_CLUSTERS -- $((_nall-${#JOBLINES[@]})) job filtrati, ${#JOBLINES[@]} restano"
  say "!! questa e' una META' della coorte: l'altra deve girare altrove, o la tabella e' incompleta"
  if [[ ${#JOBLINES[@]} -eq 0 ]]; then
    echo "REFUSING: il filtro non ha lasciato nessun job -- nome cluster sbagliato?" >&2; exit 2
  fi
fi
say "=== deep [$TAG] cohort=$COHORT ($FP) ==="
say "    arms: $ARMS"
say "    protocol=$PROTOCOL s1_rounds=$S1_ROUNDS s2_rounds=$S2_ROUNDS local_epochs=$LOCAL_EPOCHS patience=$PATIENCE batch=$BATCH"
say "    extra: ${EXTRA:-<none>}"
say "    ${#JOBLINES[@]} jobs -> $RUNDIR"

# ── DISK PREFLIGHT ───────────────────────────────────────────────────────────────────
# A full disk halfway through a multi-day run is the worst way to lose a week: the jobs
# that already finished are fine, the one writing dies mid-checkpoint, and the audit at
# the end cannot tell "not run" from "ran and could not save". Estimate BEFORE dispatch.
#
# MEASURED, not derived. The first version of this check scaled the footprint by params per
# client (0.1M at W=128 to 85.7M at W=3028, LAUNCH_RUNBOOK §5.7) and overestimated by ~30x,
# refusing runs that fit comfortably. The reason: the encoder is NOT what fills the disk.
# In a real cluster-arm directory the per-entity cost is stage2.ckpt (the prior, 3.7 MB at
# W=128) against stage1.ckpt (0.5 MB), plus one _fed_resume.pt (4.4 MB) for the whole cohort
# -- and the prior does not grow with the window the way the encoder does.
#
# Calibrated on 277 real cluster-arm directories under artifacts/_archive_20260729/ (measured
# 2026-07-30), median MB per cluster-arm:
#     ucrsplit      W=128        266 dirs    5.6 MB   (1.1 MB/entity, no stage2 saved)
#     converge60    W=128 wsd      -         27.0 MB  (5.4 MB/entity, stage2 + resume bundle)
#     ucrsplit_w2p  W>1024        11 dirs   36.4 MB   (7.3 MB/entity), max observed 55.3
# The steps below take the UPPER regime at each window (stage2 present), so the estimate is
# an over-, not under-approximation -- and the gate is still a margin, not equality, because
# a cohort with fatter clusters than the ~5 clients these were measured at will exceed it.
DISK_EST_MB=$(printf '%s\n' "${JOBLINES[@]}" | awk '
  { w = $4 + 0
    mb += (w <= 256) ? 30 : (w <= 512) ? 40 : (w <= 1024) ? 55 : (w <= 1782) ? 80 : 120 }
  END { printf "%.0f", mb }')
# df on $REPO, not on $RUNDIR: the run directory does not exist yet under --dry (it must not
# -- an orphan manifest is worse than none), and df on a missing path reports nothing, which
# silently disabled this whole check. $RUNDIR is always under $REPO, so same filesystem.
DISK_FREE_MB=$(df -Pm "$REPO" 2>/dev/null | awk 'NR==2{print $4}')
say "    disk: need ~$((DISK_EST_MB/1024)) GB (estimate), free $((DISK_FREE_MB/1024)) GB"
if [[ -n "$DISK_FREE_MB" && "$DISK_EST_MB" -gt 0 ]]; then
  if (( DISK_FREE_MB < DISK_EST_MB )); then
    echo "REFUSING: this run is estimated at ~$((DISK_EST_MB/1024)) GB of checkpoints and only" >&2
    echo "  $((DISK_FREE_MB/1024)) GB are free on $(df -P "$RUNDIR" | awk 'NR==2{print $6}')." >&2
    echo "  A run that fills the disk mid-flight leaves a tree that cannot be told apart from" >&2
    echo "  one whose jobs failed. Options, cheapest first:" >&2
    echo "    * split by dataset into several tags on the SAME cohort (fingerprint is preserved," >&2
    echo "      so the tags stay paired -- LAUNCH_RUNBOOK section 6)" >&2
    echo "    * rebuild the cohort with a smaller --max-window (the window drives the estimate)" >&2
    echo "    * free space, or set ALLOW_LOW_DISK=1 to override this check" >&2
    if [[ "${ALLOW_LOW_DISK:-}" != "1" ]]; then
      # $RUNDIR/$LOGDIR and RUN.json were written at startup, before the job list existed and
      # so before this estimate could be made. Leaving them behind puts a MANIFEST for a run
      # that never happened on disk -- the same "did it not run, or did it run and fail?"
      # ambiguity the --dry guard was added to fix. Undo, but ONLY the tree this invocation
      # created: re-running a refused command against an existing tag must not delete the
      # manifest of the real run already there.
      if [[ $TAG_IS_NEW -eq 1 ]]; then
        rm -f "$RUNDIR/RUN.json" "$ORCH"
        rmdir "$RUNDIR" "$LOGDIR" 2>/dev/null || true
      elif [[ -n "$RUNJSON_BAK" ]]; then
        cp "$RUNJSON_BAK" "$RUNDIR/RUN.json"
        echo "  (tag '$TAG' already existed -- its RUN.json was restored, nothing was changed)" >&2
      fi
      [[ -n "$RUNJSON_BAK" ]] && rm -f "$RUNJSON_BAK"
      exit 2
    fi
    say "    ALLOW_LOW_DISK=1 -- proceeding anyway"
  elif (( DISK_FREE_MB < DISK_EST_MB * 3 / 2 )); then
    say "!! disk margin under 1.5x the estimate -- a fatter-than-5-client cohort may still fill it"
  fi
fi
# Preflight cleared: the manifest just written is the one that stands, so drop the snapshot.
[[ -n "$RUNJSON_BAK" ]] && rm -f "$RUNJSON_BAK" && RUNJSON_BAK=""

if [[ $DRY -eq 1 ]]; then
  printf '%s\n' "${JOBLINES[@]}" | head -8 | awk '{printf "    %s %s %s  W=%s tol=%s\n",$1,$2,$3,$4,$5}'
  echo "    ... (${#JOBLINES[@]} jobs; togli --dry per lanciare)"
  # The checkpoint roots are worth showing: they must be one per DATASET (see the dispatch
  # loop for why a shared root lets two datasets overwrite the same stage1.ckpt), and a dry
  # run is the only cheap place to check it.
  printf '%s\n' "${JOBLINES[@]}" | awk '{print $1}' | sort -u | while read -r d; do
    echo "    --out-dir $RUNDIR/ckpt/$d"
  done
  exit 0
fi

declare -A SLOT_PID SLOT_NAME; SLOT_KEYS=()
for g in "${GPUS[@]}"; do for ((i=0;i<SLOTS_PER_GPU;i++)); do
  SLOT_PID["$g:$i"]=""; SLOT_NAME["$g:$i"]=""; SLOT_KEYS+=("$g:$i"); done; done
free_slot() { local k; for k in "${SLOT_KEYS[@]}"; do [[ -z "${SLOT_PID[$k]}" ]] && { echo "$k"; return; }; done; }
NFAIL=0; NRUN=0; NSKIP=0
reap() { local k p rc; for k in "${SLOT_KEYS[@]}"; do p="${SLOT_PID[$k]}"; [[ -z "$p" ]] && continue
  if ! kill -0 "$p" 2>/dev/null; then wait "$p" 2>/dev/null; rc=$?
    say "DONE  ${SLOT_NAME[$k]} (slot $k, rc=$rc)"
    [[ $rc -ne 0 ]] && { NFAIL=$((NFAIL+1)); say "  !! FAILED -> $LOGDIR/${SLOT_NAME[$k]}.log"; }
    SLOT_PID[$k]=""; SLOT_NAME[$k]=""; fi; done; }

for line in "${JOBLINES[@]}"; do
  read -r ds cl arm win tol out <<<"$line"
  name="${ds}__${cl}__${arm}"
  if [[ -f "$out" ]]; then NSKIP=$((NSKIP+1)); continue; fi
  mkdir -p "$(dirname "$out")"
  # --out-dir MUST carry the dataset segment. federated_eval REPLACES the whole default root
  # artifacts/fed_eval/<dataset> with --out-dir and then appends only /<cluster>, so a plain
  # $RUNDIR/ckpt makes every dataset share one namespace. On cohort `paper` 177 of 428 cluster
  # names are used by 2-4 datasets; worst of all ucr_split (W=128) and ucr_split_w2p (W=2P,
  # e.g. ucr_001 -> 408) share BOTH cluster and entity names, so the same stage1.ckpt would be
  # written by two architectures with different window lengths. Same precedent and same reason
  # as run_enc_algo_sweep.sh:68 (--out-dir $CKPT/$1).
  cmd="$PY -u pipeline/federated_eval.py --dataset $ds --cluster $cl --arms $arm \
       --protocol $PROTOCOL --s1-rounds $S1_ROUNDS --s2-rounds $S2_ROUNDS \
       --local-epochs $LOCAL_EPOCHS \
       --fed-patience-rounds $PATIENCE --batch $BATCH --seeds $SEEDS \
       --window-length $win --metrics-tolerance $tol \
       --out-dir $RUNDIR/ckpt/$ds --out-json $out $EXTRA"
  while :; do reap; slot="$(free_slot)"; [[ -n "$slot" ]] && break; sleep 20; done
  gpu="${slot%%:*}"
  say "START $name (W=$win tol=$tol) -> GPU$gpu (slot $slot)"
  # ── DOVE e' girata questa cella ─────────────────────────────────────────────────
  # Senza questo l'hardware non e' ricostruibile a posteriori: `knobs.json` registra solo
  # `amp_dtype`, `RUN.json` non ha l'host, e il log dell'orchestratore e' CONDIVISO fra g2 e
  # g4 sul filesystem montato -- quindi "GPU1" e' ambiguo (esiste su entrambi, ma e' una
  # Quadro RTX 8000 sm_75 su uno e una RTX 3090 sm_86 sull'altro).
  #
  # Serve perche' i contrasti dello studio sono DENTRO la serie: se le celle di una serie
  # finissero su architetture diverse, il confondente hardware entrerebbe nel contrasto. La
  # precauzione va presa nello scheduling, ma senza questa riga non e' VERIFICABILE dopo.
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%FT%TZ)" "$(hostname)" "$gpu" \
    "$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader -i "$gpu" 2>/dev/null | tr -d ' ' | tr ',' '/')" \
    "$ds" "$cl" "$arm" >> "$RUNDIR/placement.tsv"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    nohup bash -c "$cmd" > "$LOGDIR/$name.log" 2>&1 &
  SLOT_PID[$slot]=$!; SLOT_NAME[$slot]="$name"; NRUN=$((NRUN+1)); sleep 3
done
say "dispatched $NRUN (skipped $NSKIP already on disk); waiting"
while :; do reap; r=0
  for k in "${SLOT_KEYS[@]}"; do p="${SLOT_PID[$k]}"; [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null && r=$((r+1)); done
  [[ $r -eq 0 ]] && break; sleep 20; done

# Hard rule 2026-07-24: a run whose best round is the last one is NOT converged and NOT
# reportable. The verdict only ever went to stdout, so it is re-derived here from disk.
say "=== convergence audit ==="
"$PY" - "$RUNDIR/ckpt" <<'PYEOF' 2>&1 | tee -a "$ORCH"
import json, sys, glob, os
# BOTH stages. This used to read stage1 only, so a run whose shared PRIOR body was truncated
# was reported as "0 truncated" -- and the three arms with a shared prior are exactly the ones
# whose stage 2 can truncate. Verified 2026-07-30: stage 1 restored round 0 while stage 2's
# best round WAS its last, and the audit said 0/1.
files = sorted(glob.glob(os.path.join(sys.argv[1], "**", "fed_history.json"), recursive=True))
bad = {}
for f in files:
    h = json.load(open(f))
    hit = [s for s in ("stage1", "stage2")
           if (rounds := h.get(s) or []) and rounds[-1].get("truncated")]
    if hit:
        bad[f] = hit
for f, hit in bad.items():
    print(f"  TRUNCATED, NOT CONVERGED ({'+'.join(hit)}): {os.path.relpath(f, sys.argv[1])}")
# "0 truncated / 0 runs" on an empty glob reads exactly like a clean bill of health, and that
# is the one case where the audit checked NOTHING. Say so instead: it means every job was
# skipped or died before writing a history, or the ckpt tree is not where this looks.
if not files:
    print(f"  NOTHING CHECKED: no fed_history.json under {sys.argv[1]}")
    print("  ^ this is NOT a pass. Every job was skipped (out-json already on disk) or failed")
    print("    before writing a history. Check the per-job logs before reporting anything.")
else:
    print(f"  {len(bad)} truncated / {len(files)} runs")
if bad:
    print("  ^ these rows are NOT reportable (hard rule 2026-07-24). Raise --s1-rounds /")
    print("    --s2-rounds; --fed-patience-rounds then stops each where it actually flattens.")
PYEOF
say "=== done ($NFAIL failed) -> $RUNDIR ==="
