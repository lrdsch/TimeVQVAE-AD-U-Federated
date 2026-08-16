#!/usr/bin/env bash
# make_release.sh — export the PUBLIC repository (the one the paper links to) into release/.
#
#   bash scripts/make_release.sh              # rebuild release/ from this working tree
#   OUT=/tmp/pub bash scripts/make_release.sh # somewhere else
#
# WHY A SCRIPT AND NOT A HAND-MADE COPY. The public tree must be derivable from this one, or
# the two drift and the repo the paper points at stops being the code that produced the
# numbers. Everything under release/ is generated here and nothing is edited in place: the
# hand-written parts (README, docs, .gitignore, the two scripts that only exist for the public
# tree) live under documentation/release/ and are COPIED in. Delete release/, re-run, get the
# same tree.
#
# WHAT IS DELIBERATELY LEFT OUT: the lab's operational surface (211 scripts of GPU brokers,
# curfews, watchdogs, sshfs bridges, campaign queues), the research ledger and its planning
# docs, the pre-znorm archive, and every arm the paper does not report. What is left is the
# pipeline that produced the tables plus the analysis that reads them back.
#
# PORTABILITY. Three things in this tree are ours and cannot travel: the interpreter path,
# the g2/g4 GPU brokers, and MPS. They are patched out below by exact-literal replacement --
# `patch_file` FAILS if a block it expects is not found verbatim, so a drift in the source
# stops the export instead of silently shipping a half-patched launcher.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
OUT="${OUT:-$REPO/release}"
SRC="$REPO/documentation/release"          # hand-written public docs + public-only scripts
PY="${PY:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10}"

RESULTS=1; [[ "${1:-}" == "--no-results" ]] && RESULTS=0

# The tags whose per-cell results ship with the code. They are also the only paths under
# artifacts/runs/ that the exported .gitignore re-includes, so a tag the reader produces is
# untracked by default and cannot be committed by accident.
TAGS=(zn_main zn_a1 zn_a2 zn_enc zn_ot zn_es zn_a2s2_ctrl zn_a2s2_tau64 zn_a2s2_lep
      zn_cbfa_tau64 zn_fedtokcp zn_170_ctfp c50_local c50_central c50_a2 c50_tau64)
# zn_floor is whitelisted but NOT copied wholesale: of its 193 head/mode combinations only the
# moving-average head is the floor the paper reports, and it is copied on its own further down.
TRACKED_TAGS=("${TAGS[@]}" zn_floor)

say() { echo "[make_release] $*"; }
copy() { mkdir -p "$OUT/$(dirname "$1")"; cp -p "$REPO/$1" "$OUT/$1"; }

rm -rf "$OUT"
mkdir -p "$OUT"

# ── 1. the pipeline, verbatim ────────────────────────────────────────────────────────
# Computed as the import closure of pipeline/federated_eval.py + the analysis entry points,
# not by hand: see scripts/make_release.sh history. Anything not reachable from an entry
# point below is not shipped.
CODE=(
  config.py data.py utils.py metrics_core.py LICENSE
  model/__init__.py model/common.py model/transforms.py model/encoder.py
  model/vector_quantizer.py model/decoder.py model/prior.py model/prior_upstream.py
  pipeline/stage1.py pipeline/stage2.py pipeline/detect.py
  pipeline/federated.py pipeline/federated_eval.py pipeline/cf_eval.py
  metrics/__init__.py metrics/affiliation_local.py metrics/vus_local.py metrics/pate_local.py
  lib/__init__.py lib/proctitle.py lib/profiling.py lib/train_viz.py
  preprocessing/datasets/__init__.py preprocessing/datasets/toy_fed.py
  preprocessing/dataset/__init__.py
  preprocessing/UCR_anomaly_dataset_periods.csv
)
# scripts/ — the four families a reader needs: build the data, launch a cohort, analyse the
# output, and prove the aggregation math still holds.
SCRIPTS=(
  cohort.py                                        # pin WHICH data a run covers
  build_ucr_split.py check_ucr_split.py            # the benchmark build + its verifier
  build_toy_fed_uni.py _toy_common.py ensure_dataset.py list_entities.py
  floor_eval.py floor_heads.py                     # the parameter-free floor
  latent_probe.py                                  # kappa: do the clients tokenize alike?
  fusion_probe.py                                  # score fusion (configuration (n))
  cf_quality.py cf_figure.py fig_overview.py       # counterfactual quality + the two figures
  cf_explainable.py                                # the label-free counterfactual of Fig. 1(c)
  paper2_numbers.py                                # re-derives and CHECKS the paper's numbers
  fed_codebook_unittest.py fed_cb_server_ema_unittest.py fed_enc_algo_unittest.py
  fed_regression_unittest.py fed_val_selection_unittest.py
)
for f in "${CODE[@]}"; do copy "$f"; done
for f in "${SCRIPTS[@]}"; do copy "scripts/$f"; done
copy cohorts/ucr2p_10.json
copy cohorts/c50.json
say "code: ${#CODE[@]} files + ${#SCRIPTS[@]} scripts + 2 cohorts"

# ── 2. the two shell drivers, patched ────────────────────────────────────────────────
mkdir -p "$OUT/scripts"
"$PY" - "$REPO/scripts/launch.sh" "$OUT/scripts/launch.sh" <<'PYEOF' || exit 3
import sys
src, dst = sys.argv[1], sys.argv[2]
t = open(src, encoding="utf-8").read()

def sub(old, new, why):
    global t
    n = t.count(old)
    if n != 1:
        raise SystemExit(f"launch.sh: expected 1 occurrence of [{why}], found {n}. "
                         "The source drifted -- re-check the patch before exporting.")
    t = t.replace(old, new)

# (1) the interpreter
sub('PY="${PY:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10}"',
    'PY="${PY:-python3}"', "interpreter path")

# (2) MPS is a lab decision (a shared 20-user host), and the private pipe dir is ours
sub("""# Private MPS pipe dir. Never the compiled-in /tmp/nvidia-mps (0777, 0666 control socket):
# a server death there wedged 14 jobs for 9.9 h with no error. See scripts/mps_ctl.sh.
MPS_CTL_ENV_ONLY=1 source "$REPO/scripts/mps_ctl.sh"
""", "", "mps_ctl source")

# (3) the fp16 pin: the reasoning is load-bearing for the results, the hardware notes are ours
old_amp = t[t.index("# ── PRECISIONE FISSATA"):t.index('export FEDVQ_AMP=')]
t = t.replace(old_amp, """# ── PRECISION IS PINNED, NOT AUTO ───────────────────────────────────────────────
# `_amp_dtype()` reads FEDVQ_AMP; with `auto` it picks by compute capability (fp16 below
# Ampere, bf16 from Ampere up), so the arithmetic of a run would depend on WHICH host
# picked up the job. That is not hygiene, it is load-bearing: on ucr_001 (centralized,
# W=2P) an fp32 ablation moves AUPRC 0.646 -> 0.418 and top-1 1.00 -> 0.40, larger than
# every effect this study measures.
#
# fp16 rather than bf16 because bf16 is emulated on Turing (measured on our sm_75 node,
# matmul 4096^3: fp32 22.9 ms / fp16 3.4 ms / bf16 40.2 ms) while fp16 tensor cores exist
# on Turing, Ampere and Ada alike. The delicate parts stay fp32 regardless: autocast is
# disabled inside the quantizer while the VQ sufficient statistics are accumulated (so the
# exact-pooled-M-step property is preserved), and evaluation and metrics are fp32.
""")

# (4) GPU selection: ours is a broker for two shared department machines
start = t.index('# Override with LAUNCH_GPUS="4 5"')
end = t.index('SLOTS_PER_GPU="${SLOTS_PER_GPU:-7}"')
t = t[:start] + """# WHICH GPUs. One job per slot, SLOTS_PER_GPU slots per card, dispatched round-robin.
#   LAUNCH_GPUS="0 1"   the cards to use (default: every card nvidia-smi reports)
#   SLOTS_PER_GPU=3     concurrent jobs per card (3 saturates a 24 GB card at these sizes)
# With no CUDA device the list falls back to "0" and torch runs on the CPU -- fine for a
# smoke test, hopeless for a real cohort.
if [[ -n "${LAUNCH_GPUS:-}" ]]; then
  read -r -a GPUS <<< "$LAUNCH_GPUS"
elif command -v nvidia-smi >/dev/null 2>&1; then
  read -r -a GPUS <<< "$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\\n' ' ')"
else
  read -r -a GPUS <<< "0"
fi
""" + t[end:]
sub('SLOTS_PER_GPU="${SLOTS_PER_GPU:-7}"                  # measured optimum under MPS',
    'SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"', "slots default")

# (5) the MPS daemon block
start = t.index("# MPS: the pipe dir is exported above")
end = t.index("# ── the run manifest: what this tag IS")
t = t[:start] + t[end:]

open(dst, "w", encoding="utf-8").write(t)
PYEOF

"$PY" - "$REPO/scripts/smoke_arms.sh" "$OUT/scripts/smoke_arms.sh" <<'PYEOF' || exit 3
import sys
src, dst = sys.argv[1], sys.argv[2]
t = open(src, encoding="utf-8").read()
for old, new, why in [
    ('PY="${PY:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10}"',
     'PY="${PY:-python3}"', "interpreter"),
    ('MPS_CTL_ENV_ONLY=1 source "$REPO/scripts/mps_ctl.sh"\n', "", "mps_ctl source"),
    ('CASES+=("wsd_fed c0 128 14 128")            # real KPIs, tolerance 14\n  ', "",
     "wsd_fed case (that dataset is not public)"),
]:
    if t.count(old) != 1:
        raise SystemExit(f"smoke_arms.sh: [{why}] not found verbatim")
    t = t.replace(old, new)
open(dst, "w", encoding="utf-8").write(t)
PYEOF
chmod +x "$OUT/scripts/launch.sh" "$OUT/scripts/smoke_arms.sh"

# the unit tests: the interpreter path in their usage line, plus the two checks that reach
# outside what the public tree contains
"$PY" - "$OUT" <<'PYEOF' || exit 3
import pathlib, sys
out = pathlib.Path(sys.argv[1])

def edit(rel, pairs):
    p = out / rel
    t = p.read_text(encoding="utf-8")
    for old, new, why in pairs:
        if t.count(old) != 1:
            raise SystemExit(f"{rel}: [{why}] not found verbatim -- source drifted")
        t = t.replace(old, new)
    p.write_text(t, encoding="utf-8")

VENV = "/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10"
edit("scripts/fed_enc_algo_unittest.py", [(VENV, "python3", "interpreter")])

# The Federated-Analytics launcher is a shelved suite of the private tree and is not shipped;
# its three guards are checks about THAT file, not about anything in this repository.
edit("scripts/fed_regression_unittest.py", [
    (VENV, "python3", "interpreter"),
    ("""    fa = (REPO / "scripts" / "launch_all_fa.sh").read_text()
    check("launch_all_fa.sh refuses by default", 'FA_I_KNOW_ITS_SHELVED' in fa)
    check("...and exits non-zero", "exit 2" in fa)
    check("...printing the contamination reason, not just a warning",
          "mixture_eval.py:174" in fa)

""", "", "launch_all_fa guards"),
])

# Two comments point at CLAUDE.md, a working-tree file that is not part of the public repo.
edit("pipeline/federated_eval.py", [
    ("as comparable — the CLAUDE.md fingerprint trap.",
     "as comparable — the incomplete-fingerprint trap.", "fingerprint trap comment"),
])
edit("scripts/fig_overview.py", [
    ("finestra annulla esattamente lo scaler per-entita' (vedi CLAUDE.md).",
     "finestra annulla esattamente lo scaler per-entita'.", "znorm comment"),
])

# cf_quality.py resolves an arm LABEL to a (tag, checkpoint directory) pair through a table that
# hardcodes the tags of our own runs. Outside this tree those tags do not exist, so the public
# copy takes them from the command line, defaulting to the published names.
edit("scripts/cf_quality.py", [
    ("""ARMS = {
    "local":       ("zn_main", "local", 5),
    "centralized": ("zn_main", "centralized", 5),
    "A2":          ("zn_a2", "federated_enc_fedavg_bn-shared_prior-partial", 5),
}""",
     """ARMS = {
    "local":       ("zn_main", "local", 5),
    "centralized": ("zn_main", "centralized", 5),
    "A2":          ("zn_a2", "federated_enc_fedavg_bn-shared_prior-partial", 5),
}


def retag(baselines: str, federated: str) -> None:
    \"\"\"Point the arm table at YOUR tags. `local` and `centralized` come from one run of the
    baselines, the federated arm from another, which is how the campaign was laid out.\"\"\"
    for label in ("local", "centralized"):
        tag, arm_dir, n = ARMS[label]
        ARMS[label] = (baselines, arm_dir, n)
    tag, arm_dir, n = ARMS["A2"]
    ARMS["A2"] = (federated, arm_dir, n)""", "arm table"),
    ('    ap.add_argument("--series", required=True, help="lista separata da virgole")',
     '    ap.add_argument("--series", required=True, help="lista separata da virgole")\n'
     '    ap.add_argument("--tag-baselines", default="zn_main",\n'
     '                    help="tag holding the `local` and `centralized` runs")\n'
     '    ap.add_argument("--tag-federated", default="zn_a2",\n'
     '                    help="tag holding the federated run scored as arm A2")', "tag flags"),
    ("    a = ap.parse_args()\n",
     "    a = ap.parse_args()\n    retag(a.tag_baselines, a.tag_federated)\n", "retag call"),
])

# The val-selection test ran on a private KPI dataset. Same test, same two-client federation,
# on the synthetic dataset this repository can generate (scripts/build_toy_fed_uni.py) -- what
# it pins is the round-selection logic, which does not depend on which data it sees.
edit("scripts/fed_val_selection_unittest.py", [
    ('DATASET = "wsd_fed"\nCLIENTS = ["kpi_012", "kpi_036"]        '
     '# the two smallest clients: fast, both in c2',
     'DATASET = "toy_fed_uni"\nCLIENTS = ["uni_00", "uni_01"]           '
     '# two clients of one synthetic cluster\n'
     '# (generate the data first: python scripts/build_toy_fed_uni.py)',
     "private dataset"),
])
PYEOF
# the framework diagram of Fig. 1: a standalone matplotlib script, renamed to sit next to the
# other figure generators instead of under the paper's style directory
"$PY" - "$REPO/documentation/framework_style/a2_framework.py" "$OUT/scripts/fig_framework.py" <<'PYEOF' || exit 3
import sys
t = open(sys.argv[1], encoding="utf-8").read().replace(
    "/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10 a2_framework.py",
    "python3 scripts/fig_framework.py")
open(sys.argv[2], "w", encoding="utf-8").write(t)
PYEOF
say "drivers: launch.sh + smoke_arms.sh patched (interpreter, GPU brokers, MPS)"

# ── 3. hand-written public files ─────────────────────────────────────────────────────
cp -p "$SRC/README.md"        "$OUT/README.md"
cp -p "$SRC/gitignore"        "$OUT/.gitignore"
# the published tags, re-included one by one (see the note in the file)
for t in "${TRACKED_TAGS[@]}"; do echo "!artifacts/runs/$t/"; done >> "$OUT/.gitignore"
# NOT the repo-root requirements.txt: that one is a 154-package freeze of a Windows env, Intel
# runtime and jupyter stack included. The public one lists what the shipped code imports, at the
# versions that produced the runs.
cp -p "$SRC/requirements.txt" "$OUT/requirements.txt"
mkdir -p "$OUT/docs"
for d in "$SRC"/docs/*.md; do cp -p "$d" "$OUT/docs/$(basename "$d")"; done
for s in "$SRC"/scripts/*; do cp -p "$s" "$OUT/scripts/$(basename "$s")"; done
chmod +x "$OUT"/scripts/*.sh
say "docs: README + $(ls "$SRC"/docs/*.md | wc -l) pages + $(ls "$SRC"/scripts | wc -l) public-only scripts"

# ── 4. the result bundle ─────────────────────────────────────────────────────────────
# Tables II-V are read back from these files by scripts/paper2_numbers.py and
# scripts/c50_table.py. Shipped: the per-cell summary json, the per-client report.json, and
# the score profiles of the `local` arm (the input to the fusion analysis). NOT shipped:
# checkpoints and token caches, which are 54 GB and reproduce from the summaries anyway.
if [[ $RESULTS -eq 1 ]]; then
  n=0
  for t in "${TAGS[@]}"; do
    [[ -d "artifacts/runs/$t" ]] || { say "!! missing tag $t -- results incomplete"; continue; }
    # two passes on purpose: -maxdepth is a global option, so a single `A -o B` would clamp the
    # deep report.json branch to depth 2 as well and silently ship only the summaries.
    while IFS= read -r f; do cp -p --parents "$f" "$OUT/"; n=$((n+1)); done < <(
      find "artifacts/runs/$t" -maxdepth 2 -name '*.json')
    while IFS= read -r f; do cp -p --parents "$f" "$OUT/"; n=$((n+1)); done < <(
      find "artifacts/runs/$t/ckpt" -name 'report.json' 2>/dev/null)
  done
  # the fusion inputs: five per-timestep test profiles per development series
  while IFS= read -r f; do cp -p --parents "$f" "$OUT/"; n=$((n+1)); done < <(
    find artifacts/runs/zn_main/ckpt/ucr_split_w2p -path '*/local/*' -name 'scores.npz')
  # the floor, moving-average head only (the one the paper reports)
  for f in artifacts/runs/zn_floor/floor/*floor_ma*.json artifacts/runs/zn_floor/PROVENANCE.md; do
    [[ -e "$f" ]] && { cp -p --parents "$f" "$OUT/"; n=$((n+1)); }
  done
  # Provenance fields inside the results carry the absolute path of the machine that produced
  # them (`"path": "/home/.../detect_score_cache.npz"`, `--resume-from /home/...` in a RUN.json).
  # Rewritten to repo-relative, which is both portable and one fewer thing to leak. Every file
  # is re-parsed afterwards: a rewrite that breaks the JSON must stop the export.
  "$PY" - "$OUT" "$REPO/" <<'PYEOF' || exit 3
import json, pathlib, sys
out, prefix = pathlib.Path(sys.argv[1]), sys.argv[2]
touched = 0
for p in (out / "artifacts").rglob("*.json"):
    t = p.read_text(encoding="utf-8")
    if prefix not in t:
        continue
    new = t.replace(prefix, "")
    json.loads(new)                      # must still parse, or the export is wrong
    p.write_text(new, encoding="utf-8")
    touched += 1
print(f"[make_release] results: {touched} json files made path-relative")
PYEOF
  say "results: $n files, $(du -sh "$OUT/artifacts" | cut -f1)"
else
  say "results: skipped (--no-results)"
fi

# ── 5. a tree that is importable from the start ──────────────────────────────────────
mkdir -p "$OUT/data/raw" "$OUT/preprocessing/dataset"
cat > "$OUT/preprocessing/dataset/PLACE_THE_UCR_ARCHIVE_HERE.md" <<'EOF'
Unzip the UCR Time Series Anomaly Archive here, so that this path exists:

    preprocessing/dataset/AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData/

Archive: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip
Then build the federated benchmark with `scripts/build_ucr_split.py` (see docs/REPRODUCE.md).
EOF

say "done -> $OUT  ($(du -sh "$OUT" | cut -f1), $(find "$OUT" -type f | wc -l) files)"
