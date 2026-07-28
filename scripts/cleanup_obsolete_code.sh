#!/usr/bin/env bash
# cleanup_obsolete_code.sh — delete obsolete SOURCE files (Tier 1 + Tier 2 of the
# 2026-07-23 codebase audit). Companion to cleanup_stale_20260723.sh (artifacts).
#
# Rationale: the repo is UNIVARIATE-ONLY (the -M multivariate paper is closed) and the
# baseline-comparison harness (comparisons/) is gone. Everything below serves the closed
# MV study, an unused benchmark corpus, or a consumed one-off.
#
# NOTE: almost every target is git-tracked, so deletion is RECOVERABLE with
#   git checkout -- <path>      (or `git restore <path>`)
#
# NOT touched (needed for the re-runs): the 13 analysis scripts whose default input paths
# point at deleted artifacts (aggregate_all, cross_reduce, fewshot, onboard_reduce,
# local_recon_autopsy, codebook_recon_probe, token_partition, sweep_window_width,
# retrain_*, plot_retrain_train_test, plot_recon_width_c3) — they need repointing, not deletion.
#
# Usage:  bash scripts/cleanup_obsolete_code.sh          # dry-run
#         bash scripts/cleanup_obsolete_code.sh --yes    # delete
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
YES=0; [ "${1:-}" = "--yes" ] && YES=1

DIRS=(
  scripts/legacy                          # 52 MV-era downloaders / toy builders / one-offs
  documentation/archive                   # 9 pre-federated MV docs
  texpdfpaper                             # the CLOSED -M paper (main.tex has zero 'federated')
  figures                                 # entire fig set: only consumer was texpdfpaper/main.tex;
                                          # the live -U paper has no \includegraphics at all
  figures_mod                             # unreferenced MV-era composed diagram
  dataset-federated/data/KPI-AIOps        # 475M  "optional extra domain", never wired
  dataset-federated/data/NAB              # 315M  corpus + 89 third-party detector .py, unused
  dataset-federated/data/LEAD             # 65M   "optional extra domain", never wired
  dataset-federated/data/Yahoo-S5         # empty placeholder
  dataset-federated/data/WSD/synthetic    # 257M  build_frozen.py reads only WSD/real-world
)

FILES=(
  run_matrix.py                           # cross-model scheduler for CATCH/InterFusion/OmniAnomaly
  run_all_parallel.sh                     #   -> comparisons/ deleted, envs absent: orphaned pair
  data_integrity_manifest.json            # 14M, lists 40 MV datasets, none of the 8 live ones
  federated_method.pdf                    # stale, not rebuildable (.tex has no \documentclass)
  pipeline/cf_channel.py                  # per-CHANNEL counterfactual: MV-only, nothing imports it
  pipeline/compare_aggregations.py        # channel-aggregation: self-disables at C=1 (run.py updated)
  preprocessing/preprocess.py             # dead pair, superseded by data.py/make_dataloaders
  preprocessing/data_pipeline.py          #   (only consumer of preprocess.py; nothing calls it)
  scripts/build_toy_fed.py                # MV predecessor (C=8) of build_toy_fed_uni.py
  scripts/archive_stale_artifacts.py      # created artifacts/fed_eval/_old — now deleted
  scripts/cleanup_stale_20260723.sh       # consumed one-off (already executed)
  scripts/run_fed_clusters.sh             # superseded by run_fed_converged / run_converged_all
  scripts/flare_eval.py                   # Group-7 analytic head: ledger §6 "DEAD, do not re-open"
  scripts/flare_via.py                    #   "
  scripts/prism_eval.py                   #   "
  scripts/halo_eval.py                    #   "
  scripts/analytic_common.py              # library used ONLY by the four Group-7 scripts
  scripts/download_smap_msl.py            # MV benchmark corpora, absent from data/raw
  scripts/download_asd.py                 #   "
  documentation/DEAD_AND_OPEN_LINES.md    # superseded by RESEARCH_LEDGER, then by PROJECT_converged
  documentation/RESEARCH_PLAN.md          #   "  (its spine is the closed MV toy_fed C=8 line)
  documentation/WORKSHOP_PAPER_PLAN.md    #   "  (steps done / target the deleted MV assets)
)

# scripts/build_toy_*_channel_anomalies.py — the 9 MV channel-anomaly builders
mapfile -t CHANNEL_BUILDERS < <(ls scripts/build_toy_*_channel_anomalies.py 2>/dev/null)
# preprocessing/datasets/*.py — ALL dead except the single live loader.
# data.py:985 routes every live dataset (toy_fed*, wsd_fed, ucr_pool, ucr_ad) to toy_fed.py.
mapfile -t DEAD_LOADERS < <(ls preprocessing/datasets/*.py 2>/dev/null | grep -vE '/(toy_fed|__init__)\.py$')

echo "=== targets ==="
for d in "${DIRS[@]}";  do [ -e "$d" ] && printf "  %-8s %s\n" "$(du -sh "$d" 2>/dev/null|cut -f1)" "$d" || printf "  (absent) %s\n" "$d"; done
for f in "${FILES[@]}"; do [ -e "$f" ] && printf "  %-8s %s\n" "$(du -sh "$f" 2>/dev/null|cut -f1)" "$f" || printf "  (absent) %s\n" "$f"; done
printf "  %-8s %s\n" "${#CHANNEL_BUILDERS[@]} files" "scripts/build_toy_*_channel_anomalies.py"
printf "  %-8s %s\n" "${#DEAD_LOADERS[@]} files" "preprocessing/datasets/*.py  (keeping toy_fed.py + __init__.py)"

if [ "$YES" -ne 1 ]; then echo; echo "DRY-RUN. Re-run with --yes to delete."; exit 0; fi

echo; echo "=== deleting ==="
for d in "${DIRS[@]}";  do [ -e "$d" ] && { rm -rf "$d" && echo "  rm -rf $d"; }; done
for f in "${FILES[@]}"; do [ -e "$f" ] && { rm -f  "$f" && echo "  rm $f"; }; done
[ "${#CHANNEL_BUILDERS[@]}" -gt 0 ] && { rm -f "${CHANNEL_BUILDERS[@]}"; echo "  rm ${#CHANNEL_BUILDERS[@]} channel-anomaly builders"; }
[ "${#DEAD_LOADERS[@]}" -gt 0 ]     && { rm -f "${DEAD_LOADERS[@]}";     echo "  rm ${#DEAD_LOADERS[@]} dead dataset loaders"; }
find preprocessing -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
echo "  pruned preprocessing/__pycache__"

echo; echo "=== done. repo size: $(du -sh "$REPO" 2>/dev/null | cut -f1) ==="
echo "FOLLOW-UPS:"
echo "  * data.py still has lazy-import branches for the deleted loaders (dead code, never"
echo "    reached — every live dataset routes to toy_fed.py at data.py:985). Prune if desired."
echo "  * scripts/ensure_dataset.py still dispatches ^toy_.+_channel_anomalies\$ to the removed builders."
echo "  * git: these were tracked — commit the deletions, or 'git restore <path>' to undo."
