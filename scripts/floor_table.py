#!/usr/bin/env python3
"""FLOOR vs every deep arm — the client × model × metric matrix.

THIS is the producer of `artifacts/floor/<ds>_matrix.json`, which used to be an
orphan file with no script behind it. Run it after `scripts/run_floor.sh`.

Sources, in order of richness:
  * FLOOR      — artifacts/floor/records_<ds>.jsonl (every arm on disk)
  * deep, full — artifacts/<tree>/ckpt/<ds>/<cluster>/seed*/<arm>/<entity>/report.json
                 carries the COMPLETE detect() metric set (paper top-1/3/5,
                 event_*, best_f1, pate, vus_roc, delay, …)
  * deep, flat — artifacts/<tree>/**/*.json {records:[...]} — only the 6 keys
                 federated_eval.py:58 writes. `converged_all` (local /
                 centralized) has NO report.json, so its extended columns are
                 legitimately blank rather than zero.

Comparability, in one line: only the THRESHOLD-FREE block is comparable between
FLOOR and the deep arms. The deep arms threshold with the paper's per-τ rule,
FLOOR necessarily with the train-quantile fallback, so `f1`, `precision`,
`recall`, `fpr`, `event_*`, `affiliation_*`, `pate` and `detection_delay_mean`
are appendix material with that caveat in the caption.

The default --records-dir is artifacts/floor, which was ARCHIVED on 2026-07-29 and is empty:
on a cohort repo the floor records live one directory per tag, so the working invocation is
the second one below. A --records-dir that yields no floor row is a hard error (deep rows
alone still make a full-looking table), overridable with --allow-no-floor.

    python scripts/floor_table.py --dataset wsd_fed
    python scripts/floor_table.py --dataset wsd_fed --records-dir "artifacts/runs/*/floor"
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# Threshold-FREE — comparable across FLOOR and deep.
FREE = ["vus_pr", "auprc", "auroc", "pate_f1", "best_f1", "vus_roc"]
# Threshold-DEPENDENT — NOT comparable (different threshold rule).
THRESHOLDED = ["f1", "precision", "recall", "fpr", "affiliation_f1",
               "affiliation_precision", "affiliation_recall", "pate",
               "event_f1", "event_precision", "event_recall",
               "detection_delay_mean"]

# Runs made under the cohort system (2026-07-29 onward) live in artifacts/runs/<tag>/ and are
# the only ones whose cohort, window and tolerance are pinned. The pre-cohort trees were moved
# to artifacts/_archive_20260729/ and are kept READABLE but marked `archived`: their cluster
# set and window were never recorded, so they are not matched against anything.
TREES = {"runs": "artifacts/runs/*/{ds}/*.json",
         "converged_all": "artifacts/_archive_20260729/converged_all/{ds_short}_*.json",
         "converge60": "artifacts/_archive_20260729/converge60/**/*.json",
         "ucrsplit": "artifacts/_archive_20260729/ucrsplit/*.json",
         "fed_eval": "artifacts/_archive_20260729/fed_eval/{ds}/**/*.json"}
TRUST = {"floor": "ok", "runs": "ok",
         "converged_all": "archived", "converge60": "archived",
         "ucrsplit": "archived", "fed_eval": "archived-stale"}
KIND = {"local": "skyline", "centralized": "skyline"}

# Where the RICH deep metrics live. `report.json` carries the complete detect() set
# (paper top-1/3/5, event_*, best_f1, vus_roc, delay); the flat jsons in TREES carry
# only the 6 keys federated_eval.py:58 writes.
#
# These globs are SEPARATE from TREES on purpose. The previous version built them as
# f"artifacts/{tree}/ckpt/{ds}/..." from the dict KEY, so it looked for
# `artifacts/converge60/ckpt/...` (moved under _archive_20260729/ on 2026-07-29) and
# `artifacts/runs/ckpt/...` (the cohort layout is artifacts/runs/<tag>/ckpt/...).
# Every pattern missed, deep_reports() returned {} for every dataset, and the
# extended columns came out blank — indistinguishable from "the arm never wrote them".
#
# The three archived trees do not even share a layout: converge60 is
# ckpt/<ds>/<cluster>/seed*/..., ucrsplit is ckpt/<cluster>/seed*/... with no dataset
# level at all, and fed_eval has no ckpt level. So the depth is globbed and the
# dataset is filtered on report.json's own `dataset_name`, never on the path.
REPORT_GLOBS = {
    "runs":       "artifacts/runs/*/ckpt/**/report.json",
    "converge60": "artifacts/_archive_20260729/converge60/ckpt/**/report.json",
    "ucrsplit":   "artifacts/_archive_20260729/ucrsplit/ckpt/**/report.json",
    "fed_eval":   "artifacts/_archive_20260729/fed_eval/**/report.json",
}
# DELIBERATELY ABSENT: artifacts/_archive_20260729/ucrsplit_w2p_cb64_partial/.
# Those 39 report.json files were produced by the interrupted 2*period sweep (it stopped
# at 6 of 456 jobs and was never brought under the cohort system). They say
# `dataset_name: "ucr_split"` and `buffer: 64` — byte-for-byte the same self-description
# as the 1395 files in `ucrsplit`, which are at W=128. report.json records NO window, so
# once read there is no way to tell a W=128 row from a W=2*period one, and surfacing both
# in the ucr_split table would build rule 10 / §8.10 (never compare across windows)
# straight into the CSV. Excluded until something records the window; a tree that cannot
# be labelled honestly is worth less than no tree.

# Which trees are SHARDED BY COHORT TAG: `launch.sh` writes every deep run under
# artifacts/runs/<tag>/ckpt/..., one directory per tag, so the tag is the first path
# component below this root. The archived trees predate the cohort system (2026-07-29) and
# have no tag level at all — they map to "". report.json itself records NO tag, so the path
# is the only place it can be read from.
TAG_ROOTS = {"runs": "artifacts/runs"}


def rel_repo(p: Path) -> str:
    """Path for display, never for control flow. `Path.relative_to` RAISES when the path is
    outside the repo, and both rule 2 ("shard => distinct --out-dir") and §8 ("always a
    separate --out-dir for ablations") tell you to point --out-dir wherever you like. The
    unguarded call made every out-of-repo run exit 1 *after* writing the files: the CSV was on
    disk, the exit code said failure, and a shell chain stopped."""
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def report_tag(tree: str, path: str) -> str:
    """Cohort tag owning a report.json; "" for the trees that have no tag level."""
    root = TAG_ROOTS.get(tree)
    if root is None:
        return ""
    try:
        rel = Path(path).relative_to(REPO / root)
    except ValueError:
        return ""
    return rel.parts[0] if len(rel.parts) > 1 else ""


def deep_reports(ds: str) -> dict:
    """(tree, tag, arm, cluster, entity, seed) -> full report.json dict.

    The TAG belongs in the key and must stay there. The `runs` glob is
    artifacts/runs/*/ckpt/**/report.json, so the tag level is swallowed by the `*` and every
    tag used to collapse onto tree="runs": two tags covering the same (cluster, arm, entity,
    seed) OVERWROTE each other, and which one survived depended on the unordered glob.glob
    order. Demonstrated on a synthetic tree — two tags with vus_pr 0.90 and 0.10 on the same
    key returned ONE row at 0.10, and the 0.90 disappeared without a line of log.

    The flat branch in collect() has always ACCUMULATED its duplicates and let main()'s
    seed-collapse average them, so the two ingestion paths behaved in OPPOSITE ways on the
    same data. With the tag in the key both rows survive and the deep branch merges tags the
    way the floor branch already merges `--records-dir "artifacts/runs/*/floor"`."""
    out = {}
    for tree, pat in REPORT_GLOBS.items():
        for p in glob.glob(str(REPO / pat), recursive=True):
            parts = Path(p).parts
            if len(parts) < 6:
                continue
            cluster, seed, arm, entity = parts[-5], parts[-4], parts[-3], parts[-2]
            if not seed.startswith("seed"):
                continue          # not the <cluster>/seed*/<arm>/<entity> layout
            try:
                rep = json.load(open(p))
            except Exception:
                continue
            if rep.get("dataset_name") and rep["dataset_name"] != ds:
                continue
            out[(tree, report_tag(tree, p), arm, cluster, entity, seed)] = rep
    return out


def cohort_fingerprints() -> dict:
    """tag -> cohort_fingerprint. Hard rule (LAUNCH_RUNBOOK §6): equal fingerprints
    means comparable, unequal means NOT comparable, and no post-processing fixes it.
    Printed so a table can never quietly mix two cohorts."""
    out = {}
    for p in sorted(glob.glob(str(REPO / "artifacts/runs/*/RUN.json"))):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        out[r.get("tag") or Path(p).parent.name] = r.get("cohort_fingerprint", "?")
    return out


def floor_record_paths(ds: str, spec: str) -> list[Path]:
    """Every records_<ds>.jsonl the spec resolves to. Comma-separated, globs allowed.

    This is also the MERGE step: `launch.sh --engine floor` writes into
    artifacts/runs/<tag>/floor/, one directory per tag, while a paper table needs
    every batch at once. `--records-dir "artifacts/runs/*/floor"` reads them all and
    applies floor_eval's own (arm, entity) de-duplication across them, so batches
    split across tags behave exactly like batches split across invocations."""
    out: list[Path] = []
    for part in (s.strip() for s in spec.split(",") if s.strip()):
        base = Path(part)
        pat = str(base if base.is_absolute() else REPO / base)
        for d in sorted(glob.glob(pat)):
            p = Path(d) / f"records_{ds}.jsonl"
            if p.exists():
                out.append(p)
    return out


def collect(ds: str, records_spec: str) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    keys: set[str] = set()

    # ── FLOOR ───────────────────────────────────────────────────────────────
    # De-duplicate on (arm, entity) with the same rule floor_eval.dedupe() uses:
    # last write wins, EXCEPT that a row without the nested `_suite` block never
    # replaces one that has it.
    floor_by: dict = {}
    provenance: dict = {}
    for rec in floor_record_paths(ds, records_spec):
        n_here = 0
        for line in rec.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                       # torn line from a shared out-dir
            key = (r["_arm"], r["_cluster"], r["_entity"])
            prev = floor_by.get(key)
            if prev is not None and "_suite" in prev[0] and "_suite" not in r:
                r = {**r, "_suite": prev[0]["_suite"]}
            floor_by[key] = (r, rec)
            n_here += 1
        try:
            label = str(rec.parent.relative_to(REPO))
        except ValueError:
            label = str(rec.parent)            # a records dir outside the repo
        provenance[label] = n_here
    for (_arm, _cl, _ent), (r, _src) in sorted(floor_by.items()):
        m = {k: v for k, v in r.items()
             if not k.startswith("_") and isinstance(v, (int, float))}
        keys |= set(m)
        # `arm` is the storage identity and carries `__w<W>`; `model` is the ANALYSIS identity
        # and does not. On ucr_split_w2p the window is a per-series property (79 distinct
        # values over 180 clusters), so grouping a table by `arm` there splits ONE model into
        # 79 rows of ~5 entities and no row has an n a paired test can use. Older records have
        # no `_model`, so it is derived the same way floor_eval.model_tag does.
        rows.append({"tree": "floor", "trust": "ok", "arm": r["_arm"],
                     "model": r.get("_model") or re.sub(r"__w\d+", "", r["_arm"]),
                     "window": r.get("_window"),
                     "cluster": r["_cluster"], "entity": r["_entity"],
                     "seed": r.get("_seed", 0), **m})
    if provenance:
        print("[floor_table] records read: "
              + ", ".join(f"{k} ({v})" for k, v in provenance.items()))

    # ── deep: full report.json first ────────────────────────────────────────
    full = deep_reports(ds)
    # A key present under MORE THAN ONE cohort tag is now kept once per tag and averaged by
    # the seed-collapse in main(), which is the right thing for a sweep SHARDED over tags —
    # the floor branch above merges tags for exactly that reason. But averaging is only
    # legitimate when the tags share a cohort_fingerprint (LAUNCH_RUNBOOK §6), so name the
    # tags here rather than let a mean appear from nowhere: main() prints the fingerprints
    # at the end and the reader has to check them.
    tags_of: dict = collections.defaultdict(set)
    for (tree, tag, arm, cluster, entity, seed) in full:
        tags_of[(tree, arm, cluster, entity, seed)].add(tag)
    n_multi = sum(1 for v in tags_of.values() if len(v) > 1)
    if n_multi:
        combos = sorted({"+".join(sorted(v)) for v in tags_of.values() if len(v) > 1})
        print(f"[floor_table] {n_multi} deep key(s) exist under MORE THAN ONE cohort tag and "
              f"are AVERAGED together: {', '.join(combos)}\n"
              f"  that is a merge, not a duplicate — check the cohort fingerprints printed "
              f"below before quoting the mean.")
    for (tree, tag, arm, cluster, entity, seed), rep in full.items():
        m = {k: v for k, v in rep.items()
             if isinstance(v, (int, float)) and not isinstance(v, bool)
             and k not in ("seed", "entity_count", "buffer", "threshold",
                           "threshold_q", "n_pos_labels", "n_neg_labels",
                           "use_impulse_term")}
        keys |= set(m)
        rows.append({"tree": tree, "trust": TRUST.get(tree, "archived"), "arm": arm,
                     "model": arm, "window": None, "tag": tag,
                     "cluster": cluster, "entity": entity,
                     "seed": int(str(seed).replace("seed", "") or 0), **m})

    # ── deep: flat records for arms with no report.json ─────────────────────
    # The ckpt tree names arms WITH their knobs (`federated_enc_fedprox_mu0.1`)
    # while the flat json names them without (`federated_enc_fedprox`). Matching
    # on the prefix keeps the same arm from being counted twice under two names.
    have = collections.defaultdict(set)
    for (t, _tag, a, c, e, _s) in full:
        have[(t, c, e)].add(a)          # tag-agnostic on purpose: one report.json under ANY
                                        # tag is enough to prefer the rich metrics over flat
    ds_short = ds.split("_")[0]
    for tree, pat in TREES.items():
        for f in glob.glob(str(REPO / pat.format(ds=ds, ds_short=ds_short)), recursive=True):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            # These trees are multi-dataset: converge60 holds toy_* clusters too,
            # and an unfiltered glob silently mixes 47 toy clients into an n=31
            # wsd table.
            meta = d.get("meta") or {}
            if meta.get("dataset") and meta["dataset"] != ds:
                continue
            meta_cl = meta.get("cluster")           # converged_all has no _cluster in rows
            for r in d.get("records", []) or []:
                ent, arm = r.get("_entity"), r.get("_arm")
                cl = r.get("_cluster") or meta_cl
                if ent is None or arm is None or cl is None:
                    continue                       # unlabelled row: not placeable
                if any(a == arm or a.startswith(arm + "_") for a in have[(tree, cl, ent)]):
                    continue
                m = {k: v for k, v in r.items()
                     if not k.startswith("_") and isinstance(v, (int, float))}
                keys |= set(m)
                rows.append({"tree": tree, "trust": TRUST[tree], "arm": arm,
                             "model": arm, "window": None,
                             "cluster": cl, "entity": ent, "seed": r.get("_seed", 0), **m})

    top = sorted(k for k in keys if k.startswith("paper_top"))
    ordered = ([k for k in FREE if k in keys] + top
               + [k for k in THRESHOLDED if k in keys]
               + sorted(k for k in keys
                        if k not in FREE + THRESHOLDED + top and not k.endswith("_macro")))
    return rows, ordered


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--metric", default="vus_pr", help="metric shown in the printed table")
    ap.add_argument("--max-entities", type=int, default=200,
                    help="skip the matrix json above this (ucr_split has 1130)")
    ap.add_argument("--records-dir", default="artifacts/floor",
                    help="comma-separated dirs (globs allowed) holding records_<ds>.jsonl. "
                         'Use "artifacts/runs/*/floor" to merge every cohort tag.')
    ap.add_argument("--out-dir", default=None,
                    help="where the CSV and matrix json go (default: the first records dir)")
    ap.add_argument("--allow-no-floor", action="store_true",
                    help="write the deep-only table even when --records-dir yields no floor "
                         "row. Default is to REFUSE: a floor table with no floor in it is "
                         "half a result that reads like a whole one.")
    args = ap.parse_args()
    ds = args.dataset

    rows, metrics = collect(ds, args.records_dir)
    if not rows:
        raise SystemExit(
            f"no rows for {ds}\n"
            f"  --records-dir {args.records_dir!r} resolved to "
            f"{[str(p) for p in floor_record_paths(ds, args.records_dir)] or 'NOTHING'}\n"
            f"  run scripts/run_floor.sh (or launch.sh --engine floor) first")

    # HALF the point of this script is the floor row: it is the calibration bar every deep arm
    # is measured against. The "resolved to NOTHING" exit above CANNOT catch a missing floor,
    # because collect() found deep rows and `rows` is non-empty. Measured on 2026-07-30:
    # `--dataset wsd_fed` with the default --records-dir exited 0 printing "434 rows /
    # 14 models / 31 entities" and not one row with tree="floor" — artifacts/floor/ was
    # archived on 2026-07-29 and has been empty since. floor_stats.py then died one step later
    # with "no floor rows in the CSV", and the floor-vs-deep p-values quoted in the
    # documentation are not reproducible with the documented commands for exactly this reason.
    # Fail BEFORE writing anything: a CSV on disk next to a non-zero exit code is the failure
    # mode rel_repo() above already documents.
    n_floor = sum(1 for r in rows if r["tree"] == "floor")
    no_floor_msg = ""
    if not n_floor:
        resolved = [rel_repo(p) for p in floor_record_paths(ds, args.records_dir)]
        no_floor_msg = (
            f"NO FLOOR ROWS for {ds}. --records-dir {args.records_dir!r} resolved to "
            f"{resolved or f'no records_{ds}.jsonl at all'}.\n"
            f"  {len(rows)} deep row(s) WERE found, so the table looks complete while the "
            f"baseline every arm is compared against is missing.\n"
            f"  Cohort floor runs live one directory per tag: rerun with\n"
            f'    --records-dir "artifacts/runs/*/floor"\n'
            f"  or produce them first with scripts/run_floor.sh (launch.sh --engine floor).\n"
            f"  --allow-no-floor writes the deep-only table anyway.")
        if not args.allow_no_floor:
            raise SystemExit(no_floor_msg)
        print("⚠ " + no_floor_msg + "\n")

    # Collapse seeds -> mean per (tree, MODEL, cluster, entity). The key is the MODEL, not
    # the arm: on a per-series-window build one model wears one arm name per distinct window
    # (79 of them on ucr_split_w2p), and keying on the arm would split every model into ~5-entity
    # fragments that no paired test can use. Within one (model, cluster, entity) there is at
    # most one arm anyway, because a series has exactly one window.
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["tree"], r["trust"], r["model"], r["cluster"], r["entity"])].append(r)
    flat = []
    for (tree, trust, model, cl, ent), rs in by.items():
        o = {"tree": tree, "trust": trust, "model": model,
             "arm": sorted({x["arm"] for x in rs})[0], "cluster": cl,
             "entity": ent, "n_seeds": len(rs),
             "window": next((x["window"] for x in rs if x.get("window") is not None), None)}
        for m in metrics:
            v = [x[m] for x in rs if isinstance(x.get(m), (int, float)) and np.isfinite(x[m])]
            o[m] = float(np.mean(v)) if v else None
        flat.append(o)

    # Declare the window heterogeneity rather than hide it: on ucr_split_w2p the accumulation
    # geometry and the impulse-term MA length differ per series, so a model is not a
    # constant-size treatment across clusters. That belongs in the caption.
    wr = {}
    for r in flat:
        if r["tree"] == "floor" and r.get("window") is not None:
            lo, hi = wr.get(r["model"], (r["window"], r["window"]))
            wr[r["model"]] = (min(lo, r["window"]), max(hi, r["window"]))

    # The output directory is NOT guaranteed to exist: artifacts/floor/ was moved to
    # _archive_20260729/ on 2026-07-29, and out_csv.open("w") on a missing parent is a
    # FileNotFoundError that reads like a missing input rather than a missing mkdir.
    spec = args.out_dir or args.records_dir.split(",")[0].strip()
    if any(ch in spec for ch in "*?["):        # a glob: land in artifacts/floor/
        spec = "artifacts/floor"
    out_dir = Path(spec)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"{ds}_all_models.csv"
    cols = ["tree", "trust", "model", "arm", "window", "cluster", "entity",
            "n_seeds"] + metrics
    with out_csv.open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in sorted(flat, key=lambda r: (r["tree"], r["model"], r["cluster"],
                                            r["entity"])):
            fh.write(",".join("" if r.get(c) is None else
                              (f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]))
                              for c in cols) + "\n")

    models = sorted({(r["tree"], r["trust"], r["model"]) for r in flat})
    ents = sorted({(r["cluster"], r["entity"]) for r in flat})

    # ── matrix json (the file that used to have no producer) ────────────────
    if len(ents) <= args.max_entities:
        cell = {(r["tree"], r["model"], r["cluster"], r["entity"]): r for r in flat}
        payload = {
            "_produced_by": "scripts/floor_table.py",
            "_dataset": ds,
            "_note": ("Only the threshold-free block is comparable between FLOOR and the "
                      "deep arms: FLOOR thresholds with the train-quantile fallback, the "
                      "deep arms with the paper per-τ rule. Empty cells mean the metric "
                      "was never written for that arm (converged_all has no report.json), "
                      "never that it is zero."),
            "metrics": metrics,
            "free": [m for m in metrics if m in FREE or m.startswith("paper_top")],
            "_window_range": {m: list(v) for m, v in sorted(wr.items()) if v[0] != v[1]},
            "models": [{"tree": t, "model": a, "arm": a, "trust": tr,
                        "kind": ("baseline" if t == "floor" else
                                 "stale" if tr == "stale" else KIND.get(a, "fed")),
                        "label": (a.replace("floor_", "FLOOR · ") if t == "floor"
                                  else a.replace("federated_", ""))}
                       for t, tr, a in models],
            "entities": [{"cluster": c, "entity": e} for c, e in ents],
            "clusters": [{"name": c, "idx": [i for i, (cc, _e) in enumerate(ents) if cc == c]}
                         for c in sorted({c for c, _ in ents})],
        }
        cells, agg, clagg = {}, {}, {}
        for m in metrics:
            grid = [[(cell.get((t, a, c, e)) or {}).get(m) for t, _tr, a in models]
                    for c, e in ents]
            cells[m] = grid
            agg[m] = []
            for j in range(len(models)):
                v = [grid[i][j] for i in range(len(ents)) if grid[i][j] is not None]
                agg[m].append({"median": float(np.median(v)), "mean": float(np.mean(v)),
                               "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)}
                              if v else {"n": 0})
            clagg[m] = {}
            for c in sorted({c for c, _ in ents}):
                idx = [i for i, (cc, _e) in enumerate(ents) if cc == c]
                clagg[m][c] = [
                    (lambda v: float(np.median(v)) if v else None)(
                        [grid[i][j] for i in idx if grid[i][j] is not None])
                    for j in range(len(models))]
        payload.update({"cells": cells, "agg": agg, "clagg": clagg})
        out_json = out_dir / f"{ds}_matrix.json"
        out_json.write_text(json.dumps(payload, indent=1))
        print(f"{len(flat)} rows · {len(models)} models · {len(ents)} entities · "
              f"{len(metrics)} metrics -> {rel_repo(out_csv)} + {rel_repo(out_json)}\n")
    else:
        print(f"{len(flat)} rows · {len(models)} models · {len(ents)} entities "
              f"(> --max-entities: matrix json skipped) -> {rel_repo(out_csv)}\n")

    # ── printed table ───────────────────────────────────────────────────────
    show = [m for m in metrics if m in FREE or m.startswith("paper_top")][:8]
    hdr = (f'{"tree":14s} {"model":40s} {"n":>4s} '
           + " ".join(f"{m[:11]:>11s}" for m in show) + "  W")
    print(hdr); print("-" * len(hdr))
    for tree, trust, model in models:
        sub = [r for r in flat if (r["tree"], r["model"]) == (tree, model)]
        cells_ = []
        for m in show:
            v = [r[m] for r in sub if r.get(m) is not None]
            cells_.append(f"{np.median(v):11.4f}" if v else f'{"-":>11s}')
        mark = "" if trust == "ok" else "  <PRE-PURGE>"
        lo_hi = wr.get(model)
        wtxt = ("" if lo_hi is None else
                f"  {lo_hi[0]}" if lo_hi[0] == lo_hi[1] else f"  {lo_hi[0]}..{lo_hi[1]}")
        print(f"{tree:14s} {model:40s} {len(sub):4d} " + " ".join(cells_) + wtxt + mark)
    print("\n(medians; threshold-free block only. f1 / affiliation_* / event_* / pate /"
          " precision / recall / fpr are in the CSV but NOT comparable across "
          "floor-vs-deep: different threshold rule.)")
    spread = {m: v for m, v in wr.items() if v[0] != v[1]}
    if spread:
        lo = min(v[0] for v in spread.values()); hi = max(v[1] for v in spread.values())
        print(f"\n⚠ {len(spread)} model(s) span MORE THAN ONE window (W in {lo}..{hi}): this "
              f"build sets W per series.\n  W fixes both the accumulation geometry and the "
              f"impulse-term MA length, so the treatment is NOT of constant size across "
              f"clusters.\n  Say so in the caption; the arm column keeps the per-series "
              f"`__w<W>` tag, the model column is what a paired test is about.")

    # Comparability, stated rather than assumed. Rows from `runs` carry a cohort
    # fingerprint; rows from an archived tree do not, so a floor-vs-deep delta against
    # one of them is not fingerprint-matched and must say so in its caption.
    fps = cohort_fingerprints()
    if fps:
        print("\ncohort fingerprints (equal => comparable; unequal => NOT, and no "
              "post-processing fixes it):")
        for tag, fp in fps.items():
            print(f"  runs/{tag:20s} {fp}")
    archived = sorted({r["tree"] for r in flat if r["trust"] != "ok"})
    if archived:
        print(f"\n⚠ pre-cohort trees present ({', '.join(archived)}): no fingerprint was "
              f"ever recorded for them.\n  Any floor-vs-deep row built on those is NOT "
              f"fingerprint-matched — label it explicitly, do not delete it.")
    # Repeated at the FOOT of the table as well as before the write: the reader who scrolls to
    # the numbers must not be able to read a deep-only table as a floor-vs-deep one. Nothing
    # else in this output says the floor is absent — every deep row still prints normally.
    if no_floor_msg:
        print("\n⚠⚠ " + no_floor_msg + "\n  The table above is DEEP-ONLY: no floor-vs-deep "
              "delta can be computed from this CSV, and floor_stats.py will refuse it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
