"""
=============================================================================
  run.py — LEGACY per-entity launcher (from the multivariate -M line).
=============================================================================

  NOTE (2026-07-10): this repo is now the univariate (-U) federated study. The
  active entry point is `pipeline/federated_eval.py` (per-cluster federated arms
  on toy_fed_uni / wsd_fed). `run.py` and its multivariate `toy_*_channel_anomalies`
  defaults are OFF-PATH here; kept for the legacy per-entity pipeline only.

ONE file (Windows + Linux) that replaces run_linux/*.sh and run_ps/*.ps1.
It runs the canonical per-entity pipeline over any mix of datasets, sequentially
or in parallel across GPUs. Same contract as before: each (dataset, entity) pair
runs the 7-script chain in its own env; per-pipeline logs; one timings.csv.

  THE PIPELINE (single source of truth — edit PIPELINE_SCRIPTS below):
    stage1 -> stage2 -> detect -> quality_stage1
    -> quality_stage2 -> cf_eval

  Each entity = one pipeline = the 7 scripts run sequentially in a subprocess
  whose env carries DATASET_NAME / DATASET_ENTITY / CUDA_VISIBLE_DEVICES /
  thread caps. A non-zero exit aborts THAT pipeline only; the sweep continues.

GRANULARITY — `--datasets` is a comma list, tokens mixed freely:
    all                 the 9 toy_*_channel_anomalies families (per-entity)
    <short>             one toy family by short name: basic, point, level,
                        pattern, amplitude, correlation, group, categorical,
                        morphological
    <dataset>           a real dataset (smap, msl, ...): all its entities
    <dataset>:<entity>  exactly ONE entity (e.g. smap:A-1,
                        toy_point_channel_anomalies:toy_03)

SCALE (toy families only) — `--scale normal|32k|both`:
    normal  toy_<x>_channel_anomalies                       (default)
    32k     toy_<x>_channel_anomalies_32k
    both    run ALL normal pipelines to completion, THEN all 32k (hard barrier)
  Scale is ignored for real datasets and for explicit full names (write the
  _32k name yourself if you want the 32k variant of a specific entity).

Usage (run from the project root, with your env already active):

    # one dataset, specific entities, sequential          (old train.sh)
    python run.py --datasets smap -e A-1,A-2,A-3

    # all MSL entities, 2 GPUs, 2 concurrent
    python run.py --datasets msl -w 2 -g 0,1

    # whole 9-family toy sweep, normal then 32k, 8 concurrent  (old train_channel_all.sh)
    python run.py --datasets all --scale both -w 8 -g 0,1

    # two families only, 32k scale
    python run.py --datasets point,correlation --scale 32k -g 0 -w 4

    # one specific entity
    python run.py --datasets toy_point_channel_anomalies:toy_03

    # CPU only (no GPU)
    python run.py --datasets smap --gpus ''

    # forward extra args / preview without launching
    python run.py --datasets all --scale both --dry-run

The interpreter defaults to `python` (assumes your conda/venv env is active),
overridable with --python or $PIPELINE_PYTHON. This launcher does NOT activate
an env for you — activate it first, exactly as you do for comparisons/run.py.

Notes:
- Per-pipeline stdout+stderr go to `<log-dir>/<dataset>_<entity>.log`; the
  orchestrator's own stdout stays terse (one event line per start/finish).
- Per-script timings are appended to `<log-dir>/timings.csv` (one orchestrator
  process owns all workers, so a threading.Lock serialises writes — no flock).
- A failing script aborts only its own pipeline; the final summary lists every
  failed (dataset, entity) pair with the path to its log file.
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE

# The canonical pipeline. Every pipeline runs EXACTLY this, in order.
#   * detect.py now ALSO emits the per-channel report internally (gated by
#     y_channel, reusing the same scoring pass) — the former detect_per_channel.py
#     was merged into it, so it no longer needs its own pipeline slot.
#   * cf_eval.py only needs the stage2 checkpoint.
# To change the pipeline, change it HERE.
PIPELINE_SCRIPTS = [
    "pipeline/stage1.py", "pipeline/stage2.py", "pipeline/detect.py",
    "pipeline/per_entity_eval.py",
    "pipeline/quality_stage1.py", "pipeline/quality_stage2.py",
    # compare_aggregations.py removed 2026-07-23: channel aggregation is a no-op at
    # C=1 and the repo is univariate-only (it self-disabled on every live dataset).
    "pipeline/cf_eval.py",
]

# Named stage subsets for fast iteration (see --mode). Stems run in PIPELINE order.
MODES = {
    "research": ["stage1", "stage2", "detect", "per_entity_eval"],
    "figures":  ["stage1", "stage2", "detect", "per_entity_eval",
                 "quality_stage1", "quality_stage2"],
    "full":     [Path(s).stem for s in PIPELINE_SCRIPTS],
}

# Toy family short names -> base dataset name `toy_<short>_channel_anomalies`.
# Order is the historical sweep order (basic first). The 9 families: basic has
# 2 entities, the other 8 have 6 each => 50 pipelines per scale.
TOY_SHORT = [
    "basic", "point", "level", "pattern", "amplitude",
    "correlation", "group", "categorical", "morphological",
]


def toy_full(short: str, suffix: str) -> str:
    return f"toy_{short}_channel_anomalies{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# Job graph
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Job:
    dataset: str
    entity: str

    def __str__(self) -> str:
        return f"{self.dataset}/{self.entity}"


@dataclass
class Result:
    job: Job
    gpu: int
    rc: int                 # 0 = whole chain ok; else the failing script's rc
    failed_script: str      # "" when rc == 0
    seconds: float
    log_path: Path


def resolve_entities(python: str, dataset: str, spec: str,
                     _cache: dict[str, list[str]]) -> list[str]:
    """`spec` is "all"/"" -> ask scripts/list_entities.py; else a CSV of ids.

    list_entities.py is the single source of truth for the entity universe (it
    imports the dataset loaders), so the launcher never drifts from what
    training actually reads. Results are cached per (dataset) for "all".
    """
    spec = (spec or "all").strip()
    if spec != "all":
        return [e.strip() for e in spec.split(",") if e.strip()]
    if dataset in _cache:
        return _cache[dataset]
    out = subprocess.run(
        [python, "scripts/list_entities.py", "--dataset", dataset],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        print(f"[run] WARN: list_entities.py failed for '{dataset}' "
              f"(rc={out.returncode}):\n{out.stderr.strip()}", file=sys.stderr)
        _cache[dataset] = []
        return []
    ents = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    _cache[dataset] = ents
    return ents


def build_phase_jobs(
    specs: list[tuple[str, Optional[str]]],
    python: str,
    entities_arg: str,
    ent_cache: dict[str, list[str]],
) -> list[Job]:
    """Expand (dataset, pinned_entity|None) specs into concrete Jobs.

    A pinned entity becomes one Job directly. An unpinned dataset is fanned out
    to one Job per entity (filtered by `entities_arg`). Presence is decided by
    entity resolution, not a raw-dir guess: `list_entities.py` imports the
    loaders, so it knows e.g. smap/msl live under `data/raw/smap_msl`, not
    `data/raw/smap`. An empty result == not available on disk. Missing datasets
    are reported to stderr and skipped — a long sweep must not abort because one
    name is wrong or one dataset isn't built yet (e.g. the _32k variants).
    """
    jobs: list[Job] = []
    seen: set[tuple[str, str]] = set()

    def _add(ds: str, ent: str) -> None:
        if (ds, ent) not in seen:
            seen.add((ds, ent))
            jobs.append(Job(ds, ent))

    for ds, pinned in specs:
        if pinned is not None:
            _add(ds, pinned)
            continue
        ents = resolve_entities(python, ds, entities_arg, ent_cache)
        if not ents:
            hint = ("  (build 32k first: python scripts/build_all_toy_channel_anomalies_32k.py)"
                    if ds.endswith("_32k") else "")
            print(f"[run] MISSING: '{ds}' - no entities resolved, skipping.{hint}",
                  file=sys.stderr)
            continue
        for e in ents:
            _add(ds, e)
    return jobs


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    job: Job,
    gpu: int,
    python: str,
    threads: int,
    log_dir: Path,
    timings_csv: Path,
    timings_lock: threading.Lock,
    print_lock: threading.Lock,
) -> Result:
    """Run the 7-script chain for one (dataset, entity) in its own env.

    Aborts the chain on the first non-zero exit. All script output goes to one
    per-pipeline log; each script's wall time + exit code is appended to the
    shared timings.csv under `timings_lock`.
    """
    safe_ent = job.entity.replace("/", "_").replace("\\", "_")
    log_path = log_dir / f"{job.dataset}_{safe_ent}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu < 0 else str(gpu)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["NUMEXPR_MAX_THREADS"] = str(threads)
    env["DATASET_NAME"] = job.dataset
    env["DATASET_ENTITY"] = job.entity

    t0 = time.time()
    rc, failed = 0, ""
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# {time.strftime('%F %T')} BEGIN dataset={job.dataset} "
                   f"entity={job.entity} gpu={gpu} threads={threads}\n")
        logf.flush()
        for script in PIPELINE_SCRIPTS:
            logf.write(f"\n[{time.strftime('%T')}] >>> {script}\n")
            logf.flush()
            iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            s0 = time.time()
            try:
                proc = subprocess.run(
                    [python, script], env=env, cwd=str(PROJECT_ROOT),
                    stdout=logf, stderr=subprocess.STDOUT, check=False,
                )
                step_rc = proc.returncode
            except Exception as exc:  # interpreter missing, OS error, ...
                logf.write(f"\n# orchestrator caught exception: {exc!r}\n")
                step_rc = -1
            dur = time.time() - s0
            with timings_lock:
                with open(timings_csv, "a", encoding="utf-8") as tf:
                    tf.write(f"{iso},{job.dataset},{job.entity},{gpu},{script},{dur:.0f},{step_rc}\n")
            if step_rc != 0:
                logf.write(f"[{time.strftime('%T')}] !!! {script} failed (exit {step_rc}) "
                           f"-- aborting this pipeline.\n")
                rc, failed = step_rc, script
                break
            logf.write(f"[{time.strftime('%T')}]     done in {dur:.0f}s\n")
        if rc == 0:
            logf.write(f"\n# {time.strftime('%F %T')} END dataset={job.dataset} entity={job.entity}\n")

    return Result(job, gpu, rc, failed, time.time() - t0, log_path)


def worker(
    gpu: int,
    job_q: "queue.Queue[Job]",
    python: str,
    threads: int,
    log_dir: Path,
    timings_csv: Path,
    timings_lock: threading.Lock,
    print_lock: threading.Lock,
    results: list[Result],
    results_lock: threading.Lock,
) -> None:
    while True:
        try:
            job = job_q.get_nowait()
        except queue.Empty:
            return
        with print_lock:
            print(f"[{time.strftime('%H:%M:%S')}] GPU{gpu:>2} >> {job}", flush=True)
        res = run_pipeline(job, gpu, python, threads, log_dir,
                           timings_csv, timings_lock, print_lock)
        with print_lock:
            mark = "OK" if res.rc == 0 else "XX"
            tail = "" if res.rc == 0 else f" ({res.failed_script} rc={res.rc})"
            print(f"[{time.strftime('%H:%M:%S')}] GPU{gpu:>2} {mark} {job} "
                  f"in {res.seconds:.0f}s{tail}", flush=True)
        with results_lock:
            results.append(res)


def run_phase(
    label: str,
    jobs: list[Job],
    gpus: list[int],
    workers: int,
    python: str,
    threads: int,
    log_dir: Path,
    timings_csv: Path,
    timings_lock: threading.Lock,
) -> list[Result]:
    """One queue + N worker threads, round-robin slots over GPUs. Blocks until
    every job in this phase is done (hard barrier between phases)."""
    n_slots = max(1, workers)
    print()
    print("=" * 65)
    print(f"  PHASE {label}:  {len(jobs)} pipelines   workers: {n_slots}   "
          f"gpus: {','.join(str(g) for g in gpus)}   threads/w: {threads}")
    print(f"  scripts: {' '.join(PIPELINE_SCRIPTS)}")
    print("=" * 65)

    job_q: "queue.Queue[Job]" = queue.Queue()
    for j in jobs:
        job_q.put(j)
    results: list[Result] = []
    results_lock = threading.Lock()
    print_lock = threading.Lock()

    threads_list: list[threading.Thread] = []
    for i in range(n_slots):
        gpu = gpus[i % len(gpus)]
        t = threading.Thread(
            target=worker, name=f"slot{i}-gpu{gpu}",
            args=(gpu, job_q, python, threads, log_dir, timings_csv,
                  timings_lock, print_lock, results, results_lock),
        )
        t.start()
        threads_list.append(t)
    for t in threads_list:
        t.join()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

def default_threads(workers: int) -> int:
    ncpu = os.cpu_count() or 4
    return max(1, ncpu // max(1, workers))


def main() -> int:
    # Cross-platform safety: Windows consoles default to a legacy code page
    # (cp1252) that can't encode the arrows/dashes in --help. Force UTF-8 on the
    # Python side; harmless no-op where stdout is already UTF-8 or a pipe.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass

    p = argparse.ArgumentParser(
        description="Unified cross-platform launcher for the TimeVQVAE-AD-M pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--datasets", "-d", default="all",
                   help="Comma list. Tokens: 'all' (9 toy families), a short "
                        "family name (basic/point/level/pattern/amplitude/"
                        "correlation/group/categorical/morphological), a real "
                        "dataset (smap/msl/...), or '<dataset>:<entity>' for ONE entity.")
    p.add_argument("--scale", "-s", choices=["normal", "32k", "both"], default="normal",
                   help="Toy families only: normal | 32k | both (both = all normal "
                        "pipelines, then all 32k, hard barrier). Ignored for real "
                        "datasets and explicit full names. Default: normal.")
    p.add_argument("--entities", "-e", default="all",
                   help="Entity filter (CSV or 'all') for UNPINNED datasets. "
                        "Default 'all' = auto-enumerate from disk.")
    p.add_argument("--workers", "-w", type=int, default=1,
                   help="Concurrent pipelines (default 1 = sequential). Slots are "
                        "round-robin over --gpus.")
    p.add_argument("--gpus", "-g", default="0",
                   help="Comma GPU ids, round-robin (default '0'). Empty '' = CPU only.")
    p.add_argument("--threads", "-t", type=int, default=0,
                   help="OMP/MKL/OpenBLAS/NumExpr threads per worker "
                        "(default: cpu_count // workers).")
    p.add_argument("--python", default=None,
                   help="Interpreter path (default: $PIPELINE_PYTHON or 'python', "
                        "i.e. your active env).")
    p.add_argument("--log-dir", default="logs",
                   help="Per-pipeline log dir + timings.csv (default: logs).")
    p.add_argument("--name", "-n", default=None,
                   help="Process-title tag prefixed to every subprocess's name "
                        "(visible in ps/htop/nvidia-smi via setproctitle), e.g. "
                        "-n exp42 -> 'exp42 <dataset>/<entity> <script>'. Needs "
                        "`pip install setproctitle` in the pipeline env; no-op otherwise.")
    p.add_argument("--no-ensure", action="store_true",
                   help="Skip the automatic dataset presence/integrity check + "
                        "download/generate step (scripts/ensure_dataset.py).")
    p.add_argument("--only", default=None,
                   help="Run ONLY these pipeline scripts (comma list of basenames "
                        "w/ or w/o .py, e.g. 'cf_eval' or 'detect,cf_eval'), in "
                        "PIPELINE order, reusing existing checkpoints/artifacts — "
                        "nothing else reruns. Use to refresh one stage's outputs "
                        "after a fix WITHOUT retraining (stage1/stage2 have no "
                        "skip-if-exists guard, so they would otherwise retrain).")
    p.add_argument("--mode", choices=sorted(MODES), default=None,
                   help="Run a named subset of pipeline stages, in PIPELINE order: "
                        "'research' = stage1,stage2,detect,per_entity_eval (fast); "
                        "'figures' = + quality_stage1/2 plots; 'full' = all incl. "
                        "cf_eval. Mutually exclusive with --only.")
    p.add_argument("--skip", default=None,
                   help="Skip these pipeline scripts (comma list of basenames, .py "
                        "optional), keeping all others in PIPELINE order. Complement "
                        "of --only; applied after --mode.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip any (dataset, entity) whose detect report.json already "
                        "exists (the same completion signal the baselines use), so a "
                        "re-run of a partially-failed matrix does not retrain finished "
                        "pairs.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan per phase and exit without launching.")
    p.add_argument("--profile", action="store_true",
                   help="Turn on torch.profiler in every pipeline script (sets "
                        "TVQ_PROFILE=1 in the subprocess env). Per-script traces + "
                        "tables land in artifacts/profiling/, rolled up into "
                        "artifacts/profiling/summary.csv. Off by default.")
    args = p.parse_args()

    # Propagates to every subprocess via env.copy() in run_pipeline. Honour an
    # already-set TVQ_PROFILE too, so `TVQ_PROFILE=1 python run.py ...` also works.
    if args.profile:
        os.environ["TVQ_PROFILE"] = "1"

    # Propagates to every subprocess via env.copy() in run_pipeline; lib/proctitle.py
    # reads TVQ_PROC_TAG (+ DATASET_NAME/ENTITY) to title each process. Honour an
    # already-exported TVQ_PROC_TAG too.
    if args.name:
        os.environ["TVQ_PROC_TAG"] = args.name

    python = args.python or os.environ.get("PIPELINE_PYTHON") or "python"
    workers = max(1, args.workers)
    threads = args.threads if args.threads > 0 else default_threads(workers)

    # ── --only: restrict the chain to a subset of PIPELINE_SCRIPTS (in order),
    #    reusing existing checkpoints/artifacts. Matched on basename, .py optional.
    #    Single-process launcher, so reassigning the module global is safe: every
    #    worker reads it after this point and never mutates it.
    global PIPELINE_SCRIPTS
    all_stems = [Path(s).stem for s in PIPELINE_SCRIPTS]
    if args.only and args.mode:
        print("[run] --only and --mode are mutually exclusive.", file=sys.stderr)
        return 1
    if args.mode:
        keep = set(MODES[args.mode])
        PIPELINE_SCRIPTS = [s for s in PIPELINE_SCRIPTS if Path(s).stem in keep]
    if args.only:
        wanted = [t.strip().removesuffix(".py") for t in args.only.split(",") if t.strip()]
        unknown = [w for w in wanted if w not in all_stems]
        if unknown:
            print(f"[run] --only: unknown script(s) {unknown}; valid: "
                  f"{all_stems}", file=sys.stderr)
            return 1
        PIPELINE_SCRIPTS = [s for s in PIPELINE_SCRIPTS if Path(s).stem in set(wanted)]
    if args.skip:
        drop = [t.strip().removesuffix(".py") for t in args.skip.split(",") if t.strip()]
        unknown = [d for d in drop if d not in all_stems]
        if unknown:
            print(f"[run] --skip: unknown script(s) {unknown}; valid: "
                  f"{all_stems}", file=sys.stderr)
            return 1
        PIPELINE_SCRIPTS = [s for s in PIPELINE_SCRIPTS if Path(s).stem not in set(drop)]
    if not PIPELINE_SCRIPTS:
        print("[run] no pipeline scripts left to run after --mode/--only/--skip.",
              file=sys.stderr)
        return 1

    gpus_raw = args.gpus.strip()
    gpus = [int(x) for x in gpus_raw.split(",") if x.strip()] if gpus_raw else [-1]

    # ── Parse --datasets tokens into scalable toy families + literal specs.
    scalable: list[str] = []                       # short family names
    literals: list[tuple[str, Optional[str]]] = [] # (full_dataset, pinned_entity|None)
    for tok in (t.strip() for t in args.datasets.split(",")):
        if not tok:
            continue
        if tok == "all":
            scalable.extend(TOY_SHORT)
        elif tok in TOY_SHORT:
            scalable.append(tok)
        elif ":" in tok:
            ds, ent = (s.strip() for s in tok.split(":", 1))
            literals.append((ds, ent))
        else:
            literals.append((tok, None))
    # de-dup scalable, preserve order
    seen: set[str] = set()
    scalable = [s for s in scalable if not (s in seen or seen.add(s))]

    if not scalable and not literals:
        print("No datasets selected. Check --datasets.", file=sys.stderr)
        return 1

    scal_norm = [(toy_full(s, ""), None) for s in scalable]
    scal_32k = [(toy_full(s, "_32k"), None) for s in scalable]

    # ── Phases (hard barrier between them). Literals run once, in the first phase.
    phases: list[tuple[str, list[tuple[str, Optional[str]]]]] = []
    if args.scale == "normal":
        phases.append(("normal", scal_norm + literals))
    elif args.scale == "32k":
        phases.append(("32k", scal_32k + literals))
    else:  # both
        phases.append(("normal", scal_norm + literals))
        if scal_32k:
            phases.append(("32k", scal_32k))

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timings_csv = log_dir / "timings.csv"
    if not timings_csv.exists():
        timings_csv.write_text("start_iso,dataset,entity,gpu,script,duration_sec,exit_code\n",
                               encoding="utf-8")
    timings_lock = threading.Lock()

    print(f"[run] python   : {python}")
    print(f"[run] workers  : {workers}   gpus: {gpus}   threads/w: {threads}")
    print(f"[run] scale    : {args.scale}")
    print(f"[run] log dir  : {log_dir}")
    print(f"[run] profile  : {'ON -> artifacts/profiling/' if os.environ.get('TVQ_PROFILE','').lower() in {'1','true','on','yes'} else 'off'}")
    print(f"[run] scripts  : {' '.join(PIPELINE_SCRIPTS)}")

    # ── Ensure every requested dataset is present & intact, auto-downloading
    #    (smap/msl/asd) or generating (toy_*_channel_anomalies, incl. _32k) it
    #    if not. Logged to the terminal; provisioning is skipped on --dry-run
    #    (preview only) and entirely with --no-ensure. Provisioning runs under
    #    the pipeline interpreter `python` (numpy/pandas available there).
    if not args.no_ensure:
        ds_names = [ds for _, specs in phases for (ds, _ent) in specs]
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        try:
            from ensure_dataset import ensure_datasets
        except Exception as exc:  # pragma: no cover - defensive import guard
            print(f"[run] WARN: could not import ensure_dataset ({exc}); "
                  f"skipping dataset checks.", file=sys.stderr)
        else:
            ensure_datasets(ds_names, python=python, allow_provision=not args.dry_run)

    ent_cache: dict[str, list[str]] = {}
    phase_jobs: list[tuple[str, list[Job]]] = []
    for label, specs in phases:
        jobs = build_phase_jobs(specs, python, args.entities, ent_cache)
        phase_jobs.append((label, jobs))

    # --skip-existing: drop jobs whose detect report.json already exists, so a re-run of
    # a partially-failed matrix doesn't retrain finished pairs. The report path is
    # resolved EXACTLY as pipeline/detect.py builds it (detect.py:1379-1383). Config and
    # path helpers are imported lazily so the launcher stays importable without torch.
    # NOTE: this runs BEFORE the `total`/`--dry-run` lines on purpose, so `--dry-run`
    # prints the post-filter plan (which is also how it is tested).
    if args.skip_existing:
        import copy
        try:
            from config import load_config
            from utils import resolve_path, run_dir_for
        except Exception as exc:
            print(f"[run] WARN: --skip-existing unavailable ({exc}); running all jobs.",
                  file=sys.stderr)
        else:
            base_cfg = load_config()

            def _report_exists(j: Job) -> bool:
                c = copy.deepcopy(base_cfg)
                c.dataset.name = j.dataset
                c.dataset.entity_id = j.entity
                rbase = resolve_path(c.paths.reports) / run_dir_for(c, "stage1").relative_to(
                    resolve_path(c.paths.runs) / "stage1")
                return (rbase / f"{c.scoring.normalization}_{c.scoring.aggregation}"
                        / "report.json").exists()

            n_skipped = 0
            filtered: list[tuple[str, list[Job]]] = []
            for label, jobs in phase_jobs:
                keep = [j for j in jobs if not _report_exists(j)]
                n_skipped += len(jobs) - len(keep)
                filtered.append((label, keep))
            phase_jobs = filtered
            print(f"[run] --skip-existing: skipped {n_skipped} job(s) with an "
                  f"existing report.json.")

    total = sum(len(j) for _, j in phase_jobs)
    if total == 0:
        print("\n[run] nothing to do (no jobs after expansion). Check --datasets / data/raw/.",
              file=sys.stderr)
        return 1

    if args.dry_run:
        print("\n[run] --dry-run, listing jobs only:")
        idx = 0
        for label, jobs in phase_jobs:
            print(f"\n  ## PHASE {label} ({len(jobs)} pipelines)")
            for j in jobs:
                gpu = gpus[idx % len(gpus)]
                print(f"  [{idx:>3}] gpu={gpu}  {j}")
                idx += 1
        return 0

    all_results: list[Result] = []
    sweep_t0 = time.time()
    for label, jobs in phase_jobs:
        if not jobs:
            continue
        all_results += run_phase(label, jobs, gpus, workers, python, threads,
                                 log_dir, timings_csv, timings_lock)
    sweep_dt = time.time() - sweep_t0

    ok = [r for r in all_results if r.rc == 0]
    fail = [r for r in all_results if r.rc != 0]
    print()
    print("=" * 65)
    print(f"[run] done in {sweep_dt/60:.0f}m {sweep_dt%60:.0f}s - "
          f"{len(ok)}/{len(all_results)} OK, {len(fail)} FAILED")
    print(f"[run] per-pipeline logs: {log_dir}/<dataset>_<entity>.log")
    print(f"[run] timings CSV:       {timings_csv}")
    print("=" * 65)
    if fail:
        print("FAILED:")
        for r in sorted(fail, key=lambda r: (r.job.dataset, r.job.entity)):
            print(f"  - {r.job}  {r.failed_script} rc={r.rc}  log: {r.log_path}")
    return 0 if not fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
