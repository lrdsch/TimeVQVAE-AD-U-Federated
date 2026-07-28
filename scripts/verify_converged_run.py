#!/usr/bin/env python3
"""verify_converged_run.py — prove, from the artefacts a finished sweep leaves behind,
that the two fixes actually took effect in EVERY arm of EVERY job.

Both bugs were found only after I had already claimed they were absent, so this checks
the evidence on disk rather than the intent in the source.

CHECK 1 — batch sizes reached the trainers.
  `--batch` used to default to 16 and clobber the config unconditionally, so
  `batch_size_stage2` was dead config. Verified two ways:
    (a) each run's `meta.batch_stage1` / `meta.batch_stage2` equal the config values;
    (b) INDEPENDENTLY, the `(B batches/epoch)` printed by every trainer must equal
        ceil(n_windows / batch) recomputed from the .npy lengths on disk. (b) is the
        real check: it reads what the DataLoader actually did, not what was requested.

CHECK 2 — step-budget parity across arms.
  The `min_epochs` floor scales with batches/epoch, which is ~Kx larger on a pooled
  loader, so an unclamped floor hands the centralized skyline a bigger budget (and a
  flatter cosine) than the local baseline it is measured against. Every
  `converged protocol:` line must therefore show the SAME ceiling: stage1_max_steps
  for s1 and stage2_max_steps for s2, with no exceptions anywhere.

    python scripts/verify_converged_run.py                    # after the sweep
    python scripts/verify_converged_run.py --logdir ... --resdir ...
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))
from config import Config, apply_dataset_overrides          # noqa: E402

# "  [s1 kpi_015] converged protocol: max_steps=10000 warmup=1000 patience=2000 (12 batches/epoch)"
PROTO = re.compile(
    r"\[s(?P<stage>[12])\s+(?P<tag>\S+)\]\s+converged protocol:\s+max_steps=(?P<max>\d+)\s+"
    r"warmup=(?P<warm>\d+)\s+patience=(?P<pat>\d+)\s+\((?P<bpe>\d+) batches/epoch\)")
CLAMP = re.compile(r"\[s(?P<stage>[12])\s+(?P<tag>\S+)\]\s+min_epochs floor (?P<floor>\d+) CLAMPED")
# log file name -> dataset
DS_OF = {"toy_": "toy_fed_uni", "wsd_": "wsd_fed", "ucr_": "ucr_ad"}


def dataset_of(logname: str) -> str | None:
    for pre, ds in DS_OF.items():
        if logname.startswith(pre):
            return ds
    return None


def n_train_windows(ds: str, entity: str, window: int, stride: int) -> int | None:
    """Windows the train loader would emit for one entity, from the .npy on disk."""
    p = REPO / "data" / "raw" / ds / "train" / f"{entity}.npy"
    if not p.exists():
        return None
    T = int(np.load(p, mmap_mode="r").shape[0])
    return 0 if T < window else (T - window) // stride + 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", default=str(REPO / "logs" / "converged_all"))
    ap.add_argument("--resdir", default=str(REPO / "artifacts" / "converged_all"))
    args = ap.parse_args()

    cfg = Config()
    S1_CAP = int(cfg.training.stage1_max_steps)
    S2_CAP = int(cfg.training.stage2_max_steps)
    WINDOW, STRIDE = int(cfg.dataset.window_length), int(cfg.dataset.window_stride)
    EXP_B1, EXP_B2 = int(cfg.dataset.batch_size_stage1), int(cfg.dataset.batch_size_stage2)

    print(f"expected: batch s1={EXP_B1} s2={EXP_B2} | ceiling s1={S1_CAP} s2={S2_CAP} | "
          f"window={WINDOW} stride={STRIDE}\n")

    fail: list[str] = []
    warn: list[str] = []

    # ── CHECK 1a — declared provenance ───────────────────────────────────────
    print("=" * 72)
    print("CHECK 1a — meta.batch_stage{1,2} in each run's JSON")
    print("=" * 72)
    metas = sorted(Path(args.resdir).glob("*.json"))
    if not metas:
        fail.append(f"no result JSON under {args.resdir} (did --out-json reach the jobs?)")
        print(f"  !! none found under {args.resdir}")
    for m in metas:
        try:
            meta = json.loads(m.read_text()).get("meta", {})
        except Exception as e:
            fail.append(f"{m.name}: unreadable ({e})")
            continue
        b1, b2, proto = meta.get("batch_stage1"), meta.get("batch_stage2"), meta.get("protocol")
        ok = (b1 == EXP_B1 and b2 == EXP_B2)
        print(f"  {'OK ' if ok else 'FAIL'} {m.stem:22s} s1={b1} s2={b2} protocol={proto}")
        if not ok:
            fail.append(f"{m.stem}: batch s1={b1} s2={b2}, expected {EXP_B1}/{EXP_B2}")
        if proto != "converged":
            fail.append(f"{m.stem}: protocol={proto!r}, expected 'converged'")

    # ── CHECK 1b + CHECK 2 — what the trainers actually did ──────────────────
    print("\n" + "=" * 72)
    print("CHECK 1b — batches/epoch consistent with the batch size, recomputed from disk")
    print("CHECK 2  — every arm shares one step ceiling")
    print("=" * 72)
    logs = sorted(p for p in Path(args.logdir).glob("*.log") if not p.name.startswith("_"))
    if not logs:
        fail.append(f"no job logs under {args.logdir}")
    n_lines = 0
    clamps: list[str] = []
    for lg in logs:
        ds = dataset_of(lg.name)
        text = lg.read_text(errors="ignore")
        for c in CLAMP.finditer(text):
            clamps.append(f"{lg.stem}: s{c['stage']} {c['tag']} floor={c['floor']}")
        seen_bad = []
        for m in PROTO.finditer(text):
            n_lines += 1
            stage, tag = int(m["stage"]), m["tag"]
            mx, bpe = int(m["max"]), int(m["bpe"])
            cap = S1_CAP if stage == 1 else S2_CAP
            batch = EXP_B1 if stage == 1 else EXP_B2
            # CHECK 2: one ceiling for everyone
            if mx != cap:
                seen_bad.append(f"s{stage} {tag}: max_steps={mx} != {cap}")
            # CHECK 1b: only for per-client (non-pooled) tags, where the entity is known
            if ds and tag != "central":
                nw = n_train_windows(ds, tag, WINDOW, STRIDE)
                if nw:
                    exp_bpe = math.ceil(nw / batch)
                    if bpe != exp_bpe:
                        seen_bad.append(
                            f"s{stage} {tag}: {bpe} batches/epoch, but {nw} windows "
                            f"at batch {batch} implies {exp_bpe} "
                            f"(=> effective batch ~{math.ceil(nw / bpe)})")
        status = "OK " if not seen_bad else "FAIL"
        print(f"  {status} {lg.stem:22s} ({len(PROTO.findall(text))} protocol lines)")
        for b in seen_bad:
            print(f"       - {b}")
            fail.append(f"{lg.stem}: {b}")

    print(f"\n  checked {n_lines} 'converged protocol' lines across {len(logs)} job log(s)")
    if clamps:
        print(f"\n  min_epochs floor CLAMPED in {len(clamps)} case(s) — expected only on "
              f"large pooled arms:")
        for c in clamps[:12]:
            print(f"       - {c}")
        non_central = [c for c in clamps if " central " not in c]
        if non_central:
            warn.append(f"{len(non_central)} clamp(s) on a NON-pooled arm — unexpected")
    else:
        print("\n  no clamp fired (floor never exceeded the ceiling anywhere)")

    # ── verdict ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    for w in warn:
        print(f"WARN  {w}")
    if fail:
        print(f"VERDICT: FAILED — {len(fail)} problem(s)")
        for f in fail[:25]:
            print(f"  - {f}")
        return 1
    print("VERDICT: PASS — both fixes verified on disk for every arm of every job")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
