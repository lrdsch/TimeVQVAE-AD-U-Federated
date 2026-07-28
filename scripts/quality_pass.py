#!/usr/bin/env python3
"""quality_pass.py — run BOTH quality stages over the models `federated_eval` produced.

`federated_eval` runs stage1 -> stage2 -> detect and stops; the two quality stages are
part of run.py's per-entity chain, which the federated harness never enters. Both
`quality_stage1.evaluate` and `quality_stage2.evaluate` accept explicit checkpoint
paths, so they can be pointed at the checkpoints the harness already wrote.

Checkpoint layout walked (written by `federated_eval.save_client_ckpts`):

    artifacts/fed_eval/<dataset>/<cluster>/seed<N>/<arm>/<entity>/stage1.ckpt
                                                               .../stage2.ckpt

Reports land NEXT TO the checkpoints (`.../<entity>/quality_stage{1,2}/`) so a model's
quality artefacts stay attached to the arm that produced it — the per-entity default
run-dir would collide across arms, since every arm shares one (dataset, entity).

    python scripts/quality_pass.py --dataset wsd_fed --cluster c0 --seed 0
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))

import torch                                            # noqa: E402
from config import Config, apply_dataset_overrides, apply_env_overrides   # noqa: E402
import quality_stage1                                   # noqa: E402
import quality_stage2                                   # noqa: E402


def build_cfg(dataset: str, entity: str) -> Config:
    cfg = Config()
    cfg.dataset.name = dataset
    apply_dataset_overrides(cfg)
    apply_env_overrides(cfg)
    cfg.dataset.entity_id = entity
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cluster", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="local,centralized",
                    help="comma-separated arm names to walk (default: both baselines)")
    ap.add_argument("--entity", default=None,
                    help="restrict to ONE entity. Required when several jobs run "
                         "concurrently against the same arm directory: without it each "
                         "job would walk every checkpoint pair it finds, redoing its "
                         "neighbours' work and racing them on the same output files.")
    ap.add_argument("--root", default=str(REPO / "artifacts" / "fed_eval"),
                    help="checkpoint root; the walked path is "
                         "<root>/<dataset>/<cluster>/seed<N>/<arm>/<entity>/")
    ap.add_argument("--base", default=None,
                    help="explicit '<...>/seed<N>' directory, bypassing the path built "
                         "from --root/--dataset/--cluster. Mirrors federated_eval's "
                         "--out-dir, which drops the <dataset> level from the layout.")
    ap.add_argument("--max-plots", type=int, default=0,
                    help="stage-1 per-window plots. 0 (default) still writes the "
                         "best/median/worst HIGHLIGHTS; -1 plots every window, which "
                         "across dozens of clients will fill the disk.")
    ap.add_argument("--plot-workers", type=int, default=2,
                    help="stage-2 plot pool size. quality_stage2 defaults to "
                         "os.cpu_count()-1 (15 here), which is right for ONE run but "
                         "catastrophic when several cluster jobs run concurrently: "
                         "6 jobs x 15 workers = 90 processes on 16 cores.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a (arm, entity) whose quality_stage2 summary already exists")
    args = ap.parse_args()

    if args.base:
        base = Path(args.base)
    else:
        base = Path(args.root) / args.dataset
        if args.cluster:
            base = base / args.cluster
        base = base / f"seed{args.seed}"
    if not base.is_dir():
        print(f"[quality_pass] nothing at {base} — did the training job finish?")
        return 1

    # Only touch entities that belong to the CURRENT cluster. The artifacts path is
    # reused across experiments, and a rebuild can leave a former member's checkpoint
    # behind (e.g. the 07-16 toy rebuild reassigned uni_10/uni_11 out of M5_bearing,
    # but their pre-rebuild width_base=4 checkpoints stayed on disk). Walking every dir
    # then made quality_pass load a stale, arch-mismatched checkpoint and count it as a
    # failure — sinking the whole job's exit code even though every real model passed.
    members: set[str] | None = None
    if args.cluster:
        try:
            clusters = json.loads(
                (REPO / "data" / "raw" / args.dataset / "clusters.json").read_text())
            members = set(clusters.get(args.cluster, []))
        except Exception as e:
            print(f"[quality_pass] could not read cluster membership ({e}); "
                  f"processing every entity dir found")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    jobs: list[tuple[str, str, Path, Path]] = []
    skipped_stale: list[str] = []
    for arm in arms:
        adir = base / arm
        if not adir.is_dir():
            print(f"[quality_pass] arm '{arm}' absent under {base} — skipping")
            continue
        for edir in sorted(p for p in adir.iterdir() if p.is_dir()):
            if args.entity and edir.name != args.entity:
                continue
            if members is not None and edir.name not in members:
                skipped_stale.append(f"{arm}/{edir.name}")
                continue
            s1, s2 = edir / "stage1.ckpt", edir / "stage2.ckpt"
            if s1.exists() and s2.exists():
                jobs.append((arm, edir.name, s1, s2))
    if skipped_stale:
        print(f"[quality_pass] skipped {len(skipped_stale)} dir(s) not in cluster "
              f"'{args.cluster}' (stale/foreign): {', '.join(skipped_stale)}")

    if not jobs:
        print(f"[quality_pass] no (stage1,stage2) checkpoint pairs under {base}")
        return 1

    print(f"[quality_pass] {len(jobs)} model(s) under {base}: "
          + ", ".join(f"{a}/{e}" for a, e, _, _ in jobs))

    ok, failed = 0, []
    for i, (arm, entity, s1, s2) in enumerate(jobs, 1):
        tag = f"{arm}/{entity}"
        out1, out2 = s1.parent / "quality_stage1", s1.parent / "quality_stage2"
        if args.skip_existing and (out2 / "summary.json").exists():
            print(f"[quality_pass] ({i}/{len(jobs)}) {tag}: already done, skipping")
            ok += 1
            continue
        print(f"\n[quality_pass] ({i}/{len(jobs)}) {tag} =================")
        cfg = build_cfg(args.dataset, entity)
        try:
            quality_stage1.evaluate(cfg=cfg, stage1_ckpt=s1, output_dir=out1,
                                    device=device, max_plots=args.max_plots)
            quality_stage2.evaluate(cfg=cfg, stage1_ckpt=s1, stage2_ckpt=s2,
                                    output_dir=out2, device=device,
                                    plot_workers=args.plot_workers)
            ok += 1
        except Exception:                       # one bad model must not sink the batch
            print(f"[quality_pass] FAILED {tag}:\n{traceback.format_exc()}", flush=True)
            failed.append(tag)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n[quality_pass] done: {ok}/{len(jobs)} ok"
          + (f", FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
