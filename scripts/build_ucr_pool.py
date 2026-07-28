#!/usr/bin/env python3
"""
build_ucr_pool.py — a PUBLIC pretraining corpus in the wsd_fed on-disk format.

Motivation (L5 probe): the analytic-head line showed a ridge readout over a
COHERENT shared body reaches the centralized skyline, but every FEDERATION-LEGAL
body collapses — the body, not the head, was the problem. A body pretrained on a
PUBLIC external corpus is coherent (single pooled pretraining) AND legal (never
sees a client's private data). This script materializes that external corpus from
the UCR Anomaly Archive 2021 so it can be trained through the EXISTING
`train_centralized` path and evaluated by the EXISTING flare/prism analytic head.

Layout written (identical to wsd_fed → loaded by data.make_dataloaders unchanged):

    data/raw/ucr_pool/
      train/<eid>.npy        (T_train, 1) float32   — the UCR series' train segment
      test/<eid>.npy         (T_test,  1) float32   — remainder (unused by pretraining)
      test_label/<eid>.npy   (T_test,)   int64      — the archive's anomaly window
      clusters.json          {"pool": [<eid>, ...]}
      metadata.json          no dataset-specific keys ⇒ inherits the DEFAULT arch
                             (currently window=128, width_base=16, token_embedding_dim=64).
                             NB: the old `period → window = 2*period` override was
                             REMOVED from config.py, so window is now the plain default.

Only the TRAIN segment (values[:train_len], the pre-anomaly normal region) drives
pretraining. `test/` is written solely because make_dataloaders always loads the
test split; it is never used to train the body.

Usage:
    python scripts/build_ucr_pool.py                 # all eligible series, defaults
    python scripts/build_ucr_pool.py --train-cap 16384 --min-train 1024
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_UCR_DIR = (
    REPO / "preprocessing" / "dataset" / "AnomalyDatasets_2021"
    / "UCR_TimeSeriesAnomalyDatasets2021" / "FilesAreInHere" / "UCR_Anomaly_FullData"
)
# 148_UCR_Anomaly_Lab2Cmac011215EPG4_6000_17390_17520.txt
#   -> idx=148, name=..., train_len=6000, anom_start=17390, anom_end=17520
UCR_NAME_RE = re.compile(r"^(\d{3})_UCR_Anomaly_(.+?)_(\d+)_(\d+)_(\d+)\.txt$")

SPLITS = ("train", "test")


def read_ucr_values(path: Path) -> np.ndarray:
    """UCR .txt is either one value per line OR one whitespace-separated line.
    ``str.split()`` on any whitespace handles both layouts in one shot."""
    return np.array(path.read_text().split(), dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ucr-dir", default=str(DEFAULT_UCR_DIR))
    ap.add_argument("--out", default=str(REPO / "data" / "raw" / "ucr_pool"))
    ap.add_argument("--min-train", type=int, default=1024,
                    help="drop series whose train segment is shorter than this "
                         "(need > cfg.dataset.window_length for windows, plus room "
                         "for a val peel)")
    ap.add_argument("--train-cap", type=int, default=16384,
                    help="cap each series' train segment to its first N samples so "
                         "no single long series dominates the pool")
    ap.add_argument("--test-cap", type=int, default=4096,
                    help="cap each series' test segment (test is unused by pretraining)")
    args = ap.parse_args()

    ucr_dir = Path(args.ucr_dir)
    out = Path(args.out)
    files = sorted(ucr_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"No UCR .txt under {ucr_dir}. Download the archive first.")

    for sub in (*SPLITS, "test_label"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    entities: list[str] = []
    meta_entities: list[dict] = []
    n_skipped = 0
    for p in files:
        m = UCR_NAME_RE.match(p.name)
        if not m:
            n_skipped += 1
            continue
        idx, name = m.group(1), m.group(2)
        train_len, anom_start, anom_end = int(m.group(3)), int(m.group(4)), int(m.group(5))
        if train_len < args.min_train:
            n_skipped += 1
            continue

        values = read_ucr_values(p)
        total = values.shape[0]
        if total <= train_len + 256:            # need a usable (>=1 window) test remainder
            n_skipped += 1
            continue

        train = values[: min(train_len, args.train_cap)]
        test = values[train_len : train_len + args.test_cap]
        # Archive anomaly indices are ABSOLUTE in the full series; shift to test-relative
        # (end index inclusive) and clip to the (capped) test range.
        label = np.zeros(test.shape[0], dtype=np.int64)
        lo = max(0, anom_start - train_len)
        hi = min(test.shape[0], anom_end - train_len + 1)
        if lo < hi:
            label[lo:hi] = 1

        eid = f"ucr_{idx}"
        np.save(out / "train" / f"{eid}.npy", train[:, None].astype(np.float32))
        np.save(out / "test" / f"{eid}.npy", test[:, None].astype(np.float32))
        np.save(out / "test_label" / f"{eid}.npy", label)
        entities.append(eid)
        meta_entities.append({"entity_id": eid, "ucr_name": name,
                              "train_len": int(train.shape[0]), "test_len": int(test.shape[0])})

    if not entities:
        raise SystemExit("No eligible UCR series after filtering — loosen --min-train.")

    (out / "clusters.json").write_text(json.dumps({"pool": entities}, indent=2))
    # Deliberately NO "period" key → apply_dataset_overrides keeps window_length=256,
    # so the pretrained tokenizer/prior share wsd_fed's exact architecture.
    (out / "metadata.json").write_text(json.dumps({
        "dataset": "ucr_pool",
        "n_channels": 1,
        "source": "UCR_TimeSeriesAnomalyDatasets2021",
        "note": "public external pretraining corpus (train segments only used to train)",
        "clusters": {"pool": len(entities)},
        "entities": meta_entities,
    }, indent=2))

    tot_train = sum(e["train_len"] for e in meta_entities)
    print(f"[build_ucr_pool] wrote {len(entities)} entities -> {out}")
    print(f"[build_ucr_pool] skipped {n_skipped} series (unparsable / too short)")
    print(f"[build_ucr_pool] pooled train samples: {tot_train:,} "
          f"(~{tot_train // 256:,} non-overlap windows; stride-1 gives ~{tot_train:,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
