#!/usr/bin/env python3
"""build_ucr_split.py — UCR turned into a QUANTITY-SKEW federated benchmark.

Every other federated build in this repo makes clients out of *different* series,
so `federated < local` is always confounded: the clients differ in distribution
AND in sample count at the same time. `ucr_split` removes the first term. Each UCR
series becomes one federation (= one cluster) whose N clients are non-overlapping,
contiguous slices of that series' OWN train segment. The clients are therefore
IID by construction — same sensor, same regime, same units — and differ only in
HOW MUCH data they hold (default 10/10/20/20/30 % of the train).

That makes the arms mean something exact:

    centralized(cluster)  ==  a local model on the whole train (modulo the val peel)
    local(client i)       ==  the same model on i's share alone
    federated(cluster)    ==  can it recover the pooled model without pooling?

The gap `centralized - federated` here is pure aggregation loss: no heterogeneity
excuse is available. `data/raw/ucr_ad` is the non-federated reference build of the
same archive (same native protocol, uncapped train, no val peel).

Layout written — the standard on-disk contract of `preprocessing/datasets/toy_fed.py`:

    data/raw/ucr_split/
      train/<eid>.npy        (T_i, 1) float32  — client i's slice, PARTITION of the train
      val/<eid>.npy          (V, 1)   float32  — the held-out tail of the train, SHARED
      test/<eid>.npy         (T_test, 1) f32   — the native UCR test, SHARED, untouched
      test_label/<eid>.npy   (T_test,) int64   — the archive's anomaly window, SHARED
      clusters.json          {"ucr_001": ["ucr_001_p0", ... "ucr_001_p4"], ...}
      metadata.json          per-entity provenance incl. the [start, stop) offsets

Split geometry, per series, over the native train segment `values[:train_len]`:

    |<-- p0 10% -->|<-- p1 10% -->|<-- p2 20% -->|<-- p3 20% -->|<-- p4 30% -->|<- val 10% ->|
    0                                                                                       train_len

Contiguous and chronological: the shards concatenate back to the original train
byte-for-byte (asserted at build time). Val is the LAST 10% — the same temporal
holdout policy as `build_wsd_fed.py`. NB the ordering does put the largest client
adjacent to val; if that matters for your claim, permute `--shares`.

Test and val are written per client (identical copies), because the loader indexes
every split by entity id. Nothing is resampled, re-labelled or capped: the test
split is exactly `values[train_len:]` with the archive's own anomaly window.

`metadata.json` deliberately carries NO `metrics_tolerance`, so the VUS/PATE buffer
stays at the config default `window_length // 2` — identical to `ucr_ad`, which is
what makes the two builds comparable. (Measured UCR anomaly-segment median is 101,
so the default 64 is not absurd here; pass --metrics-tolerance to override.)

Usage:
    python scripts/build_ucr_split.py
    python scripts/build_ucr_split.py --shares 10,10,20,20,30 --val-pct 10
    python scripts/build_ucr_split.py --shares 20,20,20,20,20      # equal-size control
    python scripts/build_ucr_split.py --limit 8 --name ucr_split_dev
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_UCR_DIR = (
    REPO / "preprocessing" / "dataset" / "AnomalyDatasets_2021"
    / "UCR_TimeSeriesAnomalyDatasets2021" / "FilesAreInHere" / "UCR_Anomaly_FullData"
)
# 148_UCR_Anomaly_Lab2Cmac011215EPG4_6000_17390_17520.txt
#   -> idx=148, name=..., train_len=6000, anom_start=17390, anom_end=17520 (inclusive)
UCR_NAME_RE = re.compile(r"^(\d{3})_UCR_Anomaly_(.+?)_(\d+)_(\d+)_(\d+)\.txt$")

SPLITS = ("train", "val", "test")


def read_ucr_values(path: Path) -> np.ndarray:
    """UCR .txt is either one value per line OR one whitespace-separated line.
    ``str.split()`` on any whitespace handles both layouts in one shot."""
    return np.array(path.read_text().split(), dtype=np.float32)


def cut_points(total: int, weights: list[int]) -> list[int]:
    """Boundaries of a contiguous partition of ``range(total)`` in the given ratios.

    Integer arithmetic on the CUMULATIVE weight, never on the per-block one: the
    latter loses up to one sample per block to flooring and the shards then fail to
    reassemble into the original train. Here `b[k] = total * cum[k] // sum(w)`, so
    the boundaries are monotone, `b[0] == 0`, `b[-1] == total` exactly, and every
    rounding error is absorbed by the neighbouring block instead of vanishing.
    """
    den = sum(weights)
    cum, acc = [0], 0
    for w in weights:
        acc += w
        cum.append(acc)
    return [total * c // den for c in cum]


def anomaly_label(test_len: int, train_len: int, anom_start: int, anom_end: int) -> np.ndarray:
    """Archive anomaly indices are ABSOLUTE in the full series; shift to test-relative
    (the archive's end index is inclusive) and clip to the test range."""
    label = np.zeros(test_len, dtype=np.int64)
    lo = max(0, anom_start - train_len)
    hi = min(test_len, anom_end - train_len + 1)
    if lo < hi:
        label[lo:hi] = 1
    return label


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ucr-dir", type=Path, default=DEFAULT_UCR_DIR)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="default: data/raw/<name>")
    ap.add_argument("--name", default="ucr_split",
                    help="dataset name; must start with 'ucr_split' for data.load_records to dispatch")
    ap.add_argument("--shares", default="10,10,20,20,30",
                    help="per-client percentages of the ORIGINAL train, in chronological "
                         "order. n_clients = len(shares). Must sum to 100 - --val-pct.")
    ap.add_argument("--val-pct", type=float, default=10.0,
                    help="percentage of the original train held out as val (the tail)")
    ap.add_argument("--min-client-train", type=int, default=256,
                    help="drop a SERIES unless every one of its clients gets at least this "
                         "many samples (2x the default window_length=128)")
    ap.add_argument("--min-val", type=int, default=128,
                    help="drop a series unless the val tail is at least this long (1 window)")
    ap.add_argument("--min-test", type=int, default=256,
                    help="drop a series unless the native test remainder is at least this long")
    ap.add_argument("--train-cap", type=int, default=None,
                    help="cap the ORIGINAL train segment before splitting (default: no cap — "
                         "the point of this build is to partition the real train)")
    ap.add_argument("--test-cap", type=int, default=None,
                    help="cap the test segment (default: no cap — 'il test non si tocca')")
    ap.add_argument("--limit", type=int, default=None,
                    help="keep only the first N eligible series (smoke builds)")
    ap.add_argument("--metrics-tolerance", type=int, default=None,
                    help="write metadata.json['metrics_tolerance'] (VUS/PATE buffer). Off by "
                         "default so the buffer matches ucr_ad's window-derived one.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not args.name.startswith("ucr_split"):
        raise SystemExit(f"--name must start with 'ucr_split' (data.load_records dispatches on "
                         f"the prefix); got {args.name!r}")

    shares = [int(s) for s in args.shares.split(",") if s.strip()]
    if len(shares) < 2:
        raise SystemExit("--shares needs at least 2 clients")
    if any(s <= 0 for s in shares):
        raise SystemExit(f"--shares must all be > 0; got {shares}")
    if abs(sum(shares) + args.val_pct - 100.0) > 1e-6:
        raise SystemExit(f"--shares sum to {sum(shares)} and --val-pct is {args.val_pct}: "
                         f"they must sum to 100 (they are percentages of the same train).")
    n_clients = len(shares)
    # Everything downstream is integer ratios, so express val as a weight too. x10 keeps
    # a fractional --val-pct like 12.5 exact instead of silently flooring it.
    weights = [int(round(s * 10)) for s in shares] + [int(round(args.val_pct * 10))]

    from utils import resolve_path  # noqa: E402  (needs REPO on sys.path)

    out = Path(args.output_dir) if args.output_dir else (resolve_path("data/raw") / args.name)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite to rebuild")
        shutil.rmtree(out)

    files = sorted(Path(args.ucr_dir).glob("*.txt"))
    if not files:
        raise SystemExit(f"No UCR .txt under {args.ucr_dir}. Download the archive first.")

    for sub in (*SPLITS, "test_label"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    cluster_map: dict[str, list[str]] = {}
    meta_entities: list[dict] = []
    skipped: list[tuple[str, str]] = []
    n_series = 0

    for p in files:
        m = UCR_NAME_RE.match(p.name)
        if not m:
            skipped.append((p.name, "unparsable filename"))
            continue
        idx, ucr_name = m.group(1), m.group(2)
        train_len, anom_start, anom_end = int(m.group(3)), int(m.group(4)), int(m.group(5))

        values = read_ucr_values(p)
        total = int(values.shape[0])
        if total <= train_len:
            skipped.append((p.name, f"no test remainder (total {total} <= train_len {train_len})"))
            continue

        train_full = values[: train_len if args.train_cap is None else min(train_len, args.train_cap)]
        test = values[train_len:] if args.test_cap is None else values[train_len: train_len + args.test_cap]
        label = anomaly_label(test.shape[0], train_len, anom_start, anom_end)

        T = int(train_full.shape[0])
        b = cut_points(T, weights)
        client_lens = [b[i + 1] - b[i] for i in range(n_clients)]
        val_len = b[n_clients + 1] - b[n_clients]

        if min(client_lens) < args.min_client_train:
            skipped.append((p.name, f"smallest client slice {min(client_lens)} < {args.min_client_train}"))
            continue
        if val_len < args.min_val:
            skipped.append((p.name, f"val {val_len} < {args.min_val}"))
            continue
        if test.shape[0] < args.min_test:
            skipped.append((p.name, f"test {test.shape[0]} < {args.min_test}"))
            continue
        if args.limit is not None and n_series >= args.limit:
            skipped.append((p.name, "beyond --limit"))
            continue

        series_id = f"ucr_{idx}"
        val = train_full[b[n_clients]: b[n_clients + 1]]

        members: list[str] = []
        for i in range(n_clients):
            eid = f"{series_id}_p{i}"
            shard = train_full[b[i]: b[i + 1]]
            # (T,) -> (T, 1): TimeSeriesRecord rejects a 1-D X.
            np.save(out / "train" / f"{eid}.npy", shard[:, None].astype(np.float32))
            # val/test/test_label are the SAME arrays for every client of this series —
            # written per entity because the loader indexes each split by entity id.
            np.save(out / "val" / f"{eid}.npy", val[:, None].astype(np.float32))
            np.save(out / "test" / f"{eid}.npy", test[:, None].astype(np.float32))
            np.save(out / "test_label" / f"{eid}.npy", label)
            members.append(eid)
            meta_entities.append({
                "entity_id": eid, "series": series_id, "ucr_name": ucr_name,
                "cluster": series_id, "partition": i, "share_pct": shares[i],
                "train_start": int(b[i]), "train_stop": int(b[i + 1]),
                "train_length": int(client_lens[i]), "val_length": int(val_len),
                "test_length": int(test.shape[0]),
                "n_anomaly_points": int(label.sum()),
                "anomaly_rate": round(float(label.mean()), 6),
                "orig_train_length": T,
            })

        # The contract this whole build rests on: the shards + val are a PARTITION of
        # the original train — no overlap, nothing dropped, order preserved.
        rebuilt = np.concatenate([train_full[b[i]: b[i + 1]] for i in range(n_clients + 1)])
        if rebuilt.shape != train_full.shape or not np.array_equal(rebuilt, train_full):
            raise SystemExit(f"{series_id}: shards+val do not reassemble the original train")

        cluster_map[series_id] = members
        n_series += 1

    if not cluster_map:
        raise SystemExit("No eligible UCR series after filtering — loosen --min-client-train.")

    metadata = {
        "dataset": args.name,
        "n_channels": 1,
        "source": "UCR_TimeSeriesAnomalyDatasets2021",
        "family_description": (
            f"Quantity-skew federated benchmark: each of {n_series} UCR series is one cluster, "
            f"split into {n_clients} clients holding contiguous, non-overlapping "
            f"{'/'.join(str(s) for s in shares)}% slices of its own train segment. "
            f"val = the last {args.val_pct:g}% of the train, shared by the cluster; "
            f"test = the native UCR test remainder, untouched and shared by the cluster. "
            f"Clients are IID by construction — they differ only in sample count."
        ),
        "protocol": {
            "n_clients_per_series": n_clients,
            "shares_pct": shares,
            "val_pct": args.val_pct,
            "split": "contiguous chronological; val is the tail; test never touched",
            "test_shared_within_cluster": True,
            "val_shared_within_cluster": True,
            "train_cap": args.train_cap,
            "test_cap": args.test_cap,
        },
        "n_series": n_series,
        "n_entities": len(meta_entities),
        "clusters": {k: len(v) for k, v in sorted(cluster_map.items())},
        "entities": meta_entities,
    }
    if args.metrics_tolerance is not None:
        metadata["metrics_tolerance"] = int(args.metrics_tolerance)

    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out / "clusters.json").write_text(
        json.dumps({k: cluster_map[k] for k in sorted(cluster_map)}, indent=2), encoding="utf-8")

    tot_train = sum(e["train_length"] for e in meta_entities)
    per_share: dict[int, list[int]] = {}
    for e in meta_entities:
        per_share.setdefault(e["partition"], []).append(e["train_length"])
    print(f"[ucr_split] wrote {out}")
    print(f"[ucr_split] {n_series} series x {n_clients} clients = {len(meta_entities)} entities, "
          f"{n_series} clusters")
    print(f"[ucr_split] skipped {len(skipped)} series "
          f"(min client slice {args.min_client_train}, min val {args.min_val})")
    for i in range(n_clients):
        L = np.asarray(per_share[i])
        print(f"[ucr_split]   p{i} ({shares[i]:>2d}%): train len min {L.min():>6d} "
              f"median {int(np.median(L)):>6d} max {L.max():>6d}  "
              f"(~{int(np.median(L)) - 128 + 1} stride-1 windows at W=128)")
    print(f"[ucr_split] partitioned train samples: {tot_train:,}")
    print(f"[ucr_split] metadata.json has "
          f"{'metrics_tolerance=%d' % args.metrics_tolerance if args.metrics_tolerance is not None else 'NO metrics_tolerance -> window//2, same as ucr_ad'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
