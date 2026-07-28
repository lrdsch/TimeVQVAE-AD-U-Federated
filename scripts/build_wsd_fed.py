"""Materialize the frozen WSD federated benchmark into the repo's on-disk layout.

`dataset-federated/data/federated/WSD_frozen/wsd_federated.npz` is a single
consolidated archive (`ids`, `clusters`, `t0_unix`, `dt_sec`, and per-client
`<id>_train/_val/_test/_test_label`). Every loader, `federated.all_clients()`
and `federated.load_clusters()` in this repo instead expect the `toy_fed`
convention — one loose `.npy` per entity under `train/ val/ test/ test_label/`
plus a `clusters.json`. This script is the adapter between the two, so the same
pipeline runs on the synthetic benchmark and on the real one.

The build is a pure copy: no resampling, no re-splitting, no re-clustering. The
splits, the 4 clusters and the 1-day guard were all fixed by
`dataset-federated/scripts/build_frozen.py` and are reproduced verbatim. What
this script adds is the shape contract `(T, 1)` and a set of assertions that
re-check, at load time, the invariants `dataset-federated/MODEL.md` §1 relies on.

`metadata.json` deliberately carries NO `period` field. `config.load_config()`
rewrites `window_length = 2 * period` whenever it finds one (config.py:390), and
WSD's measured period is 1440 min — a 2880-sample window needs `train >= 2880`
*and* `val >= 2880`, which admits only 18 of the 31 clients. MODEL.md §1 rules
that out explicitly and fixes W at 128-256. The absence of the key is the
mechanism enforcing it; `--period` exists only to opt back in deliberately.

    python scripts/build_wsd_fed.py
    python scripts/build_wsd_fed.py --dev            # 8-client subset, 2 clusters
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import resolve_path  # noqa: E402

FROZEN = Path("dataset-federated/data/federated/WSD_frozen")
PERIOD = 1440          # minutes; measured, see dataset-federated/scripts/detect_periods.py
VAL_LEN = PERIOD       # val is exactly one period
MIN_TRAIN = 2 * PERIOD
MAX_ANOM_RATE = 0.10

SPLITS = ("train", "val", "test")


def _entity_id(series_id: int) -> str:
    """`kpi_004` — zero-padded so that `sorted()` on the stems is numeric order.

    `federated.all_clients()` globs `train/*.npy` and sorts lexicographically;
    an unpadded `kpi_4` would sort after `kpi_122` and silently permute the
    client order that FedAvg seeds are indexed by.
    """
    return f"kpi_{series_id:03d}"


def _load_frozen(root: Path) -> tuple[np.lib.npyio.NpzFile, np.ndarray, np.ndarray]:
    npz_path = root / "wsd_federated.npz"
    if not npz_path.exists():
        raise SystemExit(
            f"frozen archive not found at {npz_path}.\n"
            f"Rebuild it with: cd dataset-federated && python scripts/build_frozen.py"
        )
    Z = np.load(npz_path)
    return Z, Z["ids"], Z["clusters"]


def _check_client(sid: int, train, val, test, label) -> None:
    """Re-assert the MODEL.md §1 invariants on the arrays we are about to write.

    These held when `build_frozen.py` ran. Re-checking here costs microseconds
    and turns a silent protocol violation (a leaked NaN, a val that is not one
    whole period) into a build-time failure.
    """
    for name, arr in (("train", train), ("val", val), ("test", test)):
        if np.isnan(arr).any():
            raise SystemExit(f"kpi {sid}: {name} contains NaN — the frozen build guarantees none")
    if val.shape[0] != VAL_LEN:
        raise SystemExit(f"kpi {sid}: val is {val.shape[0]}, expected exactly one period ({VAL_LEN})")
    if train.shape[0] < MIN_TRAIN:
        raise SystemExit(f"kpi {sid}: train is {train.shape[0]} < 2 periods ({MIN_TRAIN})")
    if test.shape[0] != label.shape[0]:
        raise SystemExit(f"kpi {sid}: test {test.shape[0]} vs label {label.shape[0]} length mismatch")
    if not set(np.unique(label)).issubset({0, 1}):
        raise SystemExit(f"kpi {sid}: test_label is not binary")
    rate = float(label.mean())
    if rate <= 0:
        raise SystemExit(f"kpi {sid}: test has no anomaly — every client must hold >= 1 segment")
    if rate >= MAX_ANOM_RATE:
        raise SystemExit(f"kpi {sid}: test anomaly rate {rate:.3f} >= {MAX_ANOM_RATE}")


def _segment_lengths(label: np.ndarray) -> list[int]:
    d = np.diff(np.concatenate(([0], label.astype(np.int8), [0])))
    starts = np.where(d == 1)[0]
    stops = np.where(d == -1)[0]
    return (stops - starts).tolist()


def _n_segments(label: np.ndarray) -> int:
    """Count contiguous runs of 1 — reported per client so the eval can later
    split results by 1-segment vs >=2-segment clients (MODEL.md §3)."""
    return len(_segment_lengths(label))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frozen-dir", type=Path, default=FROZEN,
                   help="source WSD_frozen directory (default: %(default)s)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="default: <paths.raw_data>/<name>")
    p.add_argument("--name", type=str, default="wsd_fed",
                   help="dataset name; must start with 'wsd_fed' for data.load_records to dispatch")
    p.add_argument("--dev", action="store_true",
                   help="8-client / 2-cluster subset for smoke runs (clusters 0 and 3)")
    p.add_argument("--metrics-tolerance", type=int, default=None,
                   help="VUS/PATE buffer written to metadata.json. Default: the MEASURED median "
                        "anomaly-segment length. Leave it to config.py and you get "
                        "window_length//2 = 128, ~9x the median 14-sample segment.")
    p.add_argument("--period", type=int, default=None,
                   help="write metadata.json['period'], which makes config.py force "
                        "window_length = 2*period. MODEL.md forbids this for WSD (it drops "
                        "31 clients to 18). Off by default; pass only to reproduce that ablation.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not args.name.startswith("wsd_fed"):
        raise SystemExit(f"--name must start with 'wsd_fed' (data.load_records dispatches on the prefix); got {args.name!r}")

    frozen = args.frozen_dir if args.frozen_dir.is_absolute() else Path.cwd() / args.frozen_dir
    out = args.output_dir or (resolve_path("data/raw") / args.name)
    out = Path(out)

    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite to rebuild")
        shutil.rmtree(out)

    Z, ids, clusters = _load_frozen(frozen)
    dt_sec = int(Z["dt_sec"])
    t0_unix = Z["t0_unix"]

    keep = np.isin(clusters, (0, 3)) if args.dev else np.ones(len(ids), dtype=bool)
    ids, clusters, t0_unix = ids[keep], clusters[keep], t0_unix[keep]

    for sub in (*SPLITS, "test_label"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    cluster_map: dict[str, list[str]] = {}
    entities: list[dict] = []
    seg_lengths: list[int] = []

    for sid, cl, t0 in zip(ids, clusters, t0_unix):
        sid = int(sid)
        eid = _entity_id(sid)
        train = Z[f"{sid}_train"].astype(np.float32)
        val = Z[f"{sid}_val"].astype(np.float32)
        test = Z[f"{sid}_test"].astype(np.float32)
        label = Z[f"{sid}_test_label"].astype(np.int64)

        _check_client(sid, train, val, test, label)

        # (T,) -> (T, 1): TimeSeriesRecord rejects a 1-D X, and the toy_fed
        # loader's promotion path is the only other place this would happen.
        for split, arr in zip(SPLITS, (train, val, test)):
            np.save(out / split / f"{eid}.npy", arr[:, None])
        np.save(out / "test_label" / f"{eid}.npy", label)

        seg_lengths.extend(_segment_lengths(label))
        cluster_map.setdefault(f"c{int(cl)}", []).append(eid)
        entities.append({
            "entity_id": eid, "series_id": sid, "cluster": f"c{int(cl)}",
            "train_length": int(train.shape[0]), "val_length": int(val.shape[0]),
            "test_length": int(test.shape[0]),
            "n_anomaly_points": int(label.sum()), "n_anomaly_segments": _n_segments(label),
            "anomaly_rate": round(float(label.mean()), 6),
            "t0_unix": int(t0), "dt_sec": dt_sec,
        })

    for members in cluster_map.values():
        members.sort()

    segs = np.asarray(seg_lengths)
    # Buffer for VUS-PR / PATE. Derived from the labels, not from the window: the
    # events are what these metrics are tolerant *around*. See config.load_config().
    tolerance = args.metrics_tolerance if args.metrics_tolerance is not None \
        else int(np.median(segs))

    metadata = {
        "dataset": args.name,
        "metrics_tolerance": tolerance,
        "anomaly_segment_len": {
            "n": int(segs.size), "min": int(segs.min()), "median": int(np.median(segs)),
            "mean": round(float(segs.mean()), 2), "max": int(segs.max()),
        },
        "family_description": (
            f"Real federated benchmark: {len(ids)} univariate WSD KPIs (1-min sampling) as clients, "
            f"{len(cluster_map)} train-only KMeans clusters on the average-day deviation profile. "
            f"Frozen by dataset-federated/scripts/build_frozen.py; protocol in dataset-federated/MODEL.md."
        ),
        "source": str(frozen),
        "seed": 0,
        "n_series": int(len(ids)),
        "n_channels": 1,
        "dt_sec": dt_sec,
        "measured_period_min": PERIOD,
        "clusters": {k: len(v) for k, v in sorted(cluster_map.items())},
        "splits": {"policy": "frozen 50/50 with 1-day guard; val = last 1440 of the train half"},
        "entities": entities,
    }
    # See module docstring: presence of `period` makes config.py force W=2*period.
    if args.period is not None:
        metadata["period"] = int(args.period)

    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out / "clusters.json").write_text(
        json.dumps({k: cluster_map[k] for k in sorted(cluster_map)}, indent=2), encoding="utf-8")

    # Carried over verbatim: `correlations.csv` defines the n_eff ~= 26 correlation
    # components the significance test must use, and `manifest.csv` is the provenance
    # record. Neither is read by the loader; both are needed by the eval.
    for extra in ("manifest.csv", "correlations.csv", "selection.csv", "periods.csv"):
        src = frozen / extra
        if src.exists():
            shutil.copy2(src, out / extra)

    total_anom = sum(e["n_anomaly_points"] for e in entities)
    total_segs = sum(e["n_anomaly_segments"] for e in entities)
    print(f"[wsd_fed] wrote {out}")
    print(f"[wsd_fed] {len(ids)} clients, {len(cluster_map)} clusters "
          f"{ {k: len(v) for k, v in sorted(cluster_map.items())} }")
    print(f"[wsd_fed] {total_anom} anomalous points, {total_segs} segments "
          f"(len: min {segs.min()}, median {int(np.median(segs))}, max {segs.max()}); "
          f"val = {VAL_LEN} (1 period) for every client")
    print(f"[wsd_fed] metrics_tolerance = {tolerance} (VUS/PATE buffer); "
          f"config default would have been window_length//2 = 128")
    print(f"[wsd_fed] metadata.json has no 'period' -> window_length stays at config default "
          f"({'OVERRIDDEN: period=%d' % args.period if args.period else 'as MODEL.md requires'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
