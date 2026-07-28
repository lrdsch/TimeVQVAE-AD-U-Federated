"""Verify that every dataset under ``data/raw/`` matches a reference manifest.

Workflow:

  1. On the source machine (where every dataset is already downloaded /
     generated / copied by hand), produce the manifest::

         python scripts/check_dataset_integrity.py --snapshot

     This writes ``data_integrity_manifest.json`` at the repo root.

  2. Commit the manifest::

         git add data_integrity_manifest.json
         git commit -m "data: refresh integrity manifest"
         git push

  3. On the target machine (after `git pull` + after running
     `setup_data.sh` and copying manual datasets), verify::

         python scripts/check_dataset_integrity.py

     The script walks ``data/raw/``, recomputes the per-dataset summary, and
     reports per-dataset status:

       [OK]       folder present, file tree matches the manifest
       [DRIFT]    folder present, but file count / sizes / paths differ
       [MISSING]  folder declared in manifest but not present locally
       [EXTRA]    local folder not in manifest (untracked dataset)

     Exit code is 0 iff no DRIFT was found. MISSING/EXTRA are informational.

The "tree hash" used for matching is SHA-256 over the sorted, newline-joined
``<relpath>|<size>\\n`` listing of every file under the dataset folder. Two
trees with identical file paths and sizes produce the same hash; renames,
truncations, or additions all change it. File *contents* are not hashed by
default — pass ``--content-hash`` to also SHA-256 every file (slow, ~80GB
to read on a full local copy).

This script is intentionally stdlib-only so it can run on a fresh box
without installing the project dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_MANIFEST = ROOT / "data_integrity_manifest.json"
MANIFEST_VERSION = 1


# ---------------------------------------------------------------------------
# Filesystem walk
# ---------------------------------------------------------------------------

def _iter_files(folder: Path) -> Iterable[Path]:
    """Yield every regular file under ``folder``, depth-first, sorted at each
    directory level so the listing is deterministic across filesystems."""
    stack: list[Path] = [folder]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name)
        except (PermissionError, FileNotFoundError):
            continue
        # Sort dirs after files so we still recurse but yield files in a
        # stable, lexicographic order at each level.
        dirs: list[Path] = []
        for entry in entries:
            if entry.is_symlink():
                continue
            if entry.is_dir():
                dirs.append(entry)
            elif entry.is_file():
                yield entry
        for d in reversed(dirs):
            stack.append(d)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Per-dataset scan
# ---------------------------------------------------------------------------

@dataclass
class DatasetScan:
    name: str
    present: bool
    n_files: int = 0
    total_bytes: int = 0
    tree_hash: str = ""
    files: list[tuple[str, int, str | None]] | None = None  # (relpath, size, sha256 or None)


def scan_dataset(folder: Path, *, dataset_name: str, content_hash: bool,
                 verbose: bool) -> DatasetScan:
    if not folder.is_dir():
        return DatasetScan(name=dataset_name, present=False)

    files: list[tuple[str, int, str | None]] = []
    h = hashlib.sha256()
    total_bytes = 0
    t0 = time.time()
    for f in _iter_files(folder):
        size = f.stat().st_size
        rel = f.relative_to(folder).as_posix()
        sha = _sha256_file(f) if content_hash else None
        files.append((rel, size, sha))
        # Tree hash mixes path + size (+ content sha if enabled). Path is the
        # primary signal; size + sha catch truncated / corrupted files.
        line = f"{rel}|{size}"
        if sha is not None:
            line = f"{line}|{sha}"
        h.update(line.encode("utf-8"))
        h.update(b"\n")
        total_bytes += size

    dt = time.time() - t0
    if verbose:
        print(f"  scanned {len(files)} files in {dt:.1f}s "
              f"({total_bytes / 1e6:.1f} MB)", flush=True)

    return DatasetScan(
        name=dataset_name,
        present=True,
        n_files=len(files),
        total_bytes=total_bytes,
        tree_hash=h.hexdigest(),
        files=files,
    )


# ---------------------------------------------------------------------------
# Manifest read / write
# ---------------------------------------------------------------------------

def write_manifest(scans: list[DatasetScan], manifest_path: Path, *,
                   include_files: bool, content_hash: bool) -> None:
    manifest: dict = {
        "manifest_version": MANIFEST_VERSION,
        "generated_at_unix": int(time.time()),
        "generated_on": platform.node(),
        "raw_dir": "data/raw",
        "content_hash": content_hash,
        "datasets": {},
    }
    for s in scans:
        entry: dict = {
            "present": s.present,
            "n_files": s.n_files,
            "total_bytes": s.total_bytes,
            "tree_hash": s.tree_hash,
        }
        if include_files and s.files is not None:
            # Compact representation: list of [path, size] (+ sha if present).
            entry["files"] = [
                [p, sz] if sha is None else [p, sz, sha]
                for (p, sz, sha) in s.files
            ]
        manifest["datasets"][s.name] = entry

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8")


def read_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest not found at {manifest_path}. Run "
            f"`python scripts/{Path(__file__).name} --snapshot` on the source "
            f"machine first, then commit and push the manifest."
        )
    with manifest_path.open("r", encoding="utf-8") as fh:
        m = json.load(fh)
    version = m.get("manifest_version")
    if version != MANIFEST_VERSION:
        raise RuntimeError(
            f"manifest version mismatch: file={version}, "
            f"script={MANIFEST_VERSION}. Regenerate with --snapshot."
        )
    return m


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

@dataclass
class DatasetReport:
    name: str
    status: str  # "ok" | "drift" | "missing" | "extra"
    detail: str = ""
    diffs: list[str] | None = None


def _diff_file_lists(expected: list[list], actual: list[tuple[str, int, str | None]],
                     limit: int = 10) -> list[str]:
    """Return human-readable diff lines between manifest's file list and
    actual scan's file list. Capped to ``limit`` entries."""
    exp_map = {row[0]: (row[1], row[2] if len(row) > 2 else None) for row in expected}
    act_map = {p: (sz, sha) for p, sz, sha in actual}

    only_local = sorted(set(act_map) - set(exp_map))
    only_manifest = sorted(set(exp_map) - set(act_map))
    common = set(exp_map) & set(act_map)
    size_mismatches = sorted(
        p for p in common if exp_map[p][0] != act_map[p][0]
    )
    sha_mismatches = sorted(
        p for p in common
        if exp_map[p][1] is not None and act_map[p][1] is not None
        and exp_map[p][1] != act_map[p][1]
    )

    diffs: list[str] = []
    def _add(kind: str, paths: list[str]) -> None:
        if not paths:
            return
        diffs.append(f"  {kind}: {len(paths)}")
        for p in paths[:limit]:
            diffs.append(f"    - {p}")
        if len(paths) > limit:
            diffs.append(f"    ... and {len(paths) - limit} more")

    _add("missing files (in manifest, not local)", only_manifest)
    _add("extra files (local only)", only_local)
    _add("size mismatch", [
        f"{p} (manifest={exp_map[p][0]}B, local={act_map[p][0]}B)"
        for p in size_mismatches
    ])
    _add("content sha mismatch", sha_mismatches)
    return diffs


def compare(scan: DatasetScan, manifest_entry: dict | None,
            *, verbose: bool) -> DatasetReport:
    if manifest_entry is None:
        return DatasetReport(
            name=scan.name, status="extra",
            detail=f"local folder not in manifest "
                   f"({scan.n_files} files, {scan.total_bytes / 1e6:.1f} MB)",
        )
    if not manifest_entry.get("present", True):
        # Manifest itself records folder as absent; treat as no entry.
        if scan.present:
            return DatasetReport(name=scan.name, status="extra",
                                 detail="manifest had folder absent; now present")
        return DatasetReport(name=scan.name, status="ok",
                             detail="manifest absent, local absent")

    if not scan.present:
        return DatasetReport(
            name=scan.name, status="missing",
            detail=f"expected {manifest_entry['n_files']} files "
                   f"({manifest_entry['total_bytes'] / 1e6:.1f} MB)",
        )

    if scan.tree_hash == manifest_entry["tree_hash"]:
        return DatasetReport(
            name=scan.name, status="ok",
            detail=f"{scan.n_files} files, {scan.total_bytes / 1e6:.1f} MB",
        )

    detail = (f"local: {scan.n_files} files / {scan.total_bytes / 1e6:.1f} MB, "
              f"manifest: {manifest_entry['n_files']} files "
              f"/ {manifest_entry['total_bytes'] / 1e6:.1f} MB")

    diffs: list[str] | None = None
    if verbose and "files" in manifest_entry and scan.files is not None:
        diffs = _diff_file_lists(manifest_entry["files"], scan.files)

    return DatasetReport(name=scan.name, status="drift", detail=detail, diffs=diffs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

STATUS_MARK = {"ok": "OK ", "drift": "XX ", "missing": "-- ", "extra": "++ "}


def _print_report(reports: list[DatasetReport]) -> None:
    width = max((len(r.name) for r in reports), default=0)
    by_status: dict[str, int] = {"ok": 0, "drift": 0, "missing": 0, "extra": 0}
    for r in reports:
        mark = STATUS_MARK.get(r.status, "?? ")
        print(f"  [{mark}] {r.name:<{width}}  {r.detail}")
        if r.diffs:
            for line in r.diffs:
                print(line)
        by_status[r.status] = by_status.get(r.status, 0) + 1
    print()
    print(f"  ok={by_status['ok']}  drift={by_status['drift']}  "
          f"missing={by_status['missing']}  extra={by_status['extra']}")


def cmd_snapshot(args: argparse.Namespace) -> int:
    raw_dir = Path(args.raw_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    if not raw_dir.is_dir():
        print(f"[error] raw dir not found: {raw_dir}", file=sys.stderr)
        return 2

    folders = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        folders = [f for f in folders if f.name in wanted]

    print(f"[snapshot] raw_dir={raw_dir}")
    print(f"[snapshot] scanning {len(folders)} folder(s); "
          f"content_hash={args.content_hash}")

    scans: list[DatasetScan] = []
    t0 = time.time()
    for f in folders:
        print(f"  [{f.name}]", flush=True)
        s = scan_dataset(f, dataset_name=f.name,
                         content_hash=args.content_hash, verbose=True)
        scans.append(s)

    write_manifest(scans, manifest_path,
                   include_files=not args.no_files,
                   content_hash=args.content_hash)
    print(f"\n[snapshot] wrote manifest -> {manifest_path}")
    print(f"[snapshot] total: {sum(s.n_files for s in scans)} files, "
          f"{sum(s.total_bytes for s in scans) / 1e9:.2f} GB, "
          f"{time.time() - t0:.1f}s")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    raw_dir = Path(args.raw_dir).resolve()
    manifest_path = Path(args.manifest).resolve()

    manifest = read_manifest(manifest_path)
    manifest_content_hash = bool(manifest.get("content_hash"))
    if args.content_hash and not manifest_content_hash:
        print("[warn] manifest was written without --content-hash; only path+size "
              "will be compared (the local content hashes will be ignored).",
              file=sys.stderr)
    use_content_hash = args.content_hash and manifest_content_hash

    expected_datasets = manifest.get("datasets", {})
    local_folders = {p.name for p in raw_dir.iterdir() if p.is_dir()} \
        if raw_dir.is_dir() else set()
    all_names = sorted(set(expected_datasets) | local_folders)
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        all_names = [n for n in all_names if n in wanted]

    print(f"[check] raw_dir={raw_dir}")
    print(f"[check] manifest={manifest_path}")
    print(f"[check] manifest generated on {manifest.get('generated_on', '?')} "
          f"at unix={manifest.get('generated_at_unix', '?')}")
    print(f"[check] datasets to verify: {len(all_names)} "
          f"(content_hash check={'on' if use_content_hash else 'off'})")
    print()

    reports: list[DatasetReport] = []
    for name in all_names:
        folder = raw_dir / name
        scan = scan_dataset(folder, dataset_name=name,
                            content_hash=use_content_hash, verbose=False)
        report = compare(scan, expected_datasets.get(name), verbose=args.verbose)
        reports.append(report)

    _print_report(reports)
    drift = sum(1 for r in reports if r.status == "drift")
    return 0 if drift == 0 else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", action="store_true",
                   help="Write the manifest from the current data/raw/ tree "
                        "(source machine).")
    p.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR),
                   help="Path to the raw data root (default: data/raw).")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                   help="Manifest path (default: data_integrity_manifest.json).")
    p.add_argument("--only", default="",
                   help="Comma-separated subset of dataset folder names.")
    p.add_argument("--content-hash", action="store_true",
                   help="Also SHA-256 each file's contents. Slow (full read of "
                        "every byte) but catches silent corruption.")
    p.add_argument("--no-files", action="store_true",
                   help="(snapshot only) Omit the full per-file list — keeps the "
                        "manifest tiny but disables drift drill-down.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="(check only) On DRIFT, list which files differ.")
    args = p.parse_args(argv)

    if args.snapshot:
        return cmd_snapshot(args)
    return cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
