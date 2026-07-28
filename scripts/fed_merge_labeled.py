"""Merge single-arm federated_eval jsons into ONE combined json, RELABELING each file's
arm — so a γ-sweep (many runs of the SAME arm `federated_cb_only_ema` at different
`--cb-server-ema-decay`) becomes distinct rows in ONE `fed_aggregate.py` table.

`fed_aggregate.py` groups by `records[]._arm` and prints one table per json. A γ sweep
therefore needs each γ under a DISTINCT arm label inside a SINGLE json. Each input here is
`LABEL=path`: every record in `path` (produced by a single-arm `--arms` invocation) gets its
`_arm` overwritten to LABEL, then all records are concatenated. `meta` (dataset/cluster/commit)
is taken from the first input so the table header/provenance still resolve.

Usage:
    python scripts/fed_merge_labeled.py --out combined.json \
        federated_cb_only=cbonly.json cb_ema_g0=ema_g0.json cb_ema_g0.8=ema_g0.8.json
    python scripts/fed_aggregate.py combined.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("pairs", nargs="+", help="LABEL=path.json (each a single-arm json)")
    args = p.parse_args()

    combined: list[dict] = []
    meta: dict | None = None
    summary: dict = {}
    for pair in args.pairs:
        if "=" not in pair:
            raise SystemExit(f"expected LABEL=path.json, got {pair!r}")
        label, path = pair.split("=", 1)
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        recs = d.get("records") or []
        if not recs:
            print(f"[merge] WARN {path}: no records — skipped", file=sys.stderr)
            continue
        arms = {r.get("_arm") for r in recs}
        if len(arms) > 1:
            raise SystemExit(f"{path}: {len(arms)} arms {sorted(arms)} — relabel expects a single-arm json")
        for r in recs:
            r = dict(r)
            r["_arm"] = label
            combined.append(r)
        if meta is None:
            meta = d.get("meta", {})
        for _k, v in (d.get("summary") or {}).items():
            summary[label] = v            # fed_aggregate ignores summary, but keep it coherent
    if not combined:
        raise SystemExit("no records merged")

    n_arms = len({r["_arm"] for r in combined})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"meta": meta or {}, "summary": summary, "records": combined}, indent=2),
        encoding="utf-8")
    print(f"[merge] wrote {len(combined)} records, {n_arms} arms -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
