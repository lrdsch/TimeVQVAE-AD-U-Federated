#!/usr/bin/env python3
"""ucr_split_pair_manifest.py — pair the W=128 and W=2P ucr_split builds.

The two builds run the SAME recipe (5 clients, 10/10/20/20/30 % of the series' own train,
val = last 10 %, test untouched) and differ in exactly one thing: the rolling window that
the eligibility thresholds are multiples of. `ucr_split` uses the config constant 128;
`ucr_split_w2p` uses TimeVQVAE-AD Algorithm 1's per-series `W = 2*period`.

Because the partition does NOT depend on the window -- only the *admission* filter does --
a series present in both builds has byte-identical train/val/test/label on both sides. So
the intersection is a perfectly matched A/B: same data, same clients, only the window moves.
Outside the intersection there is no comparison to make, and that is fine as long as it is
marked; this script is what marks it.

Writes a `pairing` block into BOTH metadata.json files and prints the manifest. Idempotent.

    python scripts/ucr_split_pair_manifest.py
    python scripts/ucr_split_pair_manifest.py --a ucr_split --b ucr_split_w2p --verify-bytes 12
    python scripts/ucr_split_pair_manifest.py --markdown documentation/UCR_SPLIT_PAIRING.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load(ds: str) -> tuple[Path, dict]:
    p = REPO / "data" / "raw" / ds / "metadata.json"
    if not p.exists():
        raise SystemExit(f"missing {p} — build it first (scripts/build_ucr_split.py)")
    return p, json.loads(p.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="ucr_split", help="the W=128 build")
    ap.add_argument("--b", default="ucr_split_w2p", help="the W=2P build")
    ap.add_argument("--verify-bytes", type=int, default=12,
                    help="hash this many shared series' files on both sides (0 = skip). "
                         "The claim 'the intersection is byte-identical' is load-bearing for "
                         "the A/B, so it is checked rather than asserted.")
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args()

    pa, ma = load(args.a)
    pb, mb = load(args.b)
    A, B = set(ma["clusters"]), set(mb["clusters"])
    both, only_a, only_b = sorted(A & B), sorted(A - B), sorted(B - A)

    # why each non-shared series is missing from the other side
    def reasons(meta: dict) -> dict[str, str]:
        out = {}
        for row in meta.get("excluded_series", {}).get("series", []):
            f = row["file"]
            sid = f"ucr_{f.split('_', 1)[0]}" if f[:3].isdigit() else f
            out[sid] = row["reason"]
        return out

    why_a, why_b = reasons(ma), reasons(mb)
    windows = mb.get("windows", {})

    # ── byte-identity of the intersection ────────────────────────────────────────────
    verify = {"checked": 0, "identical": 0, "mismatches": []}
    if args.verify_bytes and both:
        step = max(1, len(both) // args.verify_bytes)
        for sid in both[::step][: args.verify_bytes]:
            for i in range(ma["protocol"]["n_clients_per_series"]):
                eid = f"{sid}_p{i}"
                for split in ("train", "val", "test", "test_label"):
                    fa = REPO / "data" / "raw" / args.a / split / f"{eid}.npy"
                    fb = REPO / "data" / "raw" / args.b / split / f"{eid}.npy"
                    if not (fa.exists() and fb.exists()):
                        verify["mismatches"].append(f"{eid}/{split}: missing file")
                        continue
                    verify["checked"] += 1
                    if digest(fa) == digest(fb):
                        verify["identical"] += 1
                    else:
                        verify["mismatches"].append(f"{eid}/{split}: SHA differs")

    block_common = {
        "sibling_builds": [args.a, args.b],
        "same_recipe": ("identical protocol (clients, shares, val tail, untouched test); "
                        "the ONLY difference is the window the eligibility thresholds scale with"),
        "n_comparable": len(both),
        "comparable_clusters": both,
        f"only_in_{args.a}": only_a,
        f"only_in_{args.b}": only_b,
        "intersection_is_byte_identical": (
            verify["checked"] > 0 and not verify["mismatches"]),
        "verification": verify,
        "how_to_read": (
            f"Compare arms ONLY on the {len(both)} clusters in `comparable_clusters`. The "
            f"{len(only_a)} + {len(only_b)} series outside it exist on one side only — there is "
            f"no matched counterpart, so they must never enter a paired test. Report them as "
            f"coverage, not as a result."),
    }
    ma["pairing"] = dict(block_common, role=f"W=128 build ({args.a})")
    mb["pairing"] = dict(block_common, role=f"W=2*period build ({args.b})")
    pa.write_text(json.dumps(ma, indent=2))
    pb.write_text(json.dumps(mb, indent=2))

    print(f"{args.a:16s} {len(A):4d} cluster")
    print(f"{args.b:16s} {len(B):4d} cluster")
    print(f"{'COMPARABILI':16s} {len(both):4d}  <- l'unico insieme su cui si fa un test appaiato")
    print(f"{'solo ' + args.a:16s} {len(only_a):4d}")
    print(f"{'solo ' + args.b:16s} {len(only_b):4d}")
    if args.verify_bytes:
        ok = "IDENTICI" if not verify["mismatches"] else f"{len(verify['mismatches'])} DIFFERENZE"
        print(f"byte-check: {verify['identical']}/{verify['checked']} file -> {ok}")
        for m in verify["mismatches"][:5]:
            print("   ", m)

    if args.markdown:
        L = [f"# `{args.a}` vs `{args.b}` — accoppiamento\n",
             "Generato da `scripts/ucr_split_pair_manifest.py`. Stesso protocollo, unica",
             "differenza la finestra a cui sono agganciate le soglie di eleggibilità.\n",
             "| | cluster |", "|---|---|",
             f"| `{args.a}` (W=128) | {len(A)} |",
             f"| `{args.b}` (W=2·periodo) | {len(B)} |",
             f"| **comparabili (intersezione)** | **{len(both)}** |",
             f"| solo `{args.a}` | {len(only_a)} |",
             f"| solo `{args.b}` | {len(only_b)} |", "",
             "⚠️ Un test appaiato gira **solo** sull'intersezione. Le serie fuori non hanno",
             "controparte: vanno riportate come copertura, mai come risultato.\n",
             f"Intersezione byte-identica: **{'sì' if not verify['mismatches'] else 'NO'}** "
             f"({verify['identical']}/{verify['checked']} file verificati).\n",
             f"## Solo in `{args.a}` — scartate dal build 2P ({len(only_a)})\n",
             "| serie | W=2P | perché fuori dal 2P |", "|---|---|---|"]
        L += [f"| `{s}` | {windows.get(s, '—')} | {why_b.get(s, '?')} |" for s in only_a]
        L += ["", f"## Solo in `{args.b}` — scartate dal build W=128 ({len(only_b)})\n",
              "| serie | W=2P | perché fuori dal 128 |", "|---|---|---|"]
        L += [f"| `{s}` | {windows.get(s, '—')} | {why_a.get(s, '?')} |" for s in only_b]
        args.markdown.write_text("\n".join(L) + "\n")
        print("markdown ->", args.markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
