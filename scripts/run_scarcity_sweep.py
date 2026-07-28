"""STEP 5c — scarcity sweep driver.

Builds the federated benchmark at several per-client train lengths and runs the
local/centralized/federated comparison (pipeline/federated_eval.py) on each,
then prints the headline trend: how the federated-vs-local detection gap grows
as local data shrinks (the calibration that proves federation's value).

Each (length) run is a separate subprocess (isolated GPU memory). Pick a free
GPU via the environment:

  # multivariate benchmark (C=8, clients fed_0..fed_5)
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python scripts/run_scarcity_sweep.py \
      --lengths 256,512,1024 --seeds 0,1,2 \
      --s1-epochs 8 --s2-epochs 8 --s1-rounds 8 --s2-rounds 8 --batch 16

  # univariate benchmark, one machine-type cluster (C=1, clients uni_00..uni_02)
  python scripts/run_scarcity_sweep.py --builder uni --cluster M1_rotary \
      --lengths 512,1024,2048 --seeds 0

`--train-length` is the FL difficulty knob: with window_length=256, a 256-step
train split yields exactly ONE window per client, which is degenerate — the
`local` arm has nothing to learn and the federated gap is an artefact rather
than a finding. Start the sweep at 512 unless you specifically want that point.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Repo-local, so the sweep works on any machine (this used to be a hard-coded
# absolute POSIX scratch path from a different host).
SCRATCH = REPO / "artifacts" / "sweep"

BUILDERS = {
    # "uni" is the active tier for the univariate (-U) study; "fed" is the legacy C=8
    # multivariate line (-M, closed 2026-07-10) — kept only as regeneration capability.
    "uni": ("build_toy_fed_uni.py", "toy_fed_uni"),  # univariate C=1, clustered clients — DEFAULT for -U
    "fed": ("build_toy_fed.py", "toy_fed"),          # legacy multivariate, fixed C=8 (off-path)
}


def run(cmd: list) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def discover_clients(dataset_dir: Path, cluster: str | None) -> list[str]:
    """Entity ids from disk (`fed_0..` or `uni_00..`), optionally one cluster."""
    if cluster:
        cpath = dataset_dir / "clusters.json"
        if not cpath.exists():
            raise SystemExit(f"{dataset_dir} has no clusters.json — --cluster is unavailable.")
        clusters = json.loads(cpath.read_text(encoding="utf-8"))
        if cluster not in clusters:
            raise SystemExit(f"unknown cluster {cluster!r}; known: {sorted(clusters)}")
        return list(clusters[cluster])
    return sorted(p.stem for p in (dataset_dir / "train").glob("*.npy"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--builder", choices=sorted(BUILDERS), default="fed",
                    help="'fed' = multivariate C=8; 'uni' = univariate C=1 with 6 clusters")
    ap.add_argument("--tier", choices=["dev", "full"], default="dev")
    ap.add_argument("--lengths", type=str, default="512,1024,2048")
    ap.add_argument("--clients", type=str, default=None,
                    help="comma-separated entity ids; default = every entity on disk")
    ap.add_argument("--cluster", type=str, default=None,
                    help="'uni' only: sweep ONE machine-type cluster (overrides --clients)")
    ap.add_argument("--arms", type=str, default="local,centralized,federated")
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--s1-epochs", type=int, default=8)
    ap.add_argument("--s2-epochs", type=int, default=8)
    ap.add_argument("--s1-rounds", type=int, default=8)
    ap.add_argument("--s2-rounds", type=int, default=8)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if args.cluster and args.builder != "uni":
        raise SystemExit("--cluster requires --builder uni (only that benchmark defines clusters).")

    PY = sys.executable
    SCRATCH.mkdir(parents=True, exist_ok=True)
    builder_script, prefix = BUILDERS[args.builder]
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    results = {}

    for L in lengths:
        name = f"{prefix}_t{L}"
        out = REPO / "data" / "raw" / name
        if args.rebuild or not (out / "train").exists():
            run([PY, REPO / "scripts" / builder_script, "--tier", args.tier,
                 "--train-length", L, "--output-dir", out, "--dataset-name", name,
                 "--overwrite"])

        # Resolve clients AFTER the build: ids depend on the builder and the tier.
        clients = ([c.strip() for c in args.clients.split(",") if c.strip()]
                   if args.clients else discover_clients(out, args.cluster))

        jp = SCRATCH / f"{name}{'_' + args.cluster if args.cluster else ''}.json"
        cmd = [PY, REPO / "pipeline" / "federated_eval.py",
               "--dataset", name, "--clients", ",".join(clients), "--arms", args.arms,
               "--seeds", args.seeds, "--s1-epochs", args.s1_epochs, "--s2-epochs", args.s2_epochs,
               "--s1-rounds", args.s1_rounds, "--s2-rounds", args.s2_rounds,
               "--local-epochs", args.local_epochs, "--batch", args.batch, "--out-json", jp]
        if args.cluster:
            cmd += ["--cluster", args.cluster]
        run(cmd)
        results[L] = json.loads(jp.read_text())

    def cell(L, arm, metric):
        return results[L]["summary"].get(arm, {}).get(metric)

    def fmt(m):
        return f"{m['mean']:.3f}±{m['std']:.3f}" if m else "-"

    scope = f"{prefix} ({args.cluster})" if args.cluster else prefix
    print(f"\n================ SCARCITY SWEEP [{scope}] — detection vus_pr vs per-client train length ================")
    print(f"{'train_len':>9} {'local':>15} {'centralized':>15} {'federated':>15} {'fed−local':>10}")
    for L in lengths:
        lo, ce, fe = cell(L, "local", "vus_pr"), cell(L, "centralized", "vus_pr"), cell(L, "federated", "vus_pr")
        gap = (fe["mean"] - lo["mean"]) if (fe and lo) else float("nan")
        print(f"{L:>9} {fmt(lo):>15} {fmt(ce):>15} {fmt(fe):>15} {gap:>+10.3f}")
    print("Expectation: fed−local gap GROWS as train_len shrinks (federation helps most when data is scarce).")

    print(f"\n================ SCARCITY SWEEP [{scope}] — counterfactual repair vs x_clean ================")
    print(f"{'train_len':>9} {'local':>15} {'centralized':>15} {'federated':>15}")
    for L in lengths:
        lo, ce, fe = (cell(L, a, "cf_repair_improvement") for a in ("local", "centralized", "federated"))
        print(f"{L:>9} {fmt(lo):>15} {fmt(ce):>15} {fmt(fe):>15}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
