"""Shared resolver for the converged per-entity artifact tree.

History: `scripts/run_fed_converged.sh` writes its output under
`artifacts/fed_eval/converged/<ds>/<cluster>/seed<N>/<arm>/<entity>/`. The current
converged runners (`run_converged_all.sh`, `run_cbonly_converged.sh`, and
`pipeline/federated_eval.py --protocol converged`) instead write the same leaf shape
WITHOUT the `converged/` segment, i.e. `artifacts/fed_eval/<ds>/<cluster>/seed<N>/...`,
and the old `converged/` tree was deleted in the 2026-07-23 purge.

Both layouts are otherwise identical, so every analysis script resolves the root through here.

LEGACY BRANCH REMOVED 2026-07-27. `conv_root` used to PREFER `<base>/converged/<ds>` whenever
that directory existed, falling back to the live tree otherwise. Since the purge deleted the
legacy tree the branch was dormant — but it was a live landmine: a single run of
`scripts/pretrain_ucr_body.py` (which by its own docstring drops its result "into the converged
tree as the `ucr_pretrained` arm") or of `run_fed_converged.sh` RECREATES that directory holding
ONE arm, and from that moment every consumer of conv_root silently resolves to the near-empty
tree, with no error. Nine scripts read through here (aggregate_all, token_partition,
codebook_recon_probe, retrain_{local_10k,bn_fair_eval,capacity}, local_recon_autopsy,
sweep_window_width, plot_retrain_train_test). The branch's stated purpose — keeping the
18-round run_fed_converged.sh working — is moot under the train-to-convergence hard rule
(2026-07-24), so the root is now unconditional.

Usage (scripts/ is sys.path[0] when a script is run as `python scripts/foo.py`):
    from _fedpaths import conv_root, conv_seed_root
    root = conv_seed_root(ds, cluster, seed)      # .../<ds>/<cl>/seed<N>
"""
from __future__ import annotations

import os

_BASE = "artifacts/fed_eval"


def conv_root(ds: str, base: str = _BASE) -> str:
    """Dataset root of the converged tree: always `<base>/<ds>`.

    Unconditional by design — see the module docstring. Do NOT reinstate a
    `<base>/converged/<ds>` preference: it silently hijacks nine analysis scripts the
    moment any runner recreates that directory."""
    return os.path.join(base, ds)


def conv_seed_root(ds: str, cluster: str, seed, base: str = _BASE) -> str:
    """`<conv_root(ds)>/<cluster>/seed<seed>` — the dir holding the per-arm subdirs."""
    return os.path.join(conv_root(ds, base), str(cluster), f"seed{seed}")


def conv_glob(ds: str, pattern: str, base: str = _BASE) -> str:
    """Glob rooted at the dataset root, e.g. conv_glob(ds, '*/seed*/*/*/report.json')."""
    return os.path.join(conv_root(ds, base), pattern)
