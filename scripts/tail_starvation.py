"""
Quantify how data-starved each client's local prior is in the TAILS.

The deep prior estimates p(token | context). Its variance at a given context ~
1/(examples of that context). We proxy "context" at two granularities on the
SHARED (cb_only merged) codebook, so counts are comparable across clients:
  * unigram  : per-codeword occupancy         (marginal normal support)
  * bigram   : time-adjacent (t, t+1) per freq (a conditional-context proxy)

For each client we report examples/context and how many contexts are STARVED
(seen < THR times) or SINGLETON (seen once) — the maximally high-variance cells
where the local prior is essentially guessing. POOLED = all clients in the
cluster summed: the same contexts, now well-sampled. The pooled/local ratio is
the variance-reduction factor a joint fit captures and a local fit cannot.

Usage: CUDA_VISIBLE_DEVICES=1 python scripts/tail_starvation.py --dataset wsd_fed --clusters c0,c2,c3 --seed 0
"""
from __future__ import annotations
import argparse, copy, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pipeline"))
from data import make_dataloaders                       # noqa: E402
from stage1 import load_stage1                          # noqa: E402
from federated import resolve_clients                   # noqa: E402
from mixture_eval import _build_cfg, CONVERGED          # noqa: E402

THR = 30            # "starved" threshold (examples per context)


@torch.no_grad()
def _tokens_for(entity, cfg, stage1, device):
    """(N, F, W) int token grid over the client's train windows."""
    c = copy.deepcopy(cfg); c.dataset.entity_id = entity
    loader = make_dataloaders(c, stage="stage2").train_loader
    grids = []
    for b in loader:
        x = b["inputs"].to(device, non_blocking=True)
        _, idx, spatial = stage1.encode_tokens(x)        # idx (B, C, F*W)
        F_, W = int(spatial[0]), int(spatial[1])
        g = idx[:, 0, :].reshape(idx.shape[0], F_, W).cpu().numpy()   # C=1 univariate
        grids.append(g)
    return np.concatenate(grids, 0) if grids else np.zeros((0, 1, 1), int)


def _uni_counts(grids, K):
    return np.bincount(grids.reshape(-1), minlength=K).astype(np.int64)   # (K,)


def _bi_counts(grids, K):
    """Time-adjacent bigram counts within each freq row: dict over K*K."""
    a = grids[:, :, :-1].reshape(-1); b = grids[:, :, 1:].reshape(-1)
    key = a.astype(np.int64) * K + b.astype(np.int64)
    return np.bincount(key, minlength=K * K).astype(np.int64)             # (K*K,)


def _stats(uni, bi, nwin):
    active = uni[uni > 0]
    med = int(np.median(active)) if active.size else 0
    p10 = int(np.percentile(active, 10)) if active.size else 0
    starved_cw = int((active < THR).sum())
    bi_seen = bi[bi > 0]
    bi_singleton = float((bi_seen == 1).mean()) if bi_seen.size else float("nan")
    bi_starved = float((bi_seen < THR).mean()) if bi_seen.size else float("nan")
    return dict(nwin=nwin, active=int(active.size), med=med, p10=p10,
                starved_cw=starved_cw, bi_distinct=int(bi_seen.size),
                bi_singleton=bi_singleton, bi_starved=bi_starved)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--clusters", default="c0,c2,c3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _build_cfg(args.dataset, 64)
    K = cfg.quantizer.codebook_size
    clusters = [c.strip() for c in args.clusters.split(",")]

    for cluster in clusters:
        entities = resolve_clients(cfg, None, cluster)
        cb = CONVERGED / args.dataset / cluster / f"seed{args.seed}" / "federated_cb_only"
        have = [e for e in entities if (cb / e / "stage1.ckpt").exists()]
        if not have:
            print(f"\n### {cluster}: no cb_only ckpts (seed{args.seed}) — skip"); continue
        c0 = copy.deepcopy(cfg); c0.dataset.entity_id = have[0]
        ex = next(iter(make_dataloaders(c0, stage="eval").train_loader))["inputs"][:1].cpu()
        stage1 = load_stage1(cb / have[0] / "stage1.ckpt", cfg, ex, device=device); stage1.eval()

        print(f"\n### {args.dataset} {cluster} — examples per context (THR<{THR}=starved) ###")
        print(f'{"client":<12}{"n_win":>7}{"act_cw":>7}{"med/cw":>7}{"p10":>5}{"starv_cw":>9}'
              f'{"bi_dist":>8}{"bi_1x%":>7}{"bi_starv%":>10}')
        uni_tot = np.zeros(K, np.int64); bi_tot = np.zeros(K * K, np.int64); nwin_tot = 0
        for e in have:
            g = _tokens_for(e, cfg, stage1, device)
            uni = _uni_counts(g, K); bi = _bi_counts(g, K)
            uni_tot += uni; bi_tot += bi; nwin_tot += g.shape[0]
            s = _stats(uni, bi, g.shape[0])
            print(f'{e:<12}{s["nwin"]:>7}{s["active"]:>7}{s["med"]:>7}{s["p10"]:>5}{s["starved_cw"]:>9}'
                  f'{s["bi_distinct"]:>8}{s["bi_singleton"]*100:>6.0f}{s["bi_starved"]*100:>9.0f}')
        sp = _stats(uni_tot, bi_tot, nwin_tot)
        print(f'{"POOLED":<12}{sp["nwin"]:>7}{sp["active"]:>7}{sp["med"]:>7}{sp["p10"]:>5}{sp["starved_cw"]:>9}'
              f'{sp["bi_distinct"]:>8}{sp["bi_singleton"]*100:>6.0f}{sp["bi_starved"]*100:>9.0f}')
        # variance-reduction headline
        loc_med = np.median([_stats(_uni_counts(_tokens_for(e, cfg, stage1, device), K),
                                    np.zeros(K*K, np.int64), 0)["med"] for e in have])
        print(f'  -> pooled median examples/codeword = {sp["med"]}  vs  local median = {int(loc_med)}  '
              f'(~{sp["med"]/max(1,loc_med):.0f}x more samples per context)')


if __name__ == "__main__":
    main()
