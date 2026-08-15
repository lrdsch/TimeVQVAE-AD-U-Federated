#!/usr/bin/env python3.10
"""Qualità dei counterfactual: accordo, plausibilità, specificità.

Tesi che la sezione del paper porta avanti (modesta e verificabile): i
counterfactual prodotti dal modello federato sono di qualità paragonabile a
quelli di `local` e `centralized` su tutti e tre gli assi, quindi il vantaggio
della federazione è nella DETECTION, non nella spiegazione — e federare non
costa qualità della spiegazione.

Tre misure, tutte appaiate sulla stessa finestra e sulla stessa maschera:

  ACCORDO      MAE fra i counterfactual di client diversi, espressa in unità
               del null stocastico INTRA-modello (K campioni dello stesso
               client, stessa maschera). Rapporto ~1 = i client concordano
               quanto un singolo modello concorda con se stesso.

  PLAUSIBILITÀ Δ = [NLL(x) − NLL(x_cf)] / NLL(x) sulle SOLE posizioni
               riscritte, sotto il prior del client stesso, con x_cf
               RI-CODIFICATO (round-trip decode→encode). Positiva = la
               riparazione è più tipica dell'originale.

  SPECIFICITÀ  G = Δ(anomalo) − Δ(normale), con |M| appaiato. Senza il braccio
               normale «il counterfactual sembra normale» è vero per
               costruzione: se il generatore liscia tutto, G ≈ 0.

Vincoli di disegno (contromisure al cherry-picking):
  * la maschera viene dalle ETICHETTE, mai dal modello — così è identica fra
    arm e fra client, e |M| è costante per costruzione (la soglia paper-style
    non è una valuta confrontabile fra arm: misurato, 0,000-0,978);
  * le finestre sono selezionate da un algoritmo con seed fisso;
  * ogni divergenza è riportata in unità del null intra-modello;
  * `centralized` è UN SOLO modello salvato 5 volte: il suo accordo fra client
    è 1 per definizione, non per misura. Marcato `degenerate` nell'output.

Solo CPU, nessuna GPU: le schede sono della campagna c50.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import sys
from pathlib import Path

# Cap dei pool BLAS PRIMA di importare numpy/torch: torch.set_num_threads non
# tocca OpenMP/MKL, e su g2 (16 core, condivisa) un solo worker era arrivato a
# 1057% di CPU.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))

from config import Config                      # noqa: E402
from data import make_dataloaders              # noqa: E402
from stage1 import load_stage1                 # noqa: E402
from stage2 import Stage2System, counterfactual, load_stage2  # noqa: E402

torch.set_num_threads(2)

BUILD = "ucr_split_w2p"

# arm -> (tag, nome della directory dell'arm, n_client attesi)
ARMS = {
    "local":       ("zn_main", "local", 5),
    "centralized": ("zn_main", "centralized", 5),
    "A2":          ("zn_a2", "federated_enc_fedavg_bn-shared_prior-partial", 5),
}


def _fill(obj, src: dict) -> None:
    """Ricostruisce una Config dal cfg_dict salvato nel checkpoint."""
    for f in dataclasses.fields(obj):
        if f.name not in src:
            continue
        cur, val = getattr(obj, f.name), src[f.name]
        if dataclasses.is_dataclass(cur) and isinstance(val, dict):
            _fill(cur, val)
        elif isinstance(cur, tuple) and isinstance(val, list):
            setattr(obj, f.name, tuple(val))
        else:
            setattr(obj, f.name, val)


def client_dirs(tag: str, arm_dir: str, series: str, n: int) -> list[Path]:
    base = REPO / "artifacts" / "runs" / tag / "ckpt" / BUILD / series / "seed0" / arm_dir
    return [base / f"{series}_p{c}" for c in range(n)]


# ─── Selezione delle finestre e delle maschere ───────────────────────────────

def _col_slices(w_time: int, w_lat: int) -> list[tuple[int, int]]:
    """Intervallo di timestep coperto da ciascuna colonna latente."""
    edges = np.linspace(0, w_time, w_lat + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(w_lat)]


def pick_windows(dataset, w_lat: int, n_tp: int, n_norm: int, seed: int):
    """Finestre anomale (con ≥10% di contesto normale) e finestre normali.

    Ritorna (tp, norm): liste di dict con `idx`, `x`, `cols` (colonne latenti da
    riscrivere). Nessuna scelta a mano: le anomale sono campionate a passo
    costante lungo l'evento, le normali a caso con seed fisso.
    """
    rng = np.random.default_rng(seed)
    n_win = len(dataset)
    lab_pos, lab_none = [], []
    w_time = dataset.window_length
    slices = _col_slices(w_time, w_lat)

    for i in range(n_win):
        y = dataset.indices[i]
        rec = dataset.records[y.record_index]
        if rec.y is None:
            continue
        lab = np.asarray(rec.y[y.start:y.stop])
        frac = float((lab > 0).mean())
        if frac <= 0.0:
            lab_none.append(i)
        elif frac <= 0.90:                       # ≥10% di contesto normale
            lab_pos.append((i, frac))

    tp = []
    if lab_pos:
        take = np.linspace(0, len(lab_pos) - 1, min(n_tp, len(lab_pos))).round().astype(int)
        for t in sorted(set(take.tolist())):
            i, _ = lab_pos[t]
            item = dataset[i]
            lab = item["labels"].numpy()
            cols = [c for c, (a, b) in enumerate(slices) if (lab[a:b] > 0).any()]
            if not cols or len(cols) >= w_lat:   # niente da riscrivere / niente contesto
                continue
            tp.append({"idx": i, "x": item["inputs"], "cols": cols,
                       "start": int(item["metadata"]["window_start"])})

    if not tp:
        return [], []

    k = int(np.median([len(t["cols"]) for t in tp]))     # |M| appaiato
    norm = []
    if lab_none:
        sel = rng.choice(len(lab_none), size=min(n_norm, len(lab_none)), replace=False)
        for s in sorted(sel.tolist()):
            i = lab_none[s]
            item = dataset[i]
            off = int(rng.integers(0, max(1, w_lat - k)))
            norm.append({"idx": i, "x": item["inputs"], "cols": list(range(off, off + k)),
                         "start": int(item["metadata"]["window_start"])})
    return tp, norm


def build_mask(cols_list: list[list[int]], B: int, C: int, F_: int, W: int) -> torch.Tensor:
    m = torch.zeros((B, C, F_, W), dtype=torch.bool)
    for b, cols in enumerate(cols_list):
        m[b, :, :, cols] = True
    return m


# ─── Metriche ────────────────────────────────────────────────────────────────

@torch.no_grad()
def plausibility(system: Stage2System, x: torch.Tensor, x_cf: torch.Tensor,
                 mask: torch.Tensor, new_indices: torch.Tensor | None = None) -> dict:
    """Δ = [NLL(x) − NLL(x_cf)] / NLL(x) sulle posizioni riscritte.

    Primaria: x_cf RI-CODIFICATO dall'encoder (round-trip decode→encode). Non
    valuta i token campionati — che per costruzione sono probabili sotto il
    prior, quindi misurarli sarebbe circolare — ma quelli che l'encoder legge
    davvero nella forma d'onda prodotta.

    `direct` è il controllo circolare, riportato solo per separare «il prior
    campiona male» da «il round-trip perde la riparazione».
    """
    B = x.shape[0]
    _, idx_o, sp = system.stage1.encode_tokens(x)
    _, idx_c, _ = system.stage1.encode_tokens(x_cf)
    F_, W = int(sp[0]), int(sp[1])
    s_o = system.prior.score_tokens(idx_o.long().reshape(B, -1))
    s_c = system.prior.score_tokens(idx_c.long().reshape(B, -1))
    m = mask.to(s_o.device)
    npos = m.sum(dim=(1, 2, 3)).clamp_min(1)
    nll_o = (s_o * m).sum(dim=(1, 2, 3)) / npos
    nll_c = (s_c * m).sum(dim=(1, 2, 3)) / npos
    out = {
        "delta": float(((nll_o - nll_c) / nll_o.abs().clamp_min(1e-8)).mean()),
        "nll_orig": float(nll_o.mean()),
        "nll_cf": float(nll_c.mean()),
        "token_kept": float((idx_o.reshape(B, -1) == idx_c.reshape(B, -1)).float().mean()),
    }
    if new_indices is not None:
        s_d = system.prior.score_tokens(new_indices.long().reshape(B, -1))
        nll_d = (s_d * m).sum(dim=(1, 2, 3)) / npos
        out["delta_direct"] = float(((nll_o - nll_d) / nll_o.abs().clamp_min(1e-8)).mean())
        out["nll_direct"] = float(nll_d.mean())
    return out


def pairwise_mae(stack: np.ndarray) -> tuple[float, float]:
    """MAE e correlazione medie a coppie fra le righe di `stack` (n, ...)."""
    n = stack.shape[0]
    if n < 2:
        return float("nan"), float("nan")
    maes, corrs = [], []
    for i, j in itertools.combinations(range(n), 2):
        maes.append(float(np.abs(stack[i] - stack[j]).mean()))
        a, b = stack[i].ravel(), stack[j].ravel()
        if a.std() > 1e-12 and b.std() > 1e-12:
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(maes)), (float(np.mean(corrs)) if corrs else float("nan"))


# ─── Cella: un arm su una serie ──────────────────────────────────────────────

def run_cell(arm: str, series: str, n_tp: int, n_norm: int, n_null: int,
             seed: int, dump_dir: Path | None) -> dict | None:
    tag, arm_dir, n_cli = ARMS[arm]
    dirs = client_dirs(tag, arm_dir, series, n_cli)
    dirs = [d for d in dirs if (d / "stage1.ckpt").exists() and (d / "stage2.ckpt").exists()]
    if not dirs:
        return None

    blob = torch.load(dirs[0] / "stage1.ckpt", map_location="cpu", weights_only=False)
    cfg = Config()
    _fill(cfg, blob["cfg_dict"])
    cfg.dataset.num_workers = 0
    assert cfg.dataset.window_normalization == "zscore", (
        f"{arm}/{series}: window_normalization={cfg.dataset.window_normalization!r}, "
        "checkpoint pre-cutoff — non usabile"
    )
    data = make_dataloaders(cfg, stage="eval")

    probe = data.test_dataset[0]["inputs"][None].float()
    s1 = load_stage1(str(dirs[0] / "stage1.ckpt"), cfg, probe, device=torch.device("cpu"))
    with torch.no_grad():
        _, _, sp = s1.encode_tokens(probe)
    F_, W_lat = int(sp[0]), int(sp[1])
    del s1

    tp, norm = pick_windows(data.test_dataset, W_lat, n_tp, n_norm, seed)
    if not tp:
        return None

    x_tp = torch.stack([t["x"] for t in tp]).float()
    x_no = torch.stack([t["x"] for t in norm]).float() if norm else None
    C = x_tp.shape[1]
    m_tp = build_mask([t["cols"] for t in tp], x_tp.shape[0], C, F_, W_lat)
    m_no = (build_mask([t["cols"] for t in norm], x_no.shape[0], C, F_, W_lat)
            if x_no is not None else None)

    per_client, cf_stack, null_ratios = [], [], []
    for d in dirs:
        system = load_stage2(str(d / "stage2.ckpt"), cfg, str(d / "stage1.ckpt"),
                             probe, device="cpu")
        torch.manual_seed(seed)
        out = counterfactual(system, x_tp, token_mask=m_tp, n_samples=1, greedy=False)
        x_cf = out["x_cf"]
        rec = dict(client=d.name)
        p_tp = plausibility(system, x_tp, x_cf, m_tp, out["new_indices"])
        rec["plaus_tp"] = p_tp["delta"]
        rec["diag_tp"] = p_tp
        if x_no is not None:
            torch.manual_seed(seed)
            out_n = counterfactual(system, x_no, token_mask=m_no, n_samples=1, greedy=False)
            p_no = plausibility(system, x_no, out_n["x_cf"], m_no, out_n["new_indices"])
            rec["plaus_norm"] = p_no["delta"]
            rec["diag_norm"] = p_no
            rec["specificity"] = rec["plaus_tp"] - rec["plaus_norm"]
            rec["specificity_direct"] = p_tp["delta_direct"] - p_no["delta_direct"]

        # null stocastico intra-modello: K campioni, stesso client, stessa maschera
        torch.manual_seed(seed + 1)
        out_k = counterfactual(system, x_tp, token_mask=m_tp, n_samples=n_null, greedy=False)
        xk = out_k["x_cf"].numpy().reshape(x_tp.shape[0], n_null, *x_tp.shape[1:])
        d_k = xk - x_tp.numpy()[:, None]
        null_mae = float(np.mean([pairwise_mae(d_k[b])[0] for b in range(d_k.shape[0])]))
        rec["null_mae"] = null_mae
        null_ratios.append(null_mae)

        cf_stack.append((x_cf.numpy() - x_tp.numpy()))     # la MODIFICA, non x_cf
        per_client.append(rec)
        del system

    # accordo fra client, finestra per finestra, in unità del null
    D = np.stack(cf_stack)                                  # (n_cli, B, C, T)
    inter = [pairwise_mae(D[:, b])[0] for b in range(D.shape[1])]
    inter_corr = [pairwise_mae(D[:, b])[1] for b in range(D.shape[1])]
    null_mean = float(np.mean(null_ratios))

    res = {
        "arm": arm, "series": series, "tag": tag,
        "n_clients": len(dirs), "n_tp": len(tp), "n_norm": len(norm),
        "W_lat": W_lat, "F": F_, "masked_cols_median": int(np.median([len(t["cols"]) for t in tp])),
        "inter_client_mae": float(np.mean(inter)),
        "inter_client_corr": float(np.nanmean(inter_corr)),
        "intra_model_mae": null_mean,
        "agreement_ratio": float(np.mean(inter) / null_mean) if null_mean > 0 else float("nan"),
        "plaus_tp": float(np.mean([r["plaus_tp"] for r in per_client])),
        "plaus_norm": (float(np.mean([r["plaus_norm"] for r in per_client]))
                       if "plaus_norm" in per_client[0] else None),
        "specificity": (float(np.mean([r["specificity"] for r in per_client]))
                        if "specificity" in per_client[0] else None),
        "specificity_direct": (float(np.mean([r["specificity_direct"] for r in per_client]))
                               if "specificity_direct" in per_client[0] else None),
        "plaus_tp_direct": float(np.mean([r["diag_tp"]["delta_direct"] for r in per_client])),
        "token_kept_tp": float(np.mean([r["diag_tp"]["token_kept"] for r in per_client])),
        "degenerate_agreement": arm == "centralized",
        "per_client": per_client,
    }

    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dump_dir / f"{arm}_{series}.npz",
            x=x_tp.numpy(), d=D,
            starts=np.array([t["start"] for t in tp]),
            cols=np.array([np.isin(np.arange(W_lat), t["cols"]) for t in tp]),
        )
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True, help="lista separata da virgole")
    ap.add_argument("--arms", default="A2,local,centralized")
    ap.add_argument("--n-tp", type=int, default=8)
    ap.add_argument("--n-norm", type=int, default=8)
    ap.add_argument("--n-null", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", default=None, help="directory per gli array delle figure")
    a = ap.parse_args()

    out = Path(a.out).resolve()
    assert "artifacts" not in out.parts, "l'output non va sotto artifacts/"
    out.parent.mkdir(parents=True, exist_ok=True)
    dump = Path(a.dump).resolve() if a.dump else None

    rows = []
    for series in a.series.split(","):
        for arm in a.arms.split(","):
            try:
                r = run_cell(arm, series.strip(), a.n_tp, a.n_norm, a.n_null, a.seed, dump)
            except AssertionError as e:
                print(f"[skip] {arm}/{series}: {e}", flush=True)
                continue
            if r is None:
                print(f"[miss] {arm}/{series}", flush=True)
                continue
            rows.append(r)
            print(f"{r['series']:<9} {r['arm']:<12} "
                  f"accordo {r['agreement_ratio']:.2f}× null (corr {r['inter_client_corr']:.3f}) · "
                  f"plaus {r['plaus_tp']:+.3f} · spec {r['specificity']:+.3f}"
                  if r["specificity"] is not None else
                  f"{r['series']:<9} {r['arm']:<12} accordo {r['agreement_ratio']:.2f}×", flush=True)
            out.write_text(json.dumps(rows, indent=1))
    out.write_text(json.dumps(rows, indent=1))
    print(f"\n-> {out}  ({len(rows)} celle)")


if __name__ == "__main__":
    main()
