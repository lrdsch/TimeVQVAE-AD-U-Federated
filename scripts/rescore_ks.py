#!/usr/bin/env python3
"""rescore_ks.py — RI-PUNTEGGIA checkpoint gia' addestrati a larghezza di maschera
`ks` arbitraria. NON allena nulla, non scrive niente dentro artifacts/.

  destinazione definitiva quando smette di essere una sonda: scripts/rescore_ks.py
  (per ora vive nello scratchpad: il repo e' in sola lettura per questo compito)

PERCHE' FUNZIONA SENZA RIALLENARE
  I rate di scoring sono letti SOLO nel percorso di punteggio, mai nel forward di
  training: model/prior.py:808 `rates = list(self.score_window_size_rates)` dentro
  `score_tokens_per_rate`, mentre i `def forward` (model/prior.py:332/548/776) non li
  guardano. Sono un attributo Python (model/prior.py:651), non un buffer: quindi
  `load_state_dict(..., strict=True)` (pipeline/stage2.py:463) NON li sovrascrive e
  `Stage2System` li prende dalla cfg VIVA (pipeline/stage2.py:113).

DAL RATE AL ks
  model/prior.py:44-64 `_paper_kernel_size(W, rate)` forza il kernel DISPARI e
  arrotonda in su. W e' il tempo LATENTE (22..42 nella coorte, NON una costante),
  quindi un rate fisso da' ks DIVERSI da serie a serie. Questo script prende il ks
  ASSOLUTO e ricava il rate per cella: rate = (ks + 0.5) / W_lat, con assert che
  `_paper_kernel_size` restituisca esattamente il ks chiesto.

CACHE
  Disattivata (`cfg.evaluation.use_score_cache = False`, config.py:402) e comunque mai
  toccata: questo script non chiama `detect.detect()` ma i suoi primitivi, quindi non
  passa neanche vicino a `_save_score_cache`. In piu' controlla mtime+size di
  `<arm>/detect_score_cache.npz` prima e dopo e fallisce se cambiano: quella slot e'
  UNA sola per ARM (pipeline/detect.py:1150-1154, `anchor.parent.parent`) ed e'
  condivisa dai 5 client, quindi scriverci sopra distruggerebbe la cache della run
  originale.

USO
  PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
  CELL=artifacts/runs/zn_main/ckpt/ucr_split_w2p/ucr_011/seed0/local/ucr_011_p0

  # CPU (sempre lecita, nessuna GPU toccata)
  $PY rescore_ks.py --client-dir $CELL --ks 1,3,paper --device cpu --threads 2 \
      --out /tmp/.../scratchpad/ksweep

  # GPU: SOLO la scheda che il guardiano concede (su g2 oggi = 1, GPU0 e' di ssanchez)
  CUDA_VISIBLE_DEVICES="$(bash scripts/_g2_gpu.sh)" \
      $PY rescore_ks.py --client-dir $CELL --ks 1,3,paper --device cuda --out ...

  --ks accetta interi dispari e la parola chiave `paper` (= i 3 rate di config.py:257
  sommati, cioe' esattamente cio' che sta gia' su disco: e' il controllo di identita').
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/leonardo/PhD/TimeVQVAE-AD-U-Federated")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))

from config import Config                                            # noqa: E402
from data import make_dataloaders                                    # noqa: E402
from stage1 import load_stage1                                       # noqa: E402
from stage2 import load_stage2                                       # noqa: E402
from utils import seed_everything                                    # noqa: E402
from model.prior import _paper_kernel_size                           # noqa: E402
import detect as D                                                   # noqa: E402
import metrics_core as MC                                            # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score   # noqa: E402


# ── cfg dal checkpoint, mai dalla CLI ────────────────────────────────────────
# stessa ragione di scripts/latent_probe.py:135 `config_from_ckpt`: finestra,
# z-norm, scaler e tolleranza DEVONO essere quelli del training.
def _fill(obj, src: dict) -> None:
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


def config_from_ckpt(stage2_ckpt: Path) -> tuple[Config, dict]:
    blob = torch.load(str(stage2_ckpt), map_location="cpu", weights_only=False)
    cfg = Config()
    _fill(cfg, blob["cfg_dict"])
    return cfg, blob["cfg_dict"]


def _stat(p: Path):
    s = p.stat()
    return {"mtime": int(s.st_mtime), "size": int(s.st_size)}


def _guard_device(dev: str) -> torch.device:
    """Su g2 GPU0 e' di un altro utente. Nessun default implicito: o CPU, o una
    lista CUDA_VISIBLE_DEVICES esplicita che non contiene lo 0."""
    if dev == "cpu":
        return torch.device("cpu")
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not vis:
        raise SystemExit(
            "--device cuda richiede CUDA_VISIBLE_DEVICES esplicito.\n"
            "  su g2:  export CUDA_VISIBLE_DEVICES=\"$(bash scripts/_g2_gpu.sh)\"")
    ids = [x.strip() for x in vis.split(",") if x.strip()]
    if "0" in ids:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={vis!r} tocca GPU0 — vietato su g2 (ssanchez).")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA non disponibile in questo processo.")
    return torch.device("cuda")


# ── controllo §2bis della pre-registrazione: SOLO-CENTRO ─────────────────────
# `ks` muove DUE variabili insieme (PREREG_KS_SWEEP_2026-08-09 §2bis):
#   (A) quanto contesto e' nascosto al prior  — la variabile del meccanismo;
#   (B) la larghezza della MEDIA che produce lo score, perche' model/prior.py:843 fa
#       `out[ri,:,:,:,w] = -gathered[j,:,:,:,lo:hi].mean(dim=-1)` — un passa-basso di
#       ampiezza ks sull'asse latente.
# Un picco stretto sull'anomalia sopravvive a ks=1 e viene SPALMATO a ks=21 per (B) sole.
# Questa variante maschera lo STESSO blocco di ks colonne ma legge la NLL della SOLA
# colonna centrale: (A) invariato, (B) azzerato. Se solo-centro ~ ks1 il guadagno era il
# filtro; se ~ paper era il contesto.
# Il resto e' una replica riga-per-riga di model/prior.py:805-845: se quel metodo cambia,
# questa copia va riallineata (l'assert su latent_time e' la sveglia minima).
def _bind_center_only(prior) -> None:
    @torch.no_grad()
    def score_tokens_per_rate(tokens: torch.Tensor) -> torch.Tensor:
        B, N = tokens.shape
        C, F_, W = prior.latent_channels, prior.latent_freq, prior.latent_time
        assert C * F_ * W == N, f"seq_len={N} != C*F*W={C*F_*W}"
        rates = list(prior.score_window_size_rates)
        out = torch.zeros(len(rates), B, C, F_, W, device=tokens.device)
        target_rows = max(1, int(getattr(prior, "score_mask_chunk", 1)))
        mb = max(1, min(W, target_rows // max(1, B)))
        base = tokens.reshape(B, C, F_, W)
        for ri, rate in enumerate(rates):
            ks = _paper_kernel_size(W, rate)
            half = ks // 2
            ranges = [(max(0, w - half), min(W, w + half + 1)) for w in range(W)]
            for c0 in range(0, W, mb):
                cols = list(range(c0, min(c0 + mb, W)))
                nc = len(cols)
                masked = base.unsqueeze(0).repeat(nc, 1, 1, 1, 1)
                for j, w in enumerate(cols):
                    lo, hi = ranges[w]
                    masked[j, :, :, :, lo:hi] = prior.mask_token_id
                logits = prior._logits(masked.reshape(nc * B, N))
                lp = logits.log_softmax(dim=-1).reshape(nc, B, C, F_, W, -1)
                tgt = base.reshape(1, B, C, F_, W, 1).expand(nc, -1, -1, -1, -1, 1)
                gathered = lp.gather(-1, tgt).squeeze(-1)
                for j, w in enumerate(cols):
                    # UNICA differenza da prior.py:843: niente .mean(lo:hi), solo il centro.
                    out[ri, :, :, :, w] = -gathered[j, :, :, :, w]
        return out

    prior.score_tokens_per_rate = score_tokens_per_rate   # attributo d'istanza:
    # pipeline/stage2.py:231 fa `self.prior.score_tokens_per_rate(tokens)` ⇒ lo raccoglie.


def _readouts(labels: np.ndarray, scores: np.ndarray, preds: np.ndarray,
              cfg: Config) -> dict:
    """SOLO le letture pre-registrate. Deliberatamente NON chiama
    `detect._detection_metrics` ne' `metrics_core.evaluate_scores`: PATE,
    affiliation e la suite unificata costano ~90 s per client su g2 (misurati) e
    nessuna di esse entra nel criterio di lettura."""
    tol = int(cfg.evaluation.paper_metrics_tolerance)
    out = {
        "auprc": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
    }
    out.update({k: float(v) for k, v in MC.vus_metrics(labels, scores, tol).items()})
    out.update(D._paper_metrics(labels, scores, cfg))          # top-K @tol e @100
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out["precision"] = float(prec)
    out["recall"] = float(rec)
    out["f1"] = float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--client-dir", required=True,
                    help="cartella di UN client: contiene stage1.ckpt e stage2.ckpt")
    ap.add_argument("--ks", default="1",
                    help="lista: interi dispari e/o 'paper' (i 3 rate di config.py:257)")
    ap.add_argument("--out", required=True, help="cartella di output (MAI dentro artifacts/)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=2, help="torch CPU threads")
    ap.add_argument("--dry-run", action="store_true",
                    help="stampa il piano (ks -> rate) e esce senza forward")
    ap.add_argument("--center-only", action="store_true",
                    help="controllo §2bis: score = NLL della SOLA colonna centrale invece "
                         "della media sul blocco mascherato. Separa 'meno contesto' da "
                         "'meno smoothing'. Le varianti prendono il suffisso _c.")
    a = ap.parse_args()

    torch.set_num_threads(max(1, int(a.threads)))
    d = Path(a.client_dir).resolve()
    s1p, s2p = d / "stage1.ckpt", d / "stage2.ckpt"
    for p in (s1p, s2p):
        if not p.exists():
            raise SystemExit(f"manca {p}")
    out_root = Path(a.out).resolve()
    if str(out_root).startswith(str(REPO / "artifacts")):
        raise SystemExit("--out non puo' stare dentro artifacts/: gli artefatti sono immutabili.")
    out_root.mkdir(parents=True, exist_ok=True)

    # sentinella sulla cache condivisa a livello di ARM (detect.py:1150-1154)
    cache_file = s2p.parent.parent / "detect_score_cache.npz"
    cache_before = _stat(cache_file) if cache_file.exists() else None

    cfg, cfg_dict = config_from_ckpt(s2p)
    cfg.evaluation.use_score_cache = False        # cintura
    cfg.evaluation.save_plots = False
    cfg.dataset.num_workers = 0
    seed_everything(cfg.seed)                     # come detect.py:1265
    dev = _guard_device(a.device)

    W_lat_ckpt = int(torch.load(str(s2p), map_location="cpu",
                                weights_only=False)["state_dict"]
                     ["prior.time_embedding.weight"].shape[0])
    paper_rates = tuple(float(r) for r in cfg_dict["prior"]["score_window_size_rates"])

    plan = []
    for tok in [t.strip() for t in a.ks.split(",") if t.strip()]:
        if tok == "paper":
            plan.append(("paper", paper_rates))
            continue
        ks = int(tok)
        if ks % 2 == 0 or ks < 1:
            raise SystemExit(f"ks deve essere dispari e >=1 (chiesto {ks})")
        if ks > W_lat_ckpt:
            raise SystemExit(f"ks={ks} > W_lat={W_lat_ckpt} per {d.name}: cella non ammissibile")
        rate = (ks + 0.5) / W_lat_ckpt
        got = _paper_kernel_size(W_lat_ckpt, rate)
        assert got == ks, f"rate {rate} -> ks {got}, atteso {ks}"
        plan.append((f"ks{ks}", (rate,)))

    print(f"[rescore] cella   : {d}")
    print(f"[rescore] entity  : {cfg.dataset.entity_id}  W={cfg.dataset.window_length} "
          f"W_lat={W_lat_ckpt}  znorm={cfg.dataset.window_normalization} "
          f"tol={cfg.evaluation.paper_metrics_tolerance}")
    print(f"[rescore] paper   : rates {paper_rates} -> ks "
          f"{[_paper_kernel_size(W_lat_ckpt, r) for r in paper_rates]}")
    for name, rates in plan:
        print(f"[rescore] piano   : {name:8s} rates={tuple(round(r, 6) for r in rates)} "
              f"ks={[_paper_kernel_size(W_lat_ckpt, r) for r in rates]} n_tau={len(rates)}")
    if a.dry_run:
        return 0

    # ── modello: caricato UNA volta, i rate si cambiano sull'attributo ──────
    data = make_dataloaders(cfg, stage="eval")
    example = next(iter(data.train_loader))["inputs"][:1].cpu()
    stage1 = load_stage1(s1p, cfg, example, device=dev)
    stage2 = load_stage2(s2p, cfg, stage1_ckpt=s1p,
                         stage1_example_inputs=example, device=dev)
    assert int(stage2.prior.latent_time) == W_lat_ckpt
    assert not stage1.training and not stage2.prior.training
    if a.center_only:
        _bind_center_only(stage2.prior)
        print("[rescore] §2bis SOLO-CENTRO attivo: score = NLL della colonna centrale, "
              "niente media sul blocco mascherato")

    for name, rates in plan:
        if a.center_only:
            name = f"{name}_c"
        t0 = time.perf_counter()
        cfg.prior.score_window_size_rates = tuple(rates)
        stage2.prior.score_window_size_rates = tuple(rates)   # cio' che legge prior.py:808
        assert tuple(stage2.prior.score_window_size_rates) == tuple(rates)

        train_e = D._compute_entities_raw(stage1, stage2, cfg, data.train_records,
                                          record_per_rate=True)   # serve alla soglia
        test_e = D._compute_entities_raw(stage1, stage2, cfg, data.test_records)
        train_scores, _ = D._finalize_entities(train_e, cfg)
        test_scores, test_labels = D._finalize_entities(test_e, cfg)
        thr = D._fit_threshold_paper(train_e, cfg)
        preds = (test_scores > thr).astype(np.int64)
        m = _readouts(test_labels, test_scores, preds, cfg)
        dt = time.perf_counter() - t0

        cell = out_root / f"{cfg.dataset.entity_id}__{name}"
        cell.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cell / "scores.npz", train_scores=train_scores,
                            test_scores=test_scores, test_labels=test_labels)
        rec = {
            "client_dir": str(d), "entity_id": cfg.dataset.entity_id,
            "stage1": {"path": str(s1p), **_stat(s1p)},
            "stage2": {"path": str(s2p), **_stat(s2p)},
            "W_lat": W_lat_ckpt, "variant": name, "center_only": bool(a.center_only),
            "rates": [float(r) for r in rates], "n_tau": len(rates),
            "kernel_sizes": [_paper_kernel_size(W_lat_ckpt, r) for r in rates],
            "paper_rates": [float(r) for r in paper_rates],
            "threshold": float(thr), "device": str(dev),
            "seconds": round(dt, 1),
            "frozen": {  # tutto cio' che DEVE restare identico fra le varianti
                "window_length": cfg.dataset.window_length,
                "eval_stride": D._resolve_eval_stride(cfg),
                "window_normalization": cfg.dataset.window_normalization,
                "scaling": cfg.dataset.scaling,
                "weight_s_local": cfg.scoring.weight_s_local,
                "weight_s_prior": cfg.scoring.weight_s_prior,
                "scoring_normalization": cfg.scoring.normalization,
                "rolling_aggregation": cfg.scoring.rolling_aggregation,
                "use_impulse_term": cfg.scoring.use_impulse_term,
                "aggregation": cfg.scoring.aggregation,
                "threshold_q": cfg.threshold.q,
                "paper_metrics_tolerance": cfg.evaluation.paper_metrics_tolerance,
                "detect_amp": D._detect_amp_tag(cfg),
            },
            "metrics": m,
        }
        (cell / "rescore.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"[rescore] {name:8s} thr={thr:10.4f} auprc={m['auprc']:.6f} "
              f"auroc={m['auroc']:.6f} vus_pr={m['vus_pr']:.6f} "
              f"top1@{cfg.evaluation.paper_metrics_tolerance}="
              f"{m[f'paper_top1_acc_at_{cfg.evaluation.paper_metrics_tolerance}']:.0f} "
              f"[{dt:.0f}s] -> {cell}")

    cache_after = _stat(cache_file) if cache_file.exists() else None
    if cache_before != cache_after:
        raise SystemExit(f"ALLARME: {cache_file} e' cambiata ({cache_before} -> {cache_after})")
    print(f"[rescore] cache d'arm intatta: {cache_file} {cache_after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
