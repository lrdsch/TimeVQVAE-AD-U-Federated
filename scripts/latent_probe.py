#!/usr/bin/env python3
"""
=============================================================================
  latent_probe.py — i client parlano davvero lingue diverse?
=============================================================================

Sonda diagnostica sui checkpoint GIÀ SU DISCO. Non addestra niente: legge
`<run>/ckpt/<dataset>/<cluster>/seed<N>/<arm>/<client>/stage{1,2}.ckpt`,
fa solo forward, e scrive un JSON in `evidence/`.

Risponde a cinque domande, in ordine di dipendenza:

  D0  Gate di attribuzione.
      I 5 client ricevono la STESSA finestra di input? `val`/`test` sono copie
      byte-identiche per costruzione (build_ucr_split), ma lo scaler per-entità
      è fittato sul train di OGNI client, che è diverso. La z-norm per finestra
      cancella algebricamente qualunque affine per-entità — a meno del
      `clamp_min(1e-4)` su sigma (data.py:386). D0 lo VERIFICA invece di
      assumerlo. Se le finestre non sono identiche, il disaccordo di token
      misurato in D1 NON è attribuibile all'encoder e tutto il resto va letto
      con quella riserva.

  D1  Gli encoder parlano lingue diverse?
      Stessa probe, tokenizzata dai 5 stage-1. Il codebook è comune per
      costruzione (broadcast + merge suff-stat), quindi l'indice j è ancorato
      allo stesso vettore e_j su tutti i client: NON c'è una permutazione del
      vocabolario da allineare. Misura:
        * kappa  (accordo chance-corretto — il null è Σ_j p_A(j)p_B(j), NON
                  1/K: due client collassati su 3 codici concordano molto "per
                  caso" e l'accordo grezzo mente)
        * agree_hungarian (accordo dopo la migliore biiezione sui K codici) —
                  separa "permutazione funzionale" da "shift di covariata"
        * tie_margin — quanta parte del disaccordo è un quasi-pareggio
                  nell'argmin, cioè numerica e non semantica
        * uso del dizionario per client (codici attivi, perplexity)

  D2  Il disaccordo SPOSTA il punteggio? (la misura decisiva)
      Su `test` (identico per i 5 client): score = prior_A su token di E_A
      contro score = prior_A su token di E_B. Stesso prior, tokenizer straniero.
      Se il campo di punteggio è invariante, il problema "lingue diverse" è
      operativamente inesistente e l'allineamento del tokenizer non serve.
      D2b: rimpiazza SOLO i buffer BatchNorm di E_A con quelli di E_B (i pesi
      restano di A) — isola il contributo delle running stats, che in questo
      repo restano locali anche quando l'encoder è federato.

  D3  Barriera di interpolazione lineare (LMC).
      θ(α) = (1-α)θ_A + αθ_B, valutato sulla val di A tokenizzata da E_A.
      Gira su OGNI arm richiesto (--arm + --lmc-arms), perché i tre regimi
      misurano cose diverse:
        * federated_cb_only  prior interamente LOCALE (local_prefixes=("",)) su
                             codebook CONDIVISO → 5 prior indipendenti che
                             parlano lo stesso dizionario
        * local              anche il codebook è locale → nessuna condivisione
                             affatto. cb_only − local = quanto il dizionario
                             condiviso compra in trasferibilità del prior
        * federated          il body è FedAvg'd: lì la misura è (quasi) vacua per
                             costruzione, e lo script lo verifica invece di
                             assumerlo (max|Δ| sui tensori condivisi)
      Riporta DUE numeri separati, perché quando i tokenizer differiscono la
      barriera grezza confonde due cose:
        * endpoint_gap   = L(θ_B su dati/token di A) - L(θ_A)  → il gap LINGUISTICO
        * excess_barrier = max_α L(θ(α)) - max(L(θ_A), L(θ_B)) → la non-convessità PURA

  D4  Mismatch fra maschera di training e maschera di scoring.
      Il prior è addestrato a completare posizioni RANDOM sparse
      (prior.py `_mask_tokens_random`, mask_mode="random" di default) ed è
      VALUTATO completando blocchi CONTIGUI di colonne temporali
      (prior.py `score_tokens_per_rate`). D4 quantifica la differenza in NLL.
      Se è grande, l'ablazione `mask_mode=column` va fatta PRIMA di qualunque
      lavoro sulla federazione, perché sposterebbe ogni cella.

Tutto in fp32, anche se la run era fp16: una diagnostica non deve portarsi
dietro il rumore dell'AMP, e la quantizzazione è un argmin su K codici dove un
quasi-pareggio si ribalta.

Uso tipico:

    python scripts/latent_probe.py \
        --run-dir artifacts/runs/zn_main --dataset ucr_split_w2p \
        --cluster ucr_014 --arm federated_cb_only --lmc-arm local \
        --device cuda:0 --out evidence/latent_probe_ucr_014.json

    python scripts/latent_probe.py --run-dir artifacts/runs/zn_main \
        --dataset ucr_split_w2p --list          # cosa è pronto, senza girare
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))                # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from config import Config                                                       # noqa: E402
from data import make_dataloaders                                               # noqa: E402
from stage1 import Stage1VQVAE                                                  # noqa: E402
from stage2 import Stage2System, _flatten_token_indices                         # noqa: E402
from model.prior import _paper_kernel_size, _masked_count                       # noqa: E402
from utils import resolve_path                                                  # noqa: E402

# Prefissi dei tensori del prior che `federated_stage2` tiene LOCALI (non federati).
# Duplicati qui invece di importati da pipeline.federated per non trascinare dentro
# l'intero modulo di training (e i suoi side-effect su cudnn.benchmark).
LOCAL_PRIOR_PREFIXES = ("prior.channel_embedding", "prior.output_bias")


# ═════════════════════════════════════════════════════════════════════════════
#   Config dal checkpoint (mai dalla CLI)
# ═════════════════════════════════════════════════════════════════════════════

def _fill(obj, src: dict) -> None:
    """Riempie ricorsivamente una dataclass da un dict (l'inverso di asdict)."""
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


def config_from_ckpt(path: Path) -> tuple[Config, dict]:
    """La config della run, ricostruita dal `cfg_dict` dentro stage2.ckpt.

    MAI dalla CLI: la sonda deve usare per forza la stessa finestra, la stessa
    z-norm e lo stesso scaler con cui il modello è stato addestrato, altrimenti
    tokenizza input che il modello non ha mai visto e ogni numero è spazzatura.
    """
    blob = torch.load(str(path), map_location="cpu")
    cfg = Config()
    _fill(cfg, blob["cfg_dict"])
    return cfg, blob["cfg_dict"]


# ═════════════════════════════════════════════════════════════════════════════
#   Discovery — quali celle sono COMPLETE (mai sondare una run in volo)
# ═════════════════════════════════════════════════════════════════════════════

def discover(run_dir: Path, dataset: str, seed: int) -> dict[str, dict[str, list[str]]]:
    base = run_dir / "ckpt" / dataset
    out: dict[str, dict[str, list[str]]] = {}
    if not base.is_dir():
        return out
    for cl in sorted(p for p in base.iterdir() if p.is_dir()):
        sd = cl / f"seed{seed}"
        if not sd.is_dir():
            continue
        arms: dict[str, list[str]] = {}
        for arm in sorted(p for p in sd.iterdir() if p.is_dir() and not p.name.startswith("_")):
            clients = sorted(
                p.parent.name for p in arm.glob("*/stage2.ckpt")
                if (p.parent / "stage1.ckpt").exists()
            )
            if clients:
                arms[arm.name] = clients
        if arms:
            out[cl.name] = arms
    return out


def expected_clients(dataset: str, cluster: str, raw_root: Path) -> list[str]:
    """I client che il cluster DOVREBBE avere, da clusters.json."""
    p = raw_root / dataset / "clusters.json"
    if not p.exists():
        return []
    with p.open() as fh:
        return sorted(json.load(fh).get(cluster, []))


# ═════════════════════════════════════════════════════════════════════════════
#   Caricamento modelli e probe
# ═════════════════════════════════════════════════════════════════════════════

def build_probe(cfg: Config, client: str, split: str, stride: int, limit: int
                ) -> tuple[torch.Tensor, list[int], np.ndarray | None]:
    """Finestre della probe per UN client, con la pipeline dati esatta della run.

    Passa da `make_dataloaders`, non da un windower fatto a mano: lo scaler
    per-entità, la z-norm per finestra e lo stride devono essere gli stessi
    del training, e replicarli qui sarebbe un modo silenzioso di sbagliare.

    Ritorna (windows (N, C, T), start di ogni finestra, serie grezza o None).
    """
    c = Config()
    _fill(c, dataclasses.asdict(cfg))
    c.dataset.entity_id = client
    c.dataset.num_workers = 0
    dl = make_dataloaders(c, stage="eval")
    ds = {"val": dl.val_dataset, "test": dl.test_dataset}[split]
    idx = list(range(0, len(ds), max(1, stride)))
    if limit and len(idx) > limit:
        idx = [idx[i] for i in np.linspace(0, len(idx) - 1, limit).astype(int)]
    xs, starts = [], []
    for i in idx:
        item = ds[i]
        xs.append(item["inputs"])
        starts.append(int(item["metadata"]["window_start"]))
    recs = {"val": dl.val_records, "test": dl.test_records}[split]
    series = recs[0].X if recs else None
    return torch.stack(xs), starts, series


def load_model(arm_dir: Path, client: str, cfg: Config, example: torch.Tensor,
               device: torch.device) -> Stage2System:
    """stage2.ckpt è AUTOSUFFICIENTE: il suo state_dict contiene `stage1.*` e
    `prior.*` insieme (stage2.save_stage2_checkpoint). Costruiamo lo scheletro,
    materializziamo i layer lazy con un input reale, e carichiamo strict=True."""
    s1 = Stage1VQVAE(cfg)
    s1.eval()
    with torch.no_grad():
        s1(example.cpu())                                   # materializza encoder/decoder lazy
    s1.to(device)
    s2 = Stage2System(cfg, s1)
    # Ordine obbligato (lo stesso di federated._build_stage2_client): i moduli eager del
    # prior vanno sul device PRIMA di materialize, altrimenti materialize tokenizza su GPU
    # e passa gli indici a una token_embedding ancora su CPU. Il `.to(device)` finale
    # raccoglie anche le embedding posizionali 3D, che nascono lazy dentro materialize.
    s2.prior.to(device)
    s2.materialize(example.to(device))                      # scopre C,F,W e costruisce il prior
    s2.to(device)
    blob = torch.load(str(arm_dir / client / "stage2.ckpt"), map_location="cpu")
    s2.load_state_dict(blob["state_dict"], strict=True)
    s2.to(device).eval()
    for p in s2.parameters():
        p.requires_grad_(False)
    return s2


# ═════════════════════════════════════════════════════════════════════════════
#   Tokenizzazione + latenti (per il tie-margin)
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def tokenize(s2: Stage2System, x: torch.Tensor, device: torch.device,
             batch: int = 256) -> torch.Tensor:
    """(N, C, T) → (N, L) long, con lo STESSO percorso di Stage2System."""
    out = []
    for i in range(0, len(x), batch):
        _, idx, _ = s2.stage1.encode_tokens(x[i:i + batch].to(device))
        out.append(_flatten_token_indices(idx).long().cpu())
    return torch.cat(out)


@torch.no_grad()
def tie_margins(s2: Stage2System, x: torch.Tensor, device: torch.device,
                batch: int = 256) -> np.ndarray:
    """Margine relativo fra 1° e 2° codice più vicino, per ogni token.

        margin = (d2 - d1) / (d2 + d1)      ∈ [0, 1]

    Vicino a 0 = quasi-pareggio: quel token si ribalta per rumore numerico, non
    perché i due encoder "pensano" cose diverse. Replica la piega del VQ
    (vector_quantizer.SharedCodebookPerChannelVQ.forward) per restare nello
    stesso ordinamento dei token.
    """
    s1 = s2.stage1
    cb = s1.quantizer._vq.codebook.weight.detach()                       # (K, d)
    cb2 = (cb ** 2).sum(-1)                                              # (K,)
    outs = []
    for i in range(0, len(x), batch):
        tf = s1.transform(x[i:i + batch].to(device))
        if hasattr(s1.quantizer, "set_groups") and getattr(s1.quantizer, "groups", None) is None:
            s1.quantizer.set_groups(tf.spec.original_channels)
        latent = s1.encoder(tf)                                          # (B, C*d, F, W)
        B, D, F_, W = latent.shape
        C = s1.quantizer.groups
        d = D // C
        z = latent.reshape(B, C, d, F_, W).reshape(B * C, d, F_, W)
        z = z.permute(0, 2, 3, 1).reshape(B * C, F_ * W, d)              # (B*C, F*W, d)
        dist = (z ** 2).sum(-1, keepdim=True) - 2.0 * (z @ cb.T) + cb2[None, None, :]
        top2 = dist.topk(2, dim=-1, largest=False).values.clamp_min(0.0)
        d1, d2 = top2[..., 0], top2[..., 1]
        m = (d2 - d1) / (d2 + d1 + 1e-12)
        outs.append(m.reshape(B, C * F_ * W).cpu().numpy())
    return np.concatenate(outs)


# ═════════════════════════════════════════════════════════════════════════════
#   Scoring (percorso di detect.py, senza l'apparato di I/O)
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def score_field(s2: Stage2System, tokens: torch.Tensor, device: torch.device,
                batch: int = 64) -> torch.Tensor:
    """(N, L) token → (N, C, F, W) NLL sommata sui rate τ. Stessa chiamata di
    Stage2System.score_batch, ma con i token FORNITI invece che ricalcolati —
    è quello che permette di dare al prior di A i token di B."""
    s2._inform_latent_shape()
    out = []
    for i in range(0, len(tokens), batch):
        t = tokens[i:i + batch].to(device)
        out.append(s2.prior.score_tokens(t).float().cpu())
    return torch.cat(out)


def assemble(field: torch.Tensor, starts: list[int], T_series: int, T_win: int,
             impulse: bool = True) -> np.ndarray:
    """(N, C, F, W) → punteggio per timestep, come detect.py:

      interpolate nearest su T  →  media su F  →  somma rolling (no divisione)
      →  a_final = (a + MA(a, window_length)) / 2  →  media sui canali.
    """
    N, C, F_, W = field.shape
    up = Fnn.interpolate(field.reshape(N * C, F_, W), size=T_win, mode="nearest")
    per_ch = up.mean(dim=1).reshape(N, C, T_win).numpy()                 # (N, C, T_win)
    acc = np.zeros((T_series, C), dtype=np.float64)
    for i, s in enumerate(starts):
        acc[s:s + T_win] += per_ch[i].T
    if impulse:
        half = T_win // 2
        ma = np.empty_like(acc)
        for c in range(C):
            csum = np.concatenate([[0.0], np.cumsum(acc[:, c])])
            lo = np.clip(np.arange(T_series) - half, 0, T_series)
            hi = np.clip(np.arange(T_series) + half, 0, T_series)
            ma[:, c] = (csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)
        acc = 0.5 * (acc + ma)
    return acc.mean(axis=1)


# ═════════════════════════════════════════════════════════════════════════════
#   D0 — gate di attribuzione
# ═════════════════════════════════════════════════════════════════════════════

def run_d0(probes: dict[str, torch.Tensor]) -> dict:
    ref_name = next(iter(probes))
    ref = probes[ref_name]
    diffs = {c: float((probes[c] - ref).abs().max()) for c in probes if c != ref_name}
    worst = max(diffs.values()) if diffs else 0.0
    clean = worst < 1e-4
    return {
        "reference": ref_name,
        "max_abs_diff_per_client": diffs,
        "worst": worst,
        "inputs_identical": bool(clean),
        "verdict": (
            "PULITO — i 5 client ricevono finestre identiche: ogni disaccordo di token "
            "in D1 è attribuibile SOLO all'encoder."
            if clean else
            f"CONTAMINATO — le finestre differiscono di {worst:.2e}. Lo scaler per-entità "
            "non è stato cancellato dalla z-norm (clamp su sigma?). D1 misura encoder+input, "
            "non solo encoder."
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
#   D1 — accordo fra tokenizer
# ═════════════════════════════════════════════════════════════════════════════

def _per_window_stats(t: torch.Tensor, K: int) -> dict:
    """Risoluzione del tokenizer DENTRO una finestra, non sul corpus.

    `active_codes` conta i codici distinti su TUTTA la probe e confonde due cose:
    quanto è ricca la descrizione di una singola finestra, e quanto varia la
    descrizione fra finestre. Un client può usare 18 codici sul corpus con 3 per
    finestra (descrizione grossolana, molto variabile) o 18 per finestra
    (descrizione ricca, stereotipata). Solo il secondo numero dice se il
    tokenizer sta perdendo RISOLUZIONE.

    È una statistica INTRA-client, quindi resta valida anche sugli arm dove il
    codebook è locale (`local`) e gli indici non sono confrontabili fra client.
    """
    a = t.numpy()
    N, L = a.shape
    counts = np.zeros((N, K), dtype=np.int32)
    np.add.at(counts, (np.repeat(np.arange(N), L), a.reshape(-1)), 1)
    distinct = (counts > 0).sum(1)
    p = counts / float(L)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(p * np.where(p > 0, np.log(p), 0.0)).sum(1)
    ppl = np.exp(ent)                       # "codici efficaci" per finestra
    return {
        "tokens_per_window": int(L),
        "distinct_codes_per_window_mean": float(distinct.mean()),
        "distinct_codes_per_window_median": float(np.median(distinct)),
        "distinct_codes_per_window_p10": float(np.percentile(distinct, 10)),
        "distinct_codes_per_window_p90": float(np.percentile(distinct, 90)),
        "effective_codes_per_window_median": float(np.median(ppl)),
    }


def run_d1(toks: dict[str, torch.Tensor], margins: dict[str, np.ndarray],
           K: int, C: int, F_: int, W: int, cb_max_delta: float = 0.0) -> dict:
    from scipy.optimize import linear_sum_assignment

    clients = list(toks)
    hists, usage = {}, {}
    for c in clients:
        t = toks[c].reshape(-1).numpy()
        h = np.bincount(t, minlength=K).astype(np.float64)
        h /= h.sum()
        hists[c] = h
        nz = h[h > 0]
        usage[c] = {
            "active_codes_corpus": int((h > 0).sum()),
            "perplexity_corpus": float(np.exp(-(nz * np.log(nz)).sum())),
            "top1_share": float(h.max()),
            **_per_window_stats(toks[c], K),
        }

    def js(p, q):
        m = 0.5 * (p + q)
        def kl(a, b):
            nz = a > 0
            return float((a[nz] * (np.log(a[nz]) - np.log(np.maximum(b[nz], 1e-12)))).sum())
        return 0.5 * kl(p, m) + 0.5 * kl(q, m)

    active = {c: set(np.flatnonzero(hists[c] > 0).tolist()) for c in clients}

    pairs = {}
    for a, b in itertools.combinations(clients, 2):
        ta, tb = toks[a].reshape(-1).numpy(), toks[b].reshape(-1).numpy()
        n = ta.size
        agree = float((ta == tb).mean())
        p_chance = float((hists[a] * hists[b]).sum())
        kappa = (agree - p_chance) / (1.0 - p_chance) if p_chance < 1.0 else float("nan")

        # Migliore biiezione sui K codici. FITTATA SU METÀ E VALUTATA SULL'ALTRA:
        # l'assegnamento ottimo ha fino a K parametri liberi, quindi in-sample è
        # una quantità fittata, non una misura. Il numero onesto è l'held-out.
        half = n // 2
        M_fit = np.zeros((K, K), dtype=np.int64)
        np.add.at(M_fit, (ta[:half], tb[:half]), 1)
        r, cc = linear_sum_assignment(-M_fit)
        sigma = np.arange(K)
        sigma[cc] = r                                     # b-code → a-code
        agree_hung_fit = float(M_fit[r, cc].sum() / max(half, 1))
        agree_hung_held = float((ta[half:] == sigma[tb[half:]]).mean()) if n - half else float("nan")

        # Sovrapposizione dei supporti: se i client colonizzano sottoinsiemi
        # DISGIUNTI del dizionario condiviso, ogni riga di token_embedding del
        # prior è addestrata da un solo client e mediarle è strutturalmente rotto.
        inter = len(active[a] & active[b])
        union = len(active[a] | active[b])

        # Accordo per riga di frequenza — il disaccordo è concentrato in una banda?
        ga = toks[a].reshape(-1, C, F_, W).numpy()
        gb = toks[b].reshape(-1, C, F_, W).numpy()
        by_freq = [float((ga[:, :, f] == gb[:, :, f]).mean()) for f in range(F_)]

        pairs[f"{a}|{b}"] = {
            "agree_raw": agree, "p_chance": p_chance, "kappa": kappa,
            "agree_hungarian_fit": agree_hung_fit,
            "agree_hungarian_heldout": agree_hung_held,
            "support_jaccard": inter / max(union, 1),
            "support_intersection": inter,
            "js_marginal": js(hists[a], hists[b]),
            "agree_by_freq": by_freq,
        }

    k_mean = float(np.mean([v["kappa"] for v in pairs.values()]))
    h_mean = float(np.mean([v["agree_hungarian_heldout"] for v in pairs.values()]))
    raw_mean = float(np.mean([v["agree_raw"] for v in pairs.values()]))
    jac_mean = float(np.mean([v["support_jaccard"] for v in pairs.values()]))
    ties = np.concatenate([margins[c].reshape(-1) for c in clients]) if margins else np.array([])
    tie_frac = float((ties < 0.01).mean()) if ties.size else 0.0

    hung_gain = h_mean - raw_mean          # entrambi GREZZI: confronto omogeneo
    act = [u["active_codes_corpus"] for u in usage.values()]
    mech = ("; il meccanismo è la COLONIZZAZIONE di sottoinsiemi quasi disgiunti del "
            "dizionario condiviso (Jaccard %.3f, codici attivi %s su K=%d), quindi ogni riga "
            "di prior.token_embedding è addestrata da un solo client" % (jac_mean, act, K)
            ) if jac_mean < 0.25 else ""
    pw = [u["distinct_codes_per_window_median"] for u in usage.values()]
    if cb_max_delta > 1e-9:
        # `local`: ogni client k-means-inizializza il PROPRIO codebook, quindi l'indice j
        # è un vettore diverso su ogni client. Accordo grezzo, kappa e Jaccard confrontano
        # etichette che non denotano la stessa cosa: sono numeri, non misure.
        return {
            "codebook_max_delta": cb_max_delta, "cross_client_interpretable": False,
            "codebook_usage": usage, "pairs": pairs,
            "median_distinct_codes_per_window": float(np.median(pw)),
            "verdict": ("CODEBOOK NON CONDIVISI (max|Δ| = %.2e fra i client). Le statistiche "
                        "CROSS-client (accordo grezzo %.3f, kappa %.3f, Jaccard %.3f) NON sono "
                        "interpretabili qui: l'indice j denota un vettore diverso su ogni "
                        "client. Restano valide solo le statistiche INTRA-client: codici "
                        "distinti per finestra (mediana %s su L=%d token, K=%d) e codici "
                        "attivi sul corpus %s."
                        % (cb_max_delta, raw_mean, k_mean, jac_mean,
                           [round(x, 1) for x in pw],
                           usage[clients[0]]["tokens_per_window"], K, act)),
        }
    if k_mean > 0.6:
        verdict = ("ENCODER CONCORDANTI (kappa=%.3f, accordo grezzo %.3f). La premessa "
                   "'latenti diversi' è falsa in questo cluster: il gauge fixing non serve."
                   % (k_mean, raw_mean))
    elif hung_gain > 0.20:
        verdict = ("PERMUTAZIONE FUNZIONALE (accordo grezzo %.3f → %.3f dopo la migliore "
                   "rietichettatura held-out, +%.3f; kappa %.3f). Gli encoder inducono in buona "
                   "parte la STESSA partizione con etichette diverse: una rietichettatura dei "
                   "token esiste ed è un intervento economico%s."
                   % (raw_mean, h_mean, hung_gain, k_mean, mech))
    else:
        verdict = ("SHIFT DI COVARIATA (accordo grezzo %.3f → %.3f dopo rietichettatura, solo "
                   "+%.3f; kappa %.3f). Nessuna rietichettatura recupera l'accordo: gli encoder "
                   "partizionano l'input in modo genuinamente diverso, serve gauge fixing%s."
                   % (raw_mean, h_mean, hung_gain, k_mean, mech))
    if tie_frac > 0.20:
        verdict += (" ⚠ %.0f%% delle assegnazioni sono quasi-pareggi (margine <0.01): una "
                    "quota del disaccordo è numerica, non semantica." % (100 * tie_frac))

    return {
        "codebook_max_delta": cb_max_delta, "cross_client_interpretable": True,
        "codebook_usage": usage,
        "median_distinct_codes_per_window": float(np.median(pw)),
        "pairs": pairs,
        "mean_kappa": k_mean,
        "mean_agree_raw": raw_mean,
        "mean_agree_hungarian_heldout": h_mean,
        "mean_support_jaccard": jac_mean,
        "mean_hungarian_gain": hung_gain,
        "tie_margin": {
            "median": float(np.median(ties)) if ties.size else None,
            "frac_below_0.01": tie_frac if ties.size else None,
            "frac_below_0.05": float((ties < 0.05).mean()) if ties.size else None,
        },
        "verdict": verdict,
    }


# ═════════════════════════════════════════════════════════════════════════════
#   D2 — il disaccordo sposta il punteggio?
# ═════════════════════════════════════════════════════════════════════════════

def _swap_bn(dst: Stage2System, src: Stage2System) -> dict:
    """Copia SOLO i buffer BatchNorm dell'encoder di `src` dentro `dst`.
    I pesi restano di `dst`. Isola il contributo delle running stats, che questo
    repo tiene locali anche negli arm che federano l'encoder."""
    sd_src = src.stage1.state_dict()
    sd_dst = dst.stage1.state_dict()
    keys = [k for k in sd_dst
            if k.startswith("encoder.") and k.endswith(("running_mean", "running_var"))]
    saved = {k: sd_dst[k].clone() for k in keys}
    with torch.no_grad():
        for k in keys:
            sd_dst[k].copy_(sd_src[k].to(sd_dst[k].device))
    return saved


def _restore(dst: Stage2System, saved: dict) -> None:
    sd = dst.stage1.state_dict()
    with torch.no_grad():
        for k, v in saved.items():
            sd[k].copy_(v)


def run_d2(models: dict[str, Stage2System], x_test: torch.Tensor, starts: list[int],
           series_len: int, T_win: int, labels: np.ndarray | None,
           device: torch.device, tol: int, do_bn: bool) -> dict:
    from scipy.stats import spearmanr

    clients = list(models)
    own_tok = {c: tokenize(models[c], x_test, device) for c in clients}
    own_field = {c: score_field(models[c], own_tok[c], device) for c in clients}
    own_ts = {c: assemble(own_field[c], starts, series_len, T_win) for c in clients}

    def hit(ts: np.ndarray) -> bool | None:
        if labels is None or labels.sum() == 0:
            return None
        am = int(np.argmax(ts))
        idx = np.flatnonzero(labels)
        return bool(idx.min() - tol <= am <= idx.max() + tol)

    pairs, flips = {}, 0
    for a, b in itertools.permutations(clients, 2):
        f_for = score_field(models[a], own_tok[b], device)               # prior_A, token di B
        ts_for = assemble(f_for, starts, series_len, T_win)
        rho_field = float(spearmanr(own_field[a].reshape(-1).numpy(),
                                    f_for.reshape(-1).numpy()).statistic)
        rho_ts = float(spearmanr(own_ts[a], ts_for).statistic)
        h_own, h_for = hit(own_ts[a]), hit(ts_for)
        flipped = (h_own is not None and h_for is not None and h_own != h_for)
        flips += int(flipped)
        pairs[f"prior={a}|tok={b}"] = {
            "spearman_field": rho_field,
            "spearman_timestep": rho_ts,
            "argmax_shift_samples": int(abs(int(np.argmax(own_ts[a])) - int(np.argmax(ts_for)))),
            "top1_hit_own": h_own, "top1_hit_foreign": h_for, "top1_flipped": flipped,
        }

    rho = float(np.mean([v["spearman_timestep"] for v in pairs.values()]))

    bn = None
    if do_bn:
        bn = {}
        for a, b in itertools.permutations(clients, 2):
            saved = _swap_bn(models[a], models[b])
            t_bn = tokenize(models[a], x_test, device)
            _restore(models[a], saved)
            same = float((t_bn == own_tok[a]).float().mean())
            full = float((own_tok[b] == own_tok[a]).float().mean())
            bn[f"{a}<-BN({b})"] = {
                "agree_with_own": same,
                "agree_full_foreign_encoder": full,
                "bn_share_of_disagreement": (
                    float((1 - same) / (1 - full)) if full < 1.0 else None
                ),
            }

    if rho > 0.98 and flips == 0:
        verdict = ("IRRILEVANTE (rho=%.4f, 0 ribaltamenti su %d). Il punteggio è insensibile "
                   "a QUALE encoder ha tokenizzato: l'allineamento del tokenizer non serve, "
                   "la Fase 1 si cancella." % (rho, len(pairs)))
    elif rho > 0.90:
        verdict = ("MARGINALE (rho=%.4f, %d ribaltamenti su %d). Il campo è quasi invariante "
                   "ma la decisione top-1 può muoversi." % (rho, flips, len(pairs)))
    else:
        verdict = ("È IL CRUX (rho=%.4f, %d ribaltamenti su %d). Il tokenizer straniero "
                   "cambia il punteggio: il gauge fixing è la precondizione di tutto."
                   % (rho, flips, len(pairs)))

    return {"pairs": pairs, "mean_spearman_timestep": rho, "n_top1_flips": flips,
            "bn_swap": bn, "verdict": verdict}


# ═════════════════════════════════════════════════════════════════════════════
#   D3 — barriera di interpolazione lineare
# ═════════════════════════════════════════════════════════════════════════════

def _fixed_masks(tokens: torch.Tensor, sched, seed: int, draws: int
                 ) -> list[torch.Tensor]:
    """Maschere FISSE (stesso seed per ogni α), altrimenti la curva di
    interpolazione misurerebbe il rumore della maschera invece della barriera."""
    g = torch.Generator().manual_seed(seed)
    N, L = tokens.shape
    out = []
    for _ in range(draws):
        m = torch.zeros(N, L, dtype=torch.bool)
        for r in range(N):
            k = _masked_count(L, sched(float(torch.rand(1, generator=g))))
            perm = torch.randperm(L, generator=g)
            m[r, perm[:k]] = True
        out.append(m)
    return out


@torch.no_grad()
def masked_nll(prior, tokens: torch.Tensor, masks: list[torch.Tensor],
               device: torch.device, batch: int = 128) -> float:
    tot, n = 0.0, 0
    for m in masks:
        for i in range(0, len(tokens), batch):
            t = tokens[i:i + batch].to(device)
            mk = m[i:i + batch].to(device)
            inp = t.clone()
            inp[mk] = prior.mask_token_id
            logits = prior._logits(inp)
            ce = Fnn.cross_entropy(logits[mk], t[mk], reduction="sum")
            tot += float(ce)
            n += int(mk.sum())
    return tot / max(n, 1)


def run_d3(models: dict[str, Stage2System], x_val: torch.Tensor, device: torch.device,
           arm_name: str, n_alpha: int, seed: int) -> dict:
    clients = list(models)
    ref = models[clients[0]]
    shared = [k for k in ref.state_dict()
              if k.startswith("prior.") and not k.startswith(LOCAL_PRIOR_PREFIXES)]

    # Il body condiviso è identico? Se sì (cb_only), la LMC su questo arm è vacua.
    max_delta = 0.0
    sd0 = ref.state_dict()
    for c in clients[1:]:
        sdc = models[c].state_dict()
        for k in shared:
            max_delta = max(max_delta, float((sd0[k].cpu() - sdc[k].cpu()).abs().max()))
    if max_delta < 1e-9:
        return {
            "arm": arm_name, "shared_body_max_delta": max_delta, "vacuous": True,
            "verdict": (
                "VACUA su questo arm: il body condiviso è IDENTICO su tutti i client "
                "(max|Δ| = %.2e su %d tensori) — è l'invariante che federated_stage2 "
                "asserisce dopo ogni broadcast. Interpolare due copie dello stesso punto "
                "non misura niente: l'invariante di broadcast REGGE, e la LMC va letta "
                "sugli arm a prior locale." % (max_delta, len(shared))
            ),
        }

    alphas = [i / (n_alpha - 1) for i in range(n_alpha)]
    results = {}
    for a, b in itertools.combinations(clients, 2):
        tok_a = tokenize(models[a], x_val, device)                       # token di A, sempre
        # La maschera è FISSA su tutta la curva: il prior la campiona a ogni forward,
        # e senza fissarla la curva misurerebbe il rumore della maschera, non la barriera.
        masks = _fixed_masks(tok_a, models[a].prior.mask_scheduling_fn, seed, draws=2)
        sd_a = {k: v.detach().clone() for k, v in models[a].state_dict().items()}
        sd_b = {k: v.detach().clone() for k, v in models[b].state_dict().items()}

        for mode, keys in (("body_only", shared),
                           ("full", [k for k in sd_a if k.startswith("prior.")])):
            curve = []
            for al in alphas:
                mix = {k: sd_a[k] for k in sd_a}
                for k in keys:
                    mix[k] = (1 - al) * sd_a[k].float() + al * sd_b[k].float()
                models[a].load_state_dict({k: v.to(sd_a[k].dtype) for k, v in mix.items()},
                                          strict=True)
                curve.append(masked_nll(models[a].prior, tok_a, masks, device))
            models[a].load_state_dict(sd_a, strict=True)
            endpoint_gap = curve[-1] - curve[0]
            excess = max(curve) - max(curve[0], curve[-1])
            results[f"{a}|{b}|{mode}"] = {
                "alphas": alphas, "nll": curve,
                "nll_own": curve[0],                   # L(θ_A) sui propri token
                "endpoint_gap": endpoint_gap,          # gap LINGUISTICO (θ_B sui token di A)
                "endpoint_gap_rel": endpoint_gap / max(curve[0], 1e-9),
                "excess_barrier": excess,              # non-convessità PURA
                "excess_rel": excess / max(curve[0], 1e-9),
                # Normalizzazione LMC standard: eccesso sopra il PEGGIORE degli estremi.
                # È questa che dice se la media cade FUORI dal segmento; dividere per la NLL
                # propria gonfia il numero ogni volta che il gap fra gli estremi è grande.
                "barrier_over_worse_endpoint": excess / max(curve[0], curve[-1], 1e-9),
                "nll_ratio_foreign": curve[-1] / max(curve[0], 1e-9),
                "nll_at_half": curve[len(curve) // 2],
            }

    body = [v for k, v in results.items() if k.endswith("body_only")]
    exc = float(np.mean([v["excess_rel"] for v in body]))
    bar = float(np.mean([v["barrier_over_worse_endpoint"] for v in body]))
    gap = float(np.mean([v["endpoint_gap"] for v in body]))
    ratio = float(np.mean([v["nll_ratio_foreign"] for v in body]))
    own = float(np.mean([v["nll_own"] for v in body]))
    head = ("barriera %.0f%% sopra il peggiore degli estremi · prior altrui x%.1f · "
            "NLL propria %.3f (gap assoluto %+.2f nat)" % (100 * bar, ratio, own, gap))
    if bar >= 0.15 and ratio >= 2.0:
        verdict = ("BARRIERA E GAP LINGUISTICO (%s). Entrambi i termini sono reali, ma il "
                   "dominante e' il secondo: il prior dell'altro client e' gia' x%.1f peggiore "
                   "PRIMA di qualunque media. Ridurre tau attacca solo la barriera e lascia "
                   "intatto il grosso." % (head, ratio))
    elif bar >= 0.15:
        verdict = ("BARRIERA REALE (%s). I client finiscono in bacini scollegati mentre i loro "
                   "token restano compatibili: serve tau piccolo (FedSGD), non FedAvg a epoche."
                   % head)
    elif ratio >= 2.0:
        verdict = ("COSTO TUTTO LINGUISTICO (%s). La curva non esce dal segmento fra gli "
                   "estremi: la geometria non oppone barriera. Il prezzo e' che il prior "
                   "dell'altro client e' inutilizzabile sui token di questo. Federare i PESI "
                   "non e' il problema, allineare i TOKEN lo e'." % head)
    else:
        verdict = ("PRIOR INTERCAMBIABILI (%s). Ne' barriera ne' gap: i prior sono gia' quasi "
                   "lo stesso oggetto e mediarli e' innocuo." % head)
    return {"arm": arm_name, "shared_body_max_delta": max_delta, "vacuous": False,
            "pairs": results, "mean_excess_rel": exc, "mean_endpoint_gap": gap,
            "mean_barrier_over_worse_endpoint": bar, "mean_nll_ratio_foreign": ratio,
            "mean_nll_own": own,
            "verdict": verdict}


# ═════════════════════════════════════════════════════════════════════════════
#   D4 — mismatch fra maschera di training e maschera di scoring
# ═════════════════════════════════════════════════════════════════════════════

def run_d4(models: dict[str, Stage2System], x_val: torch.Tensor, device: torch.device,
           seed: int) -> dict:
    per_client = {}
    for c, s2 in models.items():
        prior = s2.prior
        tok = tokenize(s2, x_val, device)
        masks = _fixed_masks(tok, prior.mask_scheduling_fn, seed, draws=4)
        nll_random = masked_nll(prior, tok, masks, device)

        # La maschera di SCORING: per ogni colonna w si maschera il blocco
        # [w-half, w+half] e si media -log p sul blocco. È esattamente il tensore
        # che score_tokens_per_rate restituisce → la sua media È la NLL sotto la
        # maschera di scoring, per costruzione. Media pesata sul numero di finestre
        # del chunk, così l'ultimo chunk corto non conta come uno pieno.
        s2._inform_latent_shape()
        acc, seen = None, 0
        with torch.no_grad():
            for i in range(0, len(tok), 64):
                chunk = tok[i:i + 64]
                r = prior.score_tokens_per_rate(chunk.to(device)).float().cpu()
                m = r.mean(dim=(2, 3, 4)).sum(dim=1)                     # somma sulle finestre
                acc = m if acc is None else acc + m
                seen += len(chunk)
        per_rate = (acc / max(seen, 1)).tolist()
        W = s2._W
        ks = [_paper_kernel_size(W, r) for r in prior.score_window_size_rates]
        per_client[c] = {
            "nll_random_trainmask": nll_random,
            "nll_column_scoremask_per_rate": per_rate,
            "rates": list(prior.score_window_size_rates),
            "kernel_sizes": ks,
            "masked_frac_score": [min(k, W) / W for k in ks],
            "ratio_worst_rate": max(per_rate) / max(nll_random, 1e-9),
        }
    ratios = [v["ratio_worst_rate"] for v in per_client.values()]
    r = float(np.mean(ratios))
    if r > 1.5:
        verdict = ("MISMATCH FORTE (×%.2f). Il prior è valutato molto fuori dal regime in cui "
                   "è addestrato: l'ablazione mask_mode=column va fatta PRIMA di qualunque "
                   "lavoro sulla federazione — sposterebbe ogni cella." % r)
    elif r > 1.15:
        verdict = ("MISMATCH MODERATO (×%.2f). Vale un'ablazione mask_mode, ma non blocca "
                   "il programma di federazione." % r)
    else:
        verdict = ("MISMATCH TRASCURABILE (×%.2f). La maschera di training è un buon proxy "
                   "di quella di scoring: mask_mode non è una priorità." % r)
    return {"per_client": per_client, "mean_ratio": r, "verdict": verdict}


# ═════════════════════════════════════════════════════════════════════════════
#   Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="artifacts/runs/zn_main",
                    help="cartella della run. Con --list accetta una LISTA separata da virgole "
                         "(o 'all' = tutte le artifacts/runs/*), per vedere in un colpo solo "
                         "zn_main, zn_enc, zn_norev, zn_a1, zn_a2… Per sondare serve UNA sola.")
    ap.add_argument("--dataset", default="ucr_split_w2p")
    ap.add_argument("--cluster", default=None, help="uno; oppure --all-complete")
    ap.add_argument("--all-complete", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="federated_cb_only", help="alias di --arms con un solo arm")
    ap.add_argument("--arms", default=None,
                    help="lista di arm da sondare, separati da virgole. Ogni arm produce un JSON. "
                         "La probe viene costruita UNA volta per cluster e riusata.")
    ap.add_argument("--lmc-arms", default="local,federated",
                    help="arm AGGIUNTIVI su cui girare D3, oltre a --arm (lista separata da virgole)")
    ap.add_argument("--probe-split", default="val", choices=["val", "test"])
    ap.add_argument("--probe-windows", type=int, default=384, help="0 = tutte")
    ap.add_argument("--d2-stride", type=int, default=5, help="sottocampionamento finestre di test")
    ap.add_argument("--d2-windows", type=int, default=0, help="tetto duro (0 = nessuno)")
    ap.add_argument("--tolerance", type=int, default=64, help="tolleranza top-1 in campioni")
    ap.add_argument("--alphas", type=int, default=9, help="punti di interpolazione in D3")
    ap.add_argument("--mask-seed", type=int, default=1234)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default="d0,d1,d2,d3,d4")
    ap.add_argument("--out-dir", default="evidence")
    ap.add_argument("--list", action="store_true", help="mostra cosa è completo ed esci")
    args = ap.parse_args()

    # ── Più tag, non solo zn_main ───────────────────────────────────────────────────────
    # La campagna non vive più sotto un tag solo: `zn_enc` (trio encoder), `zn_norev`,
    # `zn_a1`, `zn_a2`… E i tag NON sono spazi di nomi disgiunti: `federated_enc_fedavg`
    # esiste identico in `zn_enc` e in `zn_a1`, dove differisce SOLO per `--fed-enc-bn`.
    # Se il nome del report non portasse il tag, la seconda sonda sovrascriverebbe la prima
    # e distruggerebbe esattamente il contrasto appaiato che le due celle esistono per fare.
    if args.run_dir.strip() == "all":
        roots = sorted(p for p in resolve_path("artifacts/runs").iterdir() if p.is_dir())
    else:
        roots = [resolve_path(s.strip()) for s in args.run_dir.split(",") if s.strip()]
    raw_root = resolve_path(Config().paths.raw_data)

    if args.list or (not args.cluster and not args.all_complete):
        for rd in roots:
            av = discover(rd, args.dataset, args.seed)
            if not av:
                continue
            print(f"\n[{rd.name}] {rd} / {args.dataset} / seed{args.seed}\n")
            for cl, arms in av.items():
                exp = expected_clients(args.dataset, cl, raw_root)
                n_exp = len(exp) or 5
                done = [f"{a}({len(cs)}/{n_exp})" for a, cs in arms.items()]
                ready = [a for a, cs in arms.items() if len(cs) == n_exp]
                print(f"  {cl:12s} {'  '.join(done)}"
                      + (f"   → PRONTI: {','.join(ready)}" if ready else "   → nessuno completo"))
        print()
        return 0

    if len(roots) != 1:
        print(f"per sondare serve UNA sola --run-dir (ne ho {len(roots)}); "
              f"la lista è ammessa solo con --list.")
        return 2
    run_dir = roots[0]
    TAG = run_dir.name
    avail = discover(run_dir, args.dataset, args.seed)

    want = ("d0", "d1", "d2", "d3", "d4")
    only = {s.strip() for s in args.only.split(",") if s.strip() in want}
    clusters = ([c for c, arms in avail.items()
                 if args.arm in arms and len(arms[args.arm]) == (len(expected_clients(args.dataset, c, raw_root)) or 5)]
                if args.all_complete else [args.cluster])

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    # fp32 puro: una diagnostica non deve ereditare il rumore dell'AMP della run.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    rc = 0
    arm_list = ([a.strip() for a in args.arms.split(",") if a.strip()]
                if args.arms else [args.arm])
    for cluster in clusters:
      probe_cache: dict = {}
      for ARM in arm_list:
          t0 = time.time()
          print("\n" + "═" * 78)
          print(f"  {cluster}  ·  arm={ARM}  ·  seed{args.seed}  ·  device={device}")
          print("═" * 78)

          arms = avail.get(cluster, {})
          exp = expected_clients(args.dataset, cluster, raw_root)
          n_exp = len(exp) or 5
          if ARM not in arms or len(arms[ARM]) != n_exp:
              got = len(arms.get(ARM, []))
              print(f"  ✗ SALTATO: {ARM} ha {got}/{n_exp} client. "
                    f"Una cella in volo non si sonda.")
              rc = 1
              continue
          clients = arms[ARM]
          arm_dir = run_dir / "ckpt" / args.dataset / cluster / f"seed{args.seed}" / ARM

          # ── config dal checkpoint, e assert che i client concordino ──────────
          cfg, cfg_raw = config_from_ckpt(arm_dir / clients[0] / "stage2.ckpt")
          for c in clients[1:]:
              _, other = config_from_ckpt(arm_dir / c / "stage2.ckpt")
              a = {k: v for k, v in cfg_raw["dataset"].items() if k != "entity_id"}
              b = {k: v for k, v in other["dataset"].items() if k != "entity_id"}
              if a != b or cfg_raw["quantizer"] != other["quantizer"] or cfg_raw["prior"] != other["prior"]:
                  print(f"  ✗ ABORT: config divergente fra {clients[0]} e {c}.")
                  return 2
          print(f"  window={cfg.dataset.window_length}  znorm={cfg.dataset.window_normalization}  "
                f"K={cfg.quantizer.codebook_size}  prior={cfg.prior.name}")

          report: dict = {
              "cluster": cluster, "arm": ARM, "seed": args.seed, "tag": TAG,
              "dataset": args.dataset, "run_dir": str(run_dir), "clients": clients,
              "config": {"window_length": cfg.dataset.window_length,
                         "window_normalization": cfg.dataset.window_normalization,
                         "codebook_size": cfg.quantizer.codebook_size,
                         "prior": cfg.prior.name, "scaling": cfg.dataset.scaling},
          }

          # ── probe condivisa, costruita PER CLIENT (D0 verifica che coincidano) ─
          pkey = (cfg.dataset.window_length, cfg.dataset.window_normalization,
                  cfg.dataset.scaling, args.probe_split, args.probe_windows)
          if pkey in probe_cache:
              probes = probe_cache[pkey]
              print(f"  [probe] riuso ({len(probes[clients[0]])} finestre)")
          else:
              print("  [probe] costruzione finestre…", flush=True)
              probes = {}
              for c in clients:
                  xs, _, _ = build_probe(cfg, c, args.probe_split, 1, args.probe_windows)
                  probes[c] = xs
              probe_cache[pkey] = probes
          L_win = probes[clients[0]].shape[-1]
          print(f"  [probe] {len(probes[clients[0]])} finestre × {L_win} campioni "
                f"({args.probe_split})")

          if "d0" in only:
              report["D0"] = run_d0(probes)
              print(f"  [D0] {report['D0']['verdict']}")

          x_probe = probes[clients[0]]

          # ── modelli ─────────────────────────────────────────────────────────
          print("  [load] modelli…", flush=True)
          models = {c: load_model(arm_dir, c, cfg, x_probe[:1], device) for c in clients}
          m0 = models[clients[0]]
          C_, F_, W_ = m0._C, m0._F, m0._W
          print(f"  [load] latente C={C_} F={F_} W={W_} → L={C_ * F_ * W_} token, "
                f"K={cfg.quantizer.codebook_size}")

          cb_keys = [k for k in m0.state_dict()
                     if k.startswith("stage1.") and k.endswith("codebook.weight")]
          cb_ref = models[clients[0]].state_dict()
          cb_delta = max([float((cb_ref[k].cpu() - models[c].state_dict()[k].cpu()).abs().max())
                          for k in cb_keys for c in clients[1:]] or [0.0])
          report["codebook_max_delta_across_clients"] = cb_delta
          print(f"  [cb] max|Δ| del codebook fra client = {cb_delta:.2e}"
                + ("  → indici NON confrontabili fra client" if cb_delta > 1e-9 else "  → indici confrontabili"))

          if "d1" in only:
              print("  [D1] tokenizzazione + accordo…", flush=True)
              toks = {c: tokenize(models[c], x_probe, device) for c in clients}
              marg = {c: tie_margins(models[c], x_probe, device) for c in clients}
              report["D1"] = run_d1(toks, marg, cfg.quantizer.codebook_size, C_, F_, W_, cb_delta)
              d1 = report["D1"]
              act = [u["active_codes_corpus"] for u in d1["codebook_usage"].values()]
              pw = [round(u["distinct_codes_per_window_median"], 1)
                    for u in d1["codebook_usage"].values()]
              print(f"       codici/finestra (mediana)={pw} su L="
                    f"{d1['codebook_usage'][clients[0]]['tokens_per_window']} token, "
                    f"attivi sul corpus={act}/{cfg.quantizer.codebook_size}")
              if not d1.get("cross_client_interpretable", True):
                  print(f"  [D1] {d1['verdict']}")
              else:
                  print(f"       kappa={d1['mean_kappa']:.3f}  raw={d1['mean_agree_raw']:.4f}  "
                    f"hungarian(held-out)={d1['mean_agree_hungarian_heldout']:.3f}  "
                      f"jaccard={d1['mean_support_jaccard']:.3f}  "
                        f"tie<0.01={d1['tie_margin']['frac_below_0.01']:.3f}")
                  print(f"  [D1] {d1['verdict']}")

          if "d4" in only:
              print("  [D4] mismatch di maschera…", flush=True)
              report["D4"] = run_d4(models, x_probe, device, args.mask_seed)
              print(f"  [D4] {report['D4']['verdict']}")

          if "d2" in only:
              if args.d2_stride >= cfg.dataset.window_length:
                  print(f"  ✗ ABORT D2: --d2-stride {args.d2_stride} >= window "
                        f"{cfg.dataset.window_length}. Le finestre non si sovrappongono, la serie "
                        f"riassemblata ha BUCHI di copertura e argmax/top-1 non significano niente. "
                        f"Usa uno stride < {cfg.dataset.window_length}.")
                  return 2
              print("  [D2] scoring con tokenizer straniero…", flush=True)
              x_test, starts_t, series = build_probe(cfg, clients[0], "test",
                                                     args.d2_stride, args.d2_windows)
              lab_p = raw_root / args.dataset / "test_label" / f"{clients[0]}.npy"
              labels = np.load(lab_p) if lab_p.exists() else None
              print(f"       {len(x_test)} finestre di test (stride {args.d2_stride}), "
                    f"serie {len(series)} campioni", flush=True)
              report["D2"] = run_d2(models, x_test, starts_t, len(series), L_win,
                                    labels, device, args.tolerance, do_bn=True)
              d2 = report["D2"]
              print(f"       rho(timestep)={d2['mean_spearman_timestep']:.4f}  "
                    f"ribaltamenti top-1={d2['n_top1_flips']}/{len(d2['pairs'])}")
              print(f"  [D2] {d2['verdict']}")

          if "d3" in only:
              d3_arms = [ARM] + [a.strip() for a in args.lmc_arms.split(",")
                                      if a.strip() and a.strip() != ARM]
              report["D3"] = {}
              for a_name in d3_arms:
                  if a_name not in arms or len(arms[a_name]) != n_exp:
                      got = len(arms.get(a_name, []))
                      report["D3"][a_name] = {"skipped": True,
                                              "reason": f"{got}/{n_exp} client in {cluster}"}
                      print(f"  [D3] {a_name}: {got}/{n_exp} client — saltato.")
                      continue
                  print(f"  [D3] interpolazione lineare su arm={a_name}…", flush=True)
                  if a_name == ARM:
                      mm = models
                  else:
                      a_dir = run_dir / "ckpt" / args.dataset / cluster / f"seed{args.seed}" / a_name
                      mm = {c: load_model(a_dir, c, cfg, x_probe[:1], device) for c in arms[a_name]}
                  report["D3"][a_name] = run_d3(mm, x_probe, device, a_name,
                                                args.alphas, args.mask_seed)
                  print(f"  [D3:{a_name}] {report['D3'][a_name]['verdict']}")
                  if mm is not models:
                      del mm
                      if device.type == "cuda":
                          torch.cuda.empty_cache()

          report["elapsed_s"] = round(time.time() - t0, 1)
          report["D_computed"] = sorted(only)
          # Il tag entra nel nome SOLO se non è zn_main: così i 57 report già su disco
          # restano validi e ogni script che li aggrega con il glob storico continua a
          # pescare zn_main e soltanto zn_main, senza mescolare tag per sbaglio.
          stem = f"latent_probe_{args.dataset}_{cluster}_{ARM}_seed{args.seed}.json"
          out = out_dir / (stem if TAG == "zn_main" else f"latent_probe_{TAG}__{stem[len('latent_probe_'):]}")
          # MERGE, non sovrascrittura. Il path non dipende da --only, quindi una passata
          # parziale (`--only d0,d1`) cancellerebbe i blocchi D2/D3/D4 di una passata
          # completa precedente — successo silenzioso che distrugge dati. È già successo.
          if out.exists():
              try:
                  prev = json.load(out.open())
                  kept = [k for k in prev if k.startswith("D") and k not in report]
                  if kept:
                      print(f"  [merge] conservo dal report precedente: {', '.join(sorted(kept))}")
                  report = {**prev, **report}
              except (OSError, json.JSONDecodeError) as e:
                  print(f"  [merge] report precedente illeggibile ({e}); sovrascrivo.")
          with out.open("w") as fh:
              json.dump(report, fh, indent=2, default=float)
          print(f"\n  → {out}   ({report['elapsed_s']:.0f}s)")

          print("\n  ── VERDETTI ──")
          for k in ("D0", "D1", "D2", "D4"):
              if k in report and isinstance(report[k], dict) and "verdict" in report[k]:
                  print(f"   {k}: {report[k]['verdict']}")
          for a_name, v in (report.get("D3") or {}).items():
              if isinstance(v, dict) and "verdict" in v:
                  print(f"   D3[{a_name}]: {v['verdict']}")

          del models
          if device.type == "cuda":
              torch.cuda.empty_cache()

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
