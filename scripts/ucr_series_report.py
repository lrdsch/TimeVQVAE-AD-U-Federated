#!/usr/bin/env python3.10
"""Raccoglie lo stato e i numeri di UNA serie UCR (floor + deep) in un solo markdown.

Rieseguibile: rilegge sempre il disco, non tiene stato. Va lanciato mentre il run gira
per avere il parziale, e alla fine per avere la tabella definitiva.

    $PY scripts/ucr_series_report.py ucr_001 > documentation/UCR001_RESULTS.md
    $PY scripts/ucr_series_report.py ucr_011 --cohort ucr011 > documentation/UCR011_RESULTS.md

Di default i nomi si derivano dal cluster: `ucr_011` -> coorte `ucr011`, tag deep
`ucr011_v1`, tag floor `ucr011_floor`. `--cohort/--tag/--floor-tag` li sovrascrivono.

Perche' esiste. I due engine scrivono in formati diversi -- il deep un JSON per
(dataset, cluster, arm) con un blocco `summary`, il floor un JSONL con una riga per
(arm, entity) -- e nessuno dei due sa dell'altro. Il confronto floor-vs-deep, che e' la
ragione per cui il floor esiste, va fatto qui.

⚠️ Le tabelle qui girano su VUS-PR, che su UNA serie e' legittimo ma fra serie UCR non e'
aggregabile. Il deep IL top-1 lo calcola eccome -- sta in `<arm>/<client>/report.json` --
ma `federated_eval.METRIC_KEYS` non lo promuove al summary, che e' quello che questo
report legge. Per la metrica ufficiale UCR, aggregabile fra serie, usare
`scripts/topk_table.py`, che pesca dai report.json per client.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ARMS = ["local", "centralized", "federated_cb_only_ema", "federated_cb_only",
        "federated_fedavg_cb_only", "federated_shared",
        "federated_fedavg_cb_sharedprior", "federated"]
METRICS = ["vus_pr", "auprc", "auroc", "pate_f1"]

# popolati da main() -- il modulo e' uno script, non una libreria
CLUSTER = COHORT = TAG = FLOOR_TAG = ""
DEEP = FLOOR = LOGS_DEEP = ORCH = Path()
BUILDS: list[tuple[str, int]] = []


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


# ── deep ────────────────────────────────────────────────────────────────────
def deep_rows() -> dict[tuple[str, str], dict]:
    out = {}
    for ds, _ in BUILDS:
        for arm in ARMS:
            p = DEEP / ds / f"{CLUSTER}__{arm}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            s = d.get("summary", {}).get(arm, {})
            out[(ds, arm)] = {m: s.get(m, {}).get("mean") for m in METRICS} | {
                "n": s.get("vus_pr", {}).get("n"),
                "std": s.get("vus_pr", {}).get("std"),
            }
    return out


def deep_progress() -> dict[tuple[str, str], tuple[str, int | None]]:
    """(stato, ultimo round) per ogni job, letto dai log vivi."""
    out = {}
    orch = ORCH.read_text() if ORCH.exists() else ""
    for ds, _ in BUILDS:
        for arm in ARMS:
            name = f"{ds}__{CLUSTER}__{arm}"
            done = re.search(rf"DONE  {re.escape(name)} \(slot [^)]*rc=(\d+)\)", orch)
            log = LOGS_DEEP / f"{name}.log"
            rnd = None
            if log.exists():
                hits = re.findall(r"^\[fed:\w+\] round (\d+)", log.read_text(), re.M)
                if hits:
                    rnd = int(hits[-1])
            if done:
                out[(ds, arm)] = ("OK" if done.group(1) == "0" else f"FAIL rc={done.group(1)}", rnd)
            elif (DEEP / ds / f"{CLUSTER}__{arm}.json").exists():
                out[(ds, arm)] = ("OK", rnd)
            elif log.exists():
                out[(ds, arm)] = ("in corso", rnd)
            else:
                out[(ds, arm)] = ("in coda", None)
    return out


# ── floor ───────────────────────────────────────────────────────────────────
def floor_rows() -> dict[str, dict[str, dict]]:
    """{dataset: {arm: {metrica: media sui client}}}"""
    out: dict[str, dict[str, list]] = {}
    for p in sorted(FLOOR.glob("records_*.jsonl")):
        ds = p.stem.replace("records_", "")
        by_arm: dict[str, list] = {}
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            by_arm.setdefault(r["_arm"], []).append(r)
        out[ds] = by_arm
    agg = {}
    for ds, by_arm in out.items():
        agg[ds] = {}
        for arm, rows in by_arm.items():
            tolkeys = {k for r in rows for k in r if k.startswith("paper_top1_acc_at_")}
            d = {}
            for m in ["vus_pr", "auprc", "auroc"]:
                vals = [r[m] for r in rows if isinstance(r.get(m), (int, float))]
                d[m] = statistics.fmean(vals) if vals else None
            t1 = [r[k] for r in rows for k in tolkeys if isinstance(r.get(k), (int, float))]
            d["top1"] = statistics.fmean(t1) if t1 else None
            d["n"] = len(rows)
            d["W"] = rows[0].get("_window")
            agg[ds][arm] = d
    return agg


def fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def _hist(ds: str, arm: str):
    """La dir di checkpoint di un arm encoder porta il suffisso delle sue manopole
    (`federated_enc_fedproto_lam1_uniform`, `federated_enc_fedprox_mu0.01`), quindi il
    nome nudo non basta: senza il glob la colonna convergenza mentiva con un '—'."""
    base = DEEP / "ckpt" / ds / f"{CLUSTER}/seed0"
    p = base / arm / "fed_history.json"
    if p.exists():
        return p
    cand = sorted(base.glob(f"{arm}*/fed_history.json"))
    return cand[0] if len(cand) == 1 else None


def convergence(ds: str, arm: str) -> str:
    """Regola dura 2026-07-24: un run il cui round migliore e' l'ULTIMO non e' convergito
    e non e' riportabile. Il verdetto va letto dal disco, non dallo stdout del run."""
    p = _hist(ds, arm)
    if p is None:
        return "—"
    h = json.loads(p.read_text()).get("stage1") or []
    if not isinstance(h, list) or not h:
        return "—"
    sel = [r["round"] for r in h if r.get("selected")]
    last = h[-1]["round"]
    best = sel[0] if sel else None
    if best is None:
        return f"? / {last}"
    if best == last:
        return f"🔴 **TRONCATO** (best={best}=ultimo)"
    return f"r{best} / {last}"


# ── report ──────────────────────────────────────────────────────────────────
def main() -> int:
    rows, prog, fl = deep_rows(), deep_progress(), floor_rows()
    now = sh("date '+%Y-%m-%d %H:%M'")
    fp = json.loads((REPO / f"cohorts/{COHORT}.json").read_text())["fingerprint"]
    run = json.loads((DEEP / "RUN.json").read_text()) if (DEEP / "RUN.json").exists() else {}

    P = print
    P(f"# `{CLUSTER}` — floor + tutti gli arm deep\n")
    P(f"Generato da `scripts/ucr_series_report.py {CLUSTER}` il **{now}**. "
      f"Rieseguibile: rilegge il disco.\n")
    P(f"| | |\n|---|---|")
    P(f"| coorte | `{COHORT}`, fingerprint **`{fp}`** |")
    P(f"| finestre | " + " · ".join(f"`{d}` W={w}" for d, w in BUILDS) + " |")
    P(f"| tag deep | `{TAG}` — {len([1 for v in prog.values() if v[0]=='OK'])}/"
      f"{len(ARMS) * len(BUILDS)} completi |")
    P(f"| tag floor | `{FLOOR_TAG}` |")
    P(f"| protocollo | `{run.get('protocol','?')}`, s1_rounds={run.get('s1_rounds','?')}, "
      f"s2_rounds={run.get('s2_rounds','?')}, patience={run.get('fed_patience_rounds','?')} |")
    P(f"| client | 5 per build, quantity-skew `[10,10,20,20,30] %` |")
    # il commit sta nel `meta` di ogni job, non in RUN.json: prendi il primo che c'e'
    commit, dirty = "?", None
    for p in sorted(DEEP.glob(f"*/{CLUSTER}__*.json")):
        m = json.loads(p.read_text()).get("meta", {})
        if m.get("commit"):
            commit, dirty = m["commit"][:12], m.get("dirty")
            break
    P(f"| commit | `{commit}`{' ⚠️ **worktree dirty**' if dirty else ''} |\n")

    P("⚠️ **Le tabelle qui girano su VUS-PR**, legittimo su **una** serie ma "
      "[non aggregabile fra serie UCR](UCR_WINDOW_ABLATION_SET.md). Il top-1 il deep lo "
      "calcola — sta in `<arm>/<client>/report.json` — solo che `METRIC_KEYS` non lo "
      "promuove al summary che questo report legge. Per la metrica ufficiale UCR: "
      "`scripts/topk_table.py`.\n")

    for ds, w in BUILDS:
        P(f"\n## Deep — `{ds}` (W={w})\n")
        P("| arm | stato | round | best/ultimo | VUS-PR | AUPRC | AUROC | PATE-F1 | sd fra client |")
        P("|---|---|---:|---|---:|---:|---:|---:|---:|")
        for arm in ARMS:
            st, rnd = prog.get((ds, arm), ("?", None))
            r = rows.get((ds, arm), {})
            P(f"| `{arm}` | {st} | {rnd if rnd is not None else '—'} | {convergence(ds, arm)} | "
              f"{fmt(r.get('vus_pr'))} | {fmt(r.get('auprc'))} | {fmt(r.get('auroc'))} | "
              f"{fmt(r.get('pate_f1'))} | {fmt(r.get('std'))} |")

    for ds in sorted(fl):
        P(f"\n## Floor — `{ds}`\n")
        P("| arm | n righe | W | VUS-PR | AUPRC | AUROC | paper top-1 |")
        P("|---|---:|---:|---:|---:|---:|---:|")
        for arm in sorted(fl[ds], key=lambda a: -(fl[ds][a]["vus_pr"] or 0)):
            d = fl[ds][arm]
            P(f"| `{arm}` | {d['n']} | {d['W']} | {fmt(d['vus_pr'])} | {fmt(d['auprc'])} | "
              f"{fmt(d['auroc'])} | {fmt(d['top1'], 2)} |")

    # ── ricostruzione vs rilevamento: NON sono la stessa cosa ───────────────
    P("\n## Ricostruzione contro rilevamento — il protocollo di stop guarda la cosa sbagliata\n")
    P("| build | arm | merge | val_loss s1 al best | perplexity | VUS-PR |")
    P("|---|---|---|---:|---:|---:|")
    diag = []
    for ds, _ in BUILDS:
        for arm in ARMS:
            p = _hist(ds, arm)
            if p is None:
                continue
            h = json.loads(p.read_text())
            s = h.get("stage1") or []
            if not s:
                continue
            sel = [r for r in s if r.get("selected")]
            b = sel[0] if sel else min(s, key=lambda r: r["val_loss"])
            v = rows.get((ds, arm), {}).get("vus_pr")
            diag.append((ds, arm, h.get("cb_mode"), b["val_loss"], b["perplexity"], v))
            P(f"| `{ds}` | `{arm}` | {h.get('cb_mode')} | {b['val_loss']:.4f} | "
              f"{b['perplexity']:.0f}/64 | {fmt(v)} |")
    # Calcolato, non scritto a mano: la correlazione fra le due colonne su QUESTA serie.
    ok = [(v, d) for _, _, _, v, _, d in diag if d is not None]
    if len(ok) >= 3:
        try:
            rho = statistics.correlation([v for v, _ in ok], [d for _, d in ok])
            verso = ("**anti-correlate**" if rho < -0.3 else
                     "correlate" if rho > 0.3 else "**scorrelate**")
            P(f"\nSu questa serie val_loss e VUS-PR sono {verso} (Pearson **{rho:+.2f}** su "
              f"{len(ok)} arm). Attesa ingenua: correlazione negativa forte — ricostruire "
              f"meglio dovrebbe rilevare meglio.\n")
        except statistics.StatisticsError:
            pass
    P("🔴 **Precedente registrato (misurato su `ucr_001`, 2026-07-31).** Lì la ricostruzione "
      "**anti-prediceva** il rilevamento: `federated_fedavg_cb_sharedprior` a W=408 aveva la "
      "val_loss di stage 1 migliore del suo build (0,2150) e il detector peggiore di 17× "
      "(0,027).\n")
    P("❌ **Ipotesi scartata lì, da non riciclare qui: non è la perplexity.** Sembrava che un "
      "dizionario stretto rilevasse meglio (suff-stat 4–9 codeword contro FedAvg 43). "
      "`federated_fedavg_cb_only` la refuta: perplexity **57** e VUS-PR **0,418** sano, mentre "
      "l'arm collassato usava *meno* codeword (43). La perplexity non ordina i risultati.\n")
    P("⚠️ **Conseguenza sul protocollo.** `--protocol converged` ferma il training sulla "
      "val_loss di RICOSTRUZIONE, ma la metrica riportata è il RILEVAMENTO. Un arm può essere "
      "«convergito» a pieno titolo e inutile come detector, e la tabella non lo distingue da "
      "un arm fermato male. Questo vale per ogni riga, non solo per quelle FedAvg.\n")

    # ── il fattoriale 2x2: primitiva del codebook x condivisione del prior ──
    CELLS = {("suffstat", "local"): "federated_cb_only",
             ("suffstat", "shared"): "federated_shared",
             ("fedavg", "local"): "federated_fedavg_cb_only",
             ("fedavg", "shared"): "federated_fedavg_cb_sharedprior"}
    P("\n## Il fattoriale 2×2 — primitiva del codebook × condivisione del prior\n")
    for ds, w in BUILDS:
        got = {k: rows.get((ds, a), {}).get("vus_pr") for k, a in CELLS.items()}
        if not any(v is not None for v in got.values()):
            continue
        P(f"**`{ds}` (W={w})** — VUS-PR, media sui 5 client\n")
        P("| merge del codebook | prior LOCALE | prior CONDIVISO | effetto del prior |")
        P("|---|---:|---:|---:|")
        for m in ("suffstat", "fedavg"):
            loc, shr = got[(m, "local")], got[(m, "shared")]
            d = f"**{shr - loc:+.3f}**" if (loc is not None and shr is not None) else "—"
            name = "**suff-stat** (Prop. 1)" if m == "suffstat" else "FedAvg (media pesata)"
            P(f"| {name} | {fmt(loc)} | {fmt(shr)} | {d} |")
        a, b = got[("suffstat", "local")], got[("fedavg", "local")]
        c, e = got[("suffstat", "shared")], got[("fedavg", "shared")]
        P(f"| **effetto del merge** | "
          f"{f'**{a - b:+.3f}**' if (a is not None and b is not None) else '—'} | "
          f"{f'**{c - e:+.3f}**' if (c is not None and e is not None) else '—'} | |")
        P("")
    P("Riferimenti nella stessa colonna: `local` e `centralized` in cima alla sezione del "
      "build. Un arm federato che sta sotto `local` non sta pagando la federazione — sta "
      "peggiorando rispetto a non federare affatto.\n")
    P("⚠️ Leggere insieme al controllo dei gemelli qui sotto: dove la riga FedAvg ha una "
      "divergenza di stage 1 grande, la sua cella è **un sorteggio**, e l'interazione fra i "
      "due assi non è interpretabile a un seed solo.\n")

    # ── i gemelli che DEVONO avere lo stesso stage 1 ────────────────────────
    # `local_prefixes` entra solo in `_prior_shared_keys` (federated.py:1954), che tocca
    # `clients[0].s2.prior` -- lo STAGE 2. Le coppie qui sotto differiscono SOLO per quel
    # parametro, quindi il loro stage 1 e' la stessa identica configurazione con lo stesso
    # seed. Se le traiettorie divergono, la divergenza e' rumore di esecuzione -- e ogni
    # contrasto a un solo seed che passa per quello stage 1 ne eredita la varianza.
    TWINS = [("federated_cb_only", "federated_shared", "suff-stat"),
             ("federated_fedavg_cb_only", "federated_fedavg_cb_sharedprior", "FedAvg")]
    P("\n## Controllo di riproducibilità — gemelli a stage 1 identico\n")
    P("| build | coppia | merge | val_loss di stage 1 all'ultimo round comune | divergenza |")
    P("|---|---|---|---|---:|")
    any_twin = False
    for ds, _ in BUILDS:
        for x, y, merge in TWINS:
            hx = DEEP / "ckpt" / ds / f"{CLUSTER}/seed0" / x / "fed_history.json"
            hy = DEEP / "ckpt" / ds / f"{CLUSTER}/seed0" / y / "fed_history.json"
            if not (hx.exists() and hy.exists()):
                continue
            a = json.loads(hx.read_text()).get("stage1") or []
            b = json.loads(hy.read_text()).get("stage1") or []
            if not (a and b):
                continue
            any_twin = True
            i = min(len(a), len(b)) - 1
            va, vb = a[i]["val_loss"], b[i]["val_loss"]
            ratio = max(va, vb) / max(min(va, vb), 1e-12)
            flag = "🔴" if ratio > 2 else ("⚠️" if ratio > 1.3 else "ok")
            P(f"| `{ds}` | `{x}` / `{y}` | {merge} | r{i}: {va:.4f} / {vb:.4f} | {flag} **{ratio:.1f}×** |")
    if not any_twin:
        P("| — | nessuna coppia ancora completa | | | |")
    P("\nLe due colonne di ogni coppia **dovrebbero coincidere**: stesso seed, stessa "
      "configurazione di stage 1, e la seeding per round/client è esplicita "
      "(`_round_seed(seed, round, client, stage)`). Non coincidono perché "
      "`cfg.deterministic = False` (config.py:281), `torch.backends.cudnn.benchmark = True` "
      "(federated.py:1290) e l'AMP gira in **float16**: la selezione dei kernel e le "
      "riduzioni non deterministiche fanno divergere le traiettorie da differenze alla "
      "quinta cifra.\n")
    P("🔴 **Conseguenza da non aggirare.** Dove la divergenza è grande, un contrasto a UN "
      "SOLO SEED fra due arm non misura l'effetto dell'arm: misura il sorteggio. Peggio, "
      "l'early stopping lo cristallizza — un arm che pianeggia per 6 round viene fermato e "
      "il suo plateau diventa «il risultato». Prima di mettere in tabella una differenza fra "
      "arm servono **più seed**, oppure `AMP=0` + `deterministic=True` per togliere la "
      "sorgente di rumore.\n")

    # ── l'ablazione sulla finestra: il confronto APPAIATO, arm per arm ──────
    wlo, whi = BUILDS[0][1], BUILDS[-1][1]
    P(f"\n## Ablazione sulla finestra — W={wlo} contro W=2P ({whi}), rapporto "
      f"{whi / wlo:.2f}×\n")
    P("Stessa serie, stessi 5 client, stesso protocollo: **unica variabile W**. "
      "Solo gli arm chiusi su ENTRAMBI i build compaiono qui — un arm a metà "
      "non è un confronto appaiato.\n")
    P("| arm | VUS-PR | AUPRC | AUROC |")
    P("|---|---:|---:|---:|")
    n_pair = 0
    for arm in ARMS:
        a, b = rows.get(("ucr_split", arm)), rows.get(("ucr_split_w2p", arm))
        if not a or not b or a.get("vus_pr") is None or b.get("vus_pr") is None:
            continue
        n_pair += 1
        cells = []
        for m in ["vus_pr", "auprc", "auroc"]:
            d = b[m] - a[m]
            cells.append(f"{a[m]:.3f} → {b[m]:.3f} (**{d:+.3f}**)")
        P(f"| `{arm}` | " + " | ".join(cells) + " |")
    if n_pair == 0:
        P("| — | nessun arm ancora chiuso su entrambi i build | | |")
    P(f"\n{n_pair}/{len(ARMS)} arm appaiati. Un delta positivo dice che agganciare la "
      "finestra al periodo (`T = 2 × periodo`, come il paper) batte la nostra W=128 fissa.\n")
    P("⚠️ Due avvertenze sul segno. (i) Le metriche **a soglia** non seguono: su "
      "`centralized` affiliation-F1 fa 0,827 → 0,671 e F1 0,589 → 0,459. La finestra larga "
      "**ordina** meglio, ma la soglia al quantile 0,99 le va peggio — è una storia sul "
      "ranking, non su un detector calibrato. (ii) La frazione di rete federata cambia con W "
      "(encoder 44,9 % a W=128, 40,1 % a W=512), quindi parte del delta sugli arm federati è "
      "cambio di perimetro, non di finestra: vedi `UCR_WINDOW_ABLATION_SET.md` §5.1.\n")

    # ── il confronto che e' il punto ────────────────────────────────────────
    P("\n## Deep contro floor\n")
    P("| build | miglior floor (VUS-PR) | miglior deep (VUS-PR) | rapporto |")
    P("|---|---:|---:|---:|")
    for ds, _ in BUILDS:
        fb = max((d["vus_pr"] or 0 for d in fl.get(ds, {}).values()), default=None)
        db = max((r["vus_pr"] for (d2, a), r in rows.items()
                  if d2 == ds and r.get("vus_pr") is not None), default=None)
        ratio = f"{db/fb:.0f}x" if (fb and db) else "—"
        P(f"| `{ds}` | {fmt(fb)} | {fmt(db)} | {ratio} |")
    P("\nUn rapporto ≫ 1 dice che il deep stacca il floor su questa serie. È il **contrario** "
      "di quanto misurato su `wsd_fed`, dove `movavg10` a zero parametri sta a 0,507 e batte "
      "il trio `enc_*`. Una serie non è una tendenza: va confermato sulle altre 9 prima di "
      "scriverlo.\n")
    return 0


def _setup() -> None:
    """Risolve i nomi dagli argomenti e legge le finestre DALLA COORTE.

    Le finestre non si scrivono a mano: `ucr_split` e' 128 fisso, `ucr_split_w2p` e'
    per-serie (2 x periodo). Prenderle dal file della coorte e' l'unico modo perche' il
    titolo della tabella non possa mentire sulla finestra a cui i numeri sono stati presi.
    """
    global CLUSTER, COHORT, TAG, FLOOR_TAG, DEEP, FLOOR, LOGS_DEEP, ORCH, BUILDS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cluster", help="es. ucr_001")
    ap.add_argument("--cohort", default=None, help="default: cluster senza underscore")
    ap.add_argument("--tag", default=None, help="default: <coorte>_v1")
    ap.add_argument("--floor-tag", default=None, help="default: <coorte>_floor")
    a = ap.parse_args()

    CLUSTER = a.cluster
    COHORT = a.cohort or CLUSTER.replace("_", "")
    TAG = a.tag or f"{COHORT}_v1"
    FLOOR_TAG = a.floor_tag or f"{COHORT}_floor"
    DEEP = REPO / "artifacts/runs" / TAG
    FLOOR = REPO / "artifacts/runs" / FLOOR_TAG / "floor"
    LOGS_DEEP = REPO / "logs/runs" / TAG
    ORCH = LOGS_DEEP / "_orchestrator.log"

    cp = REPO / f"cohorts/{COHORT}.json"
    if not cp.exists():
        sys.exit(f"nessuna coorte {cp} — creala con scripts/cohort.py new {COHORT} ...")
    c = json.loads(cp.read_text())
    for ds, v in c["datasets"].items():
        w = v["window"] if v["window_mode"] == "fixed" else v["windows"].get(CLUSTER)
        if w is None:
            sys.exit(f"la coorte {COHORT} non ha una finestra per {CLUSTER} in {ds}")
        BUILDS.append((ds, int(w)))
    BUILDS.sort(key=lambda t: t[1])


if __name__ == "__main__":
    _setup()
    sys.exit(main())
