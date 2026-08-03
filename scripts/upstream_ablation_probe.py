#!/usr/bin/env python3
"""Ablazione delle differenze con upstream, UNA alla volta, sulla stessa serie.

DOMANDA. Sulla metrica DEGLI AUTORI (top-1/3/5) li pareggiamo: dopo il fix dello scaler
top-3 e top-5 fanno 0,778 contro 0,778, serie per serie. Ma su AUPRC siamo indietro in 7 casi
su 9 (Delta medio -0,219). Verificato il 2026-08-03: la pipeline di scoring e' la stessa da
entrambe le parti (tre scale 0,1/0,3/0,5, stride 0,1, somma sulle scale, media sulle bande,
impulso (a+MA)/2, NLL puro del prior), quindi **non e' un artefatto di aggregazione**: il nostro
profilo di punteggio e' davvero piu' sporco. Troviamo lo stesso picco, ma con piu' massa
spuria attorno.

Delle 8 differenze verificate con upstream ([[local-vs-upstream-verified]]) solo tre hanno
un meccanismo plausibile per produrre un profilo piu' sporco, e vanno tutte nella stessa
direzione — **noi regolarizziamo meno**:

    width4      corpo conv encoder: noi width_base=16, loro d=4 (siamo 4x piu' larghi).
                Ipotesi: piu' capacita' su serie di poche migliaia di punti = ricostruiamo
                bene anche il rumore = picchi spuri. E' l'unica delle tre in cui "essere
                piu' grandi" sarebbe un danno, ed e' la piu' interessante.
    dropout03   prior dropout: noi 0,2, loro 0,3.
    ema08       VQ ema_decay: noi 0,99, loro 0,8. A 0,99 il codebook si muove pochissimo,
                quindi la ricostruzione resta meno adattata ai dati.

Le altre cinque (fp16/fp32, batch, embed_dim, RMSNorm, pos-emb fattorizzata) o non hanno un
meccanismo chiaro per l'AUPRC o richiedono codice nuovo (RMSNorm non ha una variante).

RIFERIMENTO. Non serve rilanciarlo: e' la cella della campagna `ucr001_v1` W=2P `centralized`
(top-1 1,00 · AUROC 0,955 · AUPRC 0,646 · VUS-PR 0,618). Gli autori su `ucr_001` fanno
0,997 / 0,930 / 0,929, quindi il divario da spiegare e' **-0,284 di AUPRC**.

PERCHE' `ucr_001`. Il rapporto fra le sd dei 5 shard di train e' 1,04: l'artefatto dello shard
non c'e', quindi non serve lo scaler federato e la cella della campagna e' un riferimento
pulito. Ed e' una serie dove entrambi prendiamo top-1 = 1, quindi il divario AUPRC non e'
confuso con un fallimento di detection.

COME. Le tre varianti passano tutte per lo stesso meccanismo — un override sul Config dopo
`apply_env_overrides` — anche quelle che avrebbero un flag CLI (`--width-base`,
`--prior-dropout`). Mescolare flag e patch renderebbe le tre corse non confrontabili fra loro.
`quantizer.ema_decay` un flag non ce l'ha comunque.

Il resto della riga di comando e' copiato da `artifacts/runs/ucr001_v1/RUN.json` (converged,
300 round, 10 epoche locali, pazienza 6, batch 64, seed 0), cosi' l'UNICA differenza rispetto
al riferimento e' la variante.

    $PY scripts/upstream_ablation_probe.py --series 001 --window 408 --variant width4

⚠️ Scrive in `artifacts/runs/ucr<serie>_abl_<variante>/`, tag separato: non tocca le 306 celle.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

REPO = Path("/home/leonardo/PhD/TimeVQVAE-AD-U-Federated")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))

import config as cfgmod  # noqa: E402

# variante -> {percorso nel Config: (valore upstream, valore nostro)}
# Il valore nostro serve solo a controllare che il riferimento sia davvero la cella della
# campagna: se non combacia, la sonda lo dice invece di produrre un confronto silenziosamente
# sbagliato.
VARIANTI: dict[str, dict[str, tuple[object, object]]] = {
    # — le tre con un meccanismo per l'AUPRC (lanciate su g2 alle 16:56) —
    "width4":    {"encoder.width_base":  (4,   16)},
    "dropout03": {"prior.dropout":       (0.3, 0.2)},
    "ema08":     {"quantizer.ema_decay": (0.8, 0.99)},
    # — le restanti differenze verificate con upstream —
    "prior_flat": {"prior.name": ("maskgit", "maskgit_3d_pos")},   # pos-emb 1-D piatta
    "embed64":    {"prior.embed_dim": (64, 128)},
    "fp32":       {"training.amp": (False, True)},
    # ⚠️ `batch_up` non tocca il Config: i default sono GIA' i valori upstream
    # (`dataset.batch_size_stage1=256`, `stage2=128`, `config.py:89-90`). La campagna gira a
    # 64/64 soltanto perche' `launch.sh:34` passa `--batch 64`, che sovrascrive entrambi gli
    # stadi. Quindi la variante consiste nell'**omettere il flag**, non nel settare un campo.
    "batch_up":   {},
}

# Varianti che devono NON ricevere `--batch` sulla riga di comando, altrimenti il flag
# vincerebbe sull'override (gli argomenti CLI sono letti dopo `apply_env_overrides`).
SENZA_FLAG_BATCH = {"batch_up"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTI))
    ap.add_argument("--arm", default="centralized")
    ap.add_argument("--tol", type=int, default=64)
    a = ap.parse_args()
    s = f"ucr_{a.series}"

    override = VARIANTI[a.variant]
    vero_env = cfgmod.apply_env_overrides

    def env_piu_variante(cfg):
        """Applica gli override d'ambiente come sempre, poi la variante. Gli argomenti CLI
        sono letti DOPO (`federated_eval.py:1227`), quindi un flag esplicito vincerebbe
        comunque su questo: vedi `SENZA_FLAG_BATCH`."""
        vero_env(cfg)
        for path, (val_up, val_nostro) in override.items():
            sezione, campo = path.split(".")
            obj = getattr(cfg, sezione)
            attuale = getattr(obj, campo)
            if attuale != val_nostro:
                print(f"[abl] ⚠️ {path} valeva {attuale!r}, mi aspettavo {val_nostro!r} "
                      f"— il riferimento potrebbe non essere la cella della campagna")
            setattr(obj, campo, val_up)
            print(f"[abl] {path}: {attuale!r} -> {val_up!r} (valore upstream)")
        return cfg

    cfgmod.apply_env_overrides = env_piu_variante

    out = REPO / f"artifacts/runs/ucr{a.series}_abl_{a.variant}"
    sys.argv = [
        "federated_eval.py",
        "--dataset", "ucr_split_w2p", "--cluster", s, "--arms", a.arm,
        "--protocol", "converged", "--s1-rounds", "300", "--s2-rounds", "300",
        "--local-epochs", "10", "--fed-patience-rounds", "6",
        "--seeds", "0",
        "--window-length", str(a.window), "--metrics-tolerance", str(a.tol),
        "--out-dir", str(out / "ckpt/ucr_split_w2p"),
        "--out-json", str(out / f"{a.arm}.json"),
    ]
    if a.variant in SENZA_FLAG_BATCH:
        print("[abl] `--batch` OMESSO: valgono i default del Config, "
              "che sono i valori upstream 256 (stage 1) / 128 (stage 2)")
    else:
        sys.argv += ["--batch", "64"]      # come la campagna (`launch.sh:34`)
    (out / "ckpt/ucr_split_w2p").mkdir(parents=True, exist_ok=True)
    runpy.run_path(str(REPO / "pipeline/federated_eval.py"), run_name="__main__")


if __name__ == "__main__":
    main()
