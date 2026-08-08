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
    # — il criterio di ARRESTO, che non era nella lista delle otto —
    #
    # I tetti di step sono GIA' quelli di upstream: `stage1_max_steps=10_000`,
    # `stage2_max_steps=50_000` (`config.py:254-255`). A fermarci prima sono DUE cose, e
    # servono entrambe per riprodurre il loro protocollo:
    #
    #  1. la pazienza (`_converged_loop`, `federated_eval.py:392`). Su `ucr_001` lo stadio 1
    #     si ferma a ~8 700 e lo stadio 2 a ~14 000: ci alleniamo circa un terzo di loro.
    #     Si spegne con `early_stopping=False`.
    #
    #  2. ⚠️ il ripristino dei **pesi migliori su val** (`federated_eval.py:401`,
    #     `load_state_dict(best_state)`), che upstream NON fa: Lightning salva lo stato
    #     finale. Spegnere solo la pazienza sarebbe un NO-OP — si allenerebbe fino a 50 000
    #     step per poi ricaricare i pesi del passo 14 000, cioe' lo stesso modello di adesso.
    #
    # Non c'e' un flag per (2), ma c'e' una leva pulita: `early_stopping_min_delta` negativo
    # e grande rende `va < best_val - min_delta` sempre vero, quindi `best_state` viene
    # ri-catturato a OGNI validazione e alla fine contiene l'ULTIMO stato, non il migliore.
    # Nessuna patch al codice.
    #
    # ⚠️ Residuo noto: il ciclo esce a meta' epoca quando `step >= max_steps`, e dopo
    # quell'uscita non gira una validazione. Quindi lo stato conservato e' quello della fine
    # dell'ultima epoca completa — al piu' 271 batch prima del tetto, meno del 3%.
    "nostop":     {"training.early_stopping": (False, True),
                   "training.early_stopping_min_delta": (-1e9, 1e-4)},

    # ─── `fidelity` — tutte le differenze insieme, l'ultima spiaggia ─────────
    #
    # Le varianti qui sopra muovono UNA differenza per volta, e tre su tre dicono che
    # la nostra scelta e' migliore della loro. Eppure `centralized` sta 0,284 sotto gli
    # autori su `ucr_001` e 0,417 sotto il loro codice girato da noi su `ucr_043`. Le due
    # cose insieme non stanno in piedi: se ogni singolo asse ci vede avanti, la causa e'
    # in un asse mai misurato, in un'interazione, o in un errore di porting.
    #
    # Questa variante porta il modello a upstream su TUTTI gli assi contemporaneamente.
    # Non attribuisce niente — risponde a una domanda sola, ma quella che conta: **la
    # fedelta' totale chiude il divario, si' o no?** Se no, la causa non e' nel modello e
    # nemmeno nel protocollo, e l'unica cosa rimasta e' lo split federato stesso.
    #
    # ⚠️ Resta `centralized`, cioe' il federated POOLED. Non e' un fork del loro codice:
    # e' il nostro arm, con le loro scelte. La scala della federazione regge.
    "fidelity": {
        # — stack del prior (i due assi MAI misurati, per mancanza di implementazione) —
        "prior.name":          ("maskgit_upstream", "maskgit_3d_pos"),  # x-transformers + pos-emb 1-D piatta
        "prior.use_rmsnorm":   (True,  False),
        "prior.post_emb_norm": (True,  False),
        "prior.embed_dim":     (64,    128),
        "prior.dropout":       (0.3,   0.2),
        # — dinamica del codebook (tutte e tre le differenze) —
        "quantizer.ema_decay":               (0.8,  0.99),
        "quantizer.kmeans_init":             (False, True),
        "quantizer.threshold_ema_dead_code": (0,    2),
        # — encoder —
        # ⚠️ `width_base=4` E' la configurazione di upstream, verificata sul loro checkpoint
        # `saved_models/stage1-1.ckpt` il 2026-08-03 leggendo le larghezze conv reali:
        #
        #   upstream            EncBlock 2->4, 4->8, 8->16, 16->32, poi ResBlock 32->64
        #   noi width_base=4    Downsample 2->4, 4->8, 8->16, 16->32, poi Project 32->64   IDENTICO
        #   noi width_base=16   Downsample 2->16, 16->32, 32->64, 64->128, poi Project 128->64
        #
        # Il commento a `config.py:130-137` afferma che 16 "matcha upstream, che arriva a
        # dim=64": e' SBAGLIATO, confonde la larghezza massima del CORPO con quella della
        # proiezione finale. Upstream arriva a 64 solo con l'ultimo ResBlock; il corpo si ferma
        # a 32. A width_base=16 il nostro corpo e' 4x piu' largo del loro a ogni stadio.
        # Entrambi hanno profondita' 4 su W=408 (verificato: H'=3, W'=25 da entrambe le parti).
        "encoder.width_base":  (4,     16),
        "encoder.dropout":     (0.3,   0.2),
        # — numerica —
        "training.amp":        (False, True),                            # fp32
        # — protocollo: i tre punti di `nostop`, piu' il val —
        "training.early_stopping":    (False, True),
        "training.keep_last_weights": (True,  False),
        "dataset.pool_val_into_train": (True, False),
    },

    # ─── `znorm` — la differenza che NON era nella lista delle otto ──────────
    #
    # Trovata il 2026-08-03 leggendo il loro codice dopo che l'ablazione era chiusa 7 su 7
    # senza recuperi. Upstream normalizza **ogni singola finestra** (z-score per istanza),
    # sia in training (`preprocessing/preprocess.py:149`, dentro `__getitem__`) sia in
    # detect (`evaluation/__init__.py:117`). Noi usiamo lo scaling globale per-entita' e
    # `window_normalization="none"`.
    #
    # Non e' una svista loro: il docstring di `scale()` lo motiva —
    #   "instance-wise scaling. global-scaling is not used because there are time series
    #    with local-mean-shifts such as UCR_Anomaly_sddb49_20000_67950_68200.txt"
    #
    # La loro `scale()` e' riga per riga il nostro `"zscore"` (media, std NON distorta,
    # clamp a 1e-4), quindi il knob esiste gia': era solo spento.
    #
    # PROVA INDIPENDENTE: il loro modello addestrato, fatto girare nel NOSTRO detect senza
    # normalizzazione, crolla a AUPRC 0,301 (top-1 a 0 su 4 client su 5) contro lo 0,887
    # che ottiene nella loro pipeline. Un modello addestrato su finestre normalizzate a cui
    # si danno finestre non normalizzate si degrada esattamente cosi'.
    #
    # ⚠️ Con la z-score per finestra lo scaler per-client diventa IRRILEVANTE (la
    # normalizzazione e' invariante a qualunque affine precedente). Se questa variante
    # recupera, l'intera classe di artefatti da scaler — [[detect-test-score-depends-on-
    # train-shard]] e il fix dello scaler federato — sparisce per costruzione.
    "znorm": {"dataset.window_normalization": ("zscore", "none")},

    # ─── le tre ablazioni che vale la pena RIFARE sotto z-norm ───────────────
    # Le undici varianti del mondo "none" sono inservibili per scegliere la configurazione:
    # sono state prese sotto una rappresentazione mal condizionata, e il 2026-08-03 abbiamo
    # visto un'interazione enorme (`nostop` da sola -0,357, ma dentro `fidelity` il top-1
    # torna a 1,00). Solo queste tre hanno un meccanismo per cui la z-norm cambia la risposta:
    #
    #   nostop_znorm    era la peggiore (-0,357). Ma sotto z-norm il riferimento si allena
    #                   gia' a 34 114 step di stadio 2 invece di ~14 000: "allenarsi fino in
    #                   fondo" non e' piu' la stessa cosa.
    #   fidprior_znorm  era l'UNICA positiva (+0,050, al limite del rumore +-0,043).
    #   width4_znorm    senza livello e ampiezza da modellare, un encoder 4x piu' stretto
    #                   potrebbe bastare — e costa un quarto ad allenare.
    #
    # Le altre (ema08, dropout03, fp32, embed64, batch_up) erano dentro il rumore o senza
    # meccanismo: non si rifanno.
    "nostop_znorm": {"dataset.window_normalization": ("zscore", "none"),
                     "training.early_stopping": (False, True),
                     "training.keep_last_weights": (True, False)},
    "fidprior_znorm": {"dataset.window_normalization": ("zscore", "none"),
                       "prior.name":          ("maskgit_upstream", "maskgit_3d_pos"),
                       "prior.use_rmsnorm":   (True,  False),
                       "prior.post_emb_norm": (True,  False),
                       "prior.embed_dim":     (64,    128)},
    "width4_znorm": {"dataset.window_normalization": ("zscore", "none"),
                     "encoder.width_base": (4, 16)},

    # `fidelity` + la normalizzazione per finestra: la ricetta pubblicata COMPLETA.
    # `znorm` da sola dice se il pezzo mancante basta partendo dai NOSTRI default;
    # questa dice dove arriva il metodo di upstream preso per intero.
    "fidelity_znorm": {
        "prior.name":          ("maskgit_upstream", "maskgit_3d_pos"),
        "prior.use_rmsnorm":   (True,  False),
        "prior.post_emb_norm": (True,  False),
        "prior.embed_dim":     (64,    128),
        "prior.dropout":       (0.3,   0.2),
        "quantizer.ema_decay":               (0.8,  0.99),
        "quantizer.kmeans_init":             (False, True),
        "quantizer.threshold_ema_dead_code": (0,    2),
        "encoder.width_base":  (4,     16),
        "encoder.dropout":     (0.3,   0.2),
        "training.amp":        (False, True),
        "training.early_stopping":    (False, True),
        "training.keep_last_weights": (True,  False),
        "dataset.pool_val_into_train": (True, False),
        "dataset.window_normalization": ("zscore", "none"),
    },

    # Isola il SOLO asse mai misurato: lo stack del prior. Se `fidelity` recupera e
    # questa no, il merito e' altrove; se recuperano entrambe, e' qui. Tiene `--batch 64`
    # e ogni altro default nostro, quindi e' appaiata al riferimento della campagna.
    "fidelity_prior": {
        "prior.name":          ("maskgit_upstream", "maskgit_3d_pos"),
        "prior.use_rmsnorm":   (True,  False),
        "prior.post_emb_norm": (True,  False),
        "prior.embed_dim":     (64,    128),
    },
}

# Varianti che devono NON ricevere `--batch` sulla riga di comando, altrimenti il flag
# vincerebbe sull'override (gli argomenti CLI sono letti dopo `apply_env_overrides`).
SENZA_FLAG_BATCH = {"batch_up", "fidelity", "fidelity_znorm"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTI))
    ap.add_argument("--arm", default="centralized")
    ap.add_argument("--tol", type=int, default=64)
    ap.add_argument("--out-suffix", default="",
                    help="Appended to the run directory name. Use it for TVQ_SMOKE=1 dry runs "
                         "so their checkpoints can never be picked up by the real cell's resume.")
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

    out = REPO / f"artifacts/runs/ucr{a.series}_abl_{a.variant}{a.out_suffix}"
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
