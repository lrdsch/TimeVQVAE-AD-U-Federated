# Pre-registrazione — la regola di aggregazione dell'encoder serve davvero al prior condiviso?

**Aperta 2026-08-13, prima di qualunque numero.** Criteri di decisione fissati qui e non
modificabili dopo la prima lettura.

## 1. La domanda, e perché il paper oggi non la chiude

Il contributo 1 dice: *condividere il prior giova solo dopo che il tokenizer è allineato*.
L'allineamento, in tutte le configurazioni di punta, è prodotto da **FedAvg sull'encoder**.

Il blocco (k)–(n) di Tabella I mostra che la regola di aggregazione dell'encoder è un **nullo**
— FedProx, FedProto e la sola inizializzazione condivisa sono indistinguibili da FedAvg. Ma quel
nullo è misurato **col prior locale**, cioè nel regime in cui, per tesi nostra, il prior non fa
niente. Estrapolarlo al regime col prior condiviso non è coperto dai dati.

Il vuoto **non è totale**: i due estremi dell'asse encoder col prior condiviso sono misurati.

| backbone | contrasto | W/L | mediana AUPRC |
|---|---|---|---|
| encoder **locali** | (f) − (c) | 4/6 | **−0,001** |
| encoder **FedAvg** | (h) − (g) | 9/1 | **+0,069** |

Manca il mezzo: **una partenza comune senza risincronizzazione**. È l'unica delle tre
alternative che non si colloca per struttura — FedProx media comunque i pesi al server (il
termine prossimale agisce sull'obiettivo locale), FedProto non li media mai.

> **H1.** Se l'allineamento richiede la *media dei pesi*, `commoninit` + prior condiviso
> riproduce il nullo di (f).
> **H0 (che ucciderebbe il meccanismo).** Se basta una *partenza comune*, riproduce il guadagno
> di (h) — e allora FedAvg sull'encoder in A2 non sta facendo il lavoro che gli attribuiamo.

## 2. Fase 0 — sonda κ (CPU, nessuna GPU, minuti)

Nessun codice nuovo: `scripts/latent_probe.py` calcola già l'accordo **corretto per il caso**
fra le assegnazioni di token dei client (null = Σ_j p_A(j)p_B(j), riga 430).

```
scripts/latent_probe.py --run-dir artifacts/runs/zn_enc \
  --arms federated_enc_fedavg,federated_enc_commoninit --all-complete
```

I checkpoint ci sono per **entrambi gli arm su tutte e 10 le serie** (verificato). Di `fedprox` e
`fedproto` sono rimasti solo gli out-json, quindi la sonda copre i due estremi — che è il
contrasto giusto per la domanda.

**A cosa serve:** è il *meccanismo*, non l'esito. Se κ crolla senza la media dei pesi, la Fase 1
diventa conferma di una spiegazione invece che scoperta di un fatto. Non è un gate: la Fase 1 si
fa comunque, perché κ alto non implica detection.

## 3. Fase 1 — la cella decisiva (GPU, ~2-3h su 5 corsie)

Tag **`zn_ci_prior`**, 10 serie di sviluppo, seed 0.

```
--s1-rounds 0 --s2-rounds 300 --fed-enc-prior partial --fed-enc-cb suffstat
--window-normalization zscore --protocol converged
--resume-from artifacts/runs/zn_enc/ckpt/ucr_split_w2p/<serie>/seed0/federated_enc_commoninit
```

`--s1-rounds 0` **congela lo stage 1**: si riusa esattamente l'encoder di `commoninit` e si
rifà il solo stage 2 col corpo del prior condiviso. La directory diventa
`federated_enc_commoninit_prior-partial` (il suffisso `prior-` è emesso quando il prior non è
locale, `federated_eval.py:1239`), quindi nessuna collisione col run esistente.

**Il contrasto è a FLAG SINGOLO** — unico caso in tutto il paper:
`zn_ci_prior` − `zn_enc/federated_enc_commoninit`, stesso stage 1 bit-identico, cambia solo il
prior. (g)→(h) e (c)→(f) sono invece coppie di run pieni.

⚠️ **Nessuno stage-2 knob**: niente `--tau-steps`, niente `--fed-s2-val fixed`, niente
`--server-opt`. Il regime di stage 2 deve essere quello di (f) e (h) (`epochs`), altrimenti il
contrasto non è confrontabile con le due colonne di riferimento.

⚠️ **Il regime BN è quello di `commoninit`** (affine condivisa solo all'init, running stats
locali), **non** quello di A2 (`pooled`). La cella è appaiata a `commoninit`, **non** ad A2, e
va dichiarato in tabella e in prosa. Questo è il motivo per cui il blocco esiste già come tale.

### Criteri di decisione, fissati ora

Metrica primaria **AUPRC media sui 5 client**, unità = **la serie**, n=10, appaiata. Riportata
anche `paper_top1_acc_at_64`. Δ_ci = `zn_ci_prior` − `commoninit`.

| esito | criterio | conseguenza |
|---|---|---|
| **H1 confermata** | Δ_ci ≤ 6 vittorie/10 **e** mediana < +0,035 (metà di +0,069) | la precondizione regge e guadagna un meccanismo: l'allineamento richiede la media dei pesi. Fase 2 **non si fa**; FedProx/FedProto restano una frase strutturale. |
| **H0, il meccanismo cade** | Δ_ci ≥ 8/10 **e** mediana ≥ +0,035 | FedAvg sull'encoder non è necessario: A2 è sovradimensionato. Fase 2 **obbligatoria** e §V.D va riscritta. |
| **intermedio** | tutto il resto | si riporta come intermedio e si gira **solo FedProto** (l'altro meccanismo che non media) per capire se è specifico di `commoninit`. |

Regole dure che valgono comunque: **train to convergence** (`--protocol converged`), le celle
TRUNCATED non sono riportabili; tag distinto registrato in `cohorts/zn_ownership.json` con
`scripts/zn_owner.py --check`; g2 con `export LAUNCH_GPUS="$(bash scripts/_g2_gpu.sh)"`; seed
singolo ⇒ **nessuna barra d'errore**, e differenze per-serie sotto qualche centesimo non sono
effetti.

## 4. Fase 2 — condizionale, e **molto più economica del previsto**

🔴 **Correzione 2026-08-13, dopo `latent_probe.py --list`.** Avevo scritto che di `fedprox` e
`fedproto` non restavano checkpoint. **È falso**: ci sono, su **8 serie su 10** (mancano solo
`ucr_001` e `ucr_011`). Le directory portano i parametri nel nome —
`federated_enc_fedproto_lam1_uniform` e `federated_enc_fedprox_mu0.01` — e il glob che avevo
usato cercava `federated_enc_fedproto`, quindi non trovava niente.

Conseguenza: la Fase 2 **non** è fatta di run pieni. Sono resume di solo stage 2 identici alla
Fase 1, ~8 celle per arm, ordine di 2 ore ciascuno su 5 corsie. L'intero grid regola × prior
costa quindi ~6-8 ore, non 20-40.

Questo **non** cambia i criteri di §3, e non cambia l'ordine: `commoninit` resta la cella
decisiva, perché è l'unica che non si colloca per struttura (FedProx media i pesi al server come
(h), FedProto non li media mai come (f)). Ma nel ramo H1 la Fase 2 diventa un completamento
opzionale a basso costo invece di una spesa da evitare, e nel ramo H0 non è più un ostacolo.

⚠️ Su `ucr_001` e `ucr_011` il grid resterebbe scoperto per FedProx/FedProto: n=8, e sono due
delle serie facili. Va dichiarato, non colmato con run pieni.

## 5. Rischi, e il dry run che li chiude

Il percorso di resume **non è mai stato esercitato su un arm della famiglia fedproto**. Prima
delle 10 celle si gira **una sola serie (`ucr_011`)** e si verifica, sul log:

1. la directory creata è `federated_enc_commoninit_prior-partial` e non riusa quella esistente;
2. lo stage 1 **non** viene riallenato (nessun round `[fed:suffstat]` nuovo);
3. il banner dello stage 2 dichiara **55 chiavi condivise** con channel embedding e output bias
   locali — le stesse di (h), altrimenti il contrasto non è lo stesso trattamento;
4. la cella chiude **CONVERGED**.

Se uno dei quattro non torna, ci si ferma e si rivede il piano invece di lanciare le altre nove.

## 6. Cosa NON si potrà dire

Che questa cella ablaziona A2: non lo fa, il regime BN è diverso. Che chiude l'asse encoder:
copre un meccanismo su tre, e gli altri due per struttura, non per misura. E niente sulle serie
non discriminanti, dove nessuno dei due bracci ha spazio per muoversi.
