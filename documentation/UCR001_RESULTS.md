# `ucr_001` — floor + tutti gli arm deep

Generato da `scripts/ucr_series_report.py ucr_001` il **2026-08-01 04:29**. Rieseguibile: rilegge il disco.

| | |
|---|---|
| coorte | `ucr001`, fingerprint **`c604c97f8001290c`** |
| finestre | `ucr_split` W=128 · `ucr_split_w2p` W=408 |
| tag deep | `ucr001_v1` — 16/16 completi |
| tag floor | `ucr001_floor` |
| protocollo | `converged`, s1_rounds=300, s2_rounds=300, patience=6 |
| client | 5 per build, quantity-skew `[10,10,20,20,30] %` |
| commit | `a70668883297` ⚠️ **worktree dirty** |

⚠️ **Le tabelle qui girano su VUS-PR**, legittimo su **una** serie ma [non aggregabile fra serie UCR](UCR_WINDOW_ABLATION_SET.md). Il top-1 il deep lo calcola — sta in `<arm>/<client>/report.json` — solo che `METRIC_KEYS` non lo promuove al summary che questo report legge. Per la metrica ufficiale UCR: `scripts/topk_table.py`.


## Deep — `ucr_split` (W=128)

| arm | stato | round | best/ultimo | VUS-PR | AUPRC | AUROC | PATE-F1 | sd fra client |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `local` | OK | — | — | 0.353 | 0.374 | 0.832 | 0.414 | 0.099 |
| `centralized` | OK | — | — | 0.556 | 0.593 | 0.832 | 0.673 | 0.005 |
| `federated_cb_only_ema` | OK | 63 | r57 / 63 | 0.315 | 0.348 | 0.823 | 0.390 | 0.072 |
| `federated_cb_only` | OK | 26 | r20 / 26 | 0.404 | 0.434 | 0.827 | 0.503 | 0.110 |
| `federated_fedavg_cb_only` | OK | 12 | r6 / 12 | 0.230 | 0.239 | 0.809 | 0.261 | 0.135 |
| `federated_shared` | OK | 26 | r20 / 26 | 0.280 | 0.312 | 0.803 | 0.358 | 0.046 |
| `federated_fedavg_cb_sharedprior` | OK | 28 | r22 / 28 | 0.373 | 0.381 | 0.827 | 0.402 | 0.129 |
| `federated` | OK | 42 | r36 / 42 | 0.293 | 0.323 | 0.778 | 0.359 | 0.106 |

## Deep — `ucr_split_w2p` (W=408)

| arm | stato | round | best/ultimo | VUS-PR | AUPRC | AUROC | PATE-F1 | sd fra client |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `local` | OK | — | — | 0.468 | 0.474 | 0.938 | 0.533 | 0.131 |
| `centralized` | OK | — | — | 0.618 | 0.646 | 0.955 | 0.722 | 0.018 |
| `federated_cb_only_ema` | OK | 17 | r11 / 17 | 0.422 | 0.442 | 0.953 | 0.508 | 0.135 |
| `federated_cb_only` | OK | 20 | r15 / 20 | 0.461 | 0.481 | 0.949 | 0.547 | 0.155 |
| `federated_fedavg_cb_only` | OK | 27 | r21 / 27 | 0.418 | 0.432 | 0.936 | 0.472 | 0.184 |
| `federated_shared` | OK | 27 | r21 / 27 | 0.177 | 0.181 | 0.886 | 0.200 | 0.122 |
| `federated_fedavg_cb_sharedprior` | OK | 23 | r17 / 23 | 0.027 | 0.026 | 0.619 | 0.026 | 0.017 |
| `federated` | OK | 21 | r15 / 21 | 0.257 | 0.256 | 0.908 | 0.290 | 0.175 |

## Floor — `ucr_split`

| arm | n righe | W | VUS-PR | AUPRC | AUROC | paper top-1 |
|---|---:|---:|---:|---:|---:|---:|
| `floor_ar_p32_lam0.0001_tau16_rounds30__fed_localgd` | 5 | 128 | 0.016 | 0.016 | 0.561 | 0.00 |
| `floor_pca_K8_gamma0.05__fed_oneclient` | 5 | 128 | 0.013 | 0.012 | 0.499 | 0.00 |
| `floor_ar_p32_lam0.0001__central` | 5 | 128 | 0.012 | 0.012 | 0.470 | 0.00 |
| `floor_ar_p32_lam0.0001__fed_exact` | 5 | 128 | 0.012 | 0.012 | 0.470 | 0.00 |
| `floor_ar_p32_lam0.0001__fed_fedavg` | 5 | 128 | 0.012 | 0.012 | 0.469 | 0.00 |
| `floor_ar_p32_lam0.0001__fed_oneclient` | 5 | 128 | 0.012 | 0.012 | 0.468 | 0.00 |
| `floor_ar_p32_lam0.0001__fed_fedavg_uniform` | 5 | 128 | 0.012 | 0.012 | 0.468 | 0.00 |
| `floor_ar_p32_lam0.0001_mu0.1_rounds30__fed_prox` | 5 | 128 | 0.012 | 0.012 | 0.466 | 0.00 |
| `floor_ar_p32_lam0.0001` | 5 | 128 | 0.012 | 0.012 | 0.465 | 0.00 |
| `floor_ar_p32_lam0.0001__fed_scaleonly` | 5 | 128 | 0.012 | 0.012 | 0.465 | 0.00 |
| `floor_pca_K8_gamma0.05_cap2__central_capN` | 5 | 128 | 0.012 | 0.012 | 0.479 | 0.00 |
| `floor_ar_p32_lam0.0001_cap2__central_capN` | 5 | 128 | 0.012 | 0.012 | 0.462 | 0.00 |
| `floor_pca_K8_gamma0.05` | 5 | 128 | 0.012 | 0.012 | 0.463 | 0.00 |
| `floor_pca_K8_gamma0.05__fed_fedavg_uniform` | 5 | 128 | 0.012 | 0.012 | 0.461 | 0.00 |
| `floor_pca_K8_gamma0.05__fed_fedavg` | 5 | 128 | 0.012 | 0.011 | 0.451 | 0.00 |
| `floor_pca_K8_gamma0.05__central` | 5 | 128 | 0.011 | 0.011 | 0.446 | 0.00 |
| `floor_pca_K8_gamma0.05__fed_exact` | 5 | 128 | 0.011 | 0.011 | 0.446 | 0.00 |
| `floor_pca_K8_gamma0.05__fed_naive_aligned` | 5 | 128 | 0.011 | 0.011 | 0.441 | 0.00 |
| `floor_pca_K8_gamma0.05__fed_naive` | 5 | 128 | 0.011 | 0.011 | 0.419 | 0.00 |
| `floor_ma_causal_k10` | 5 | 128 | 0.010 | 0.010 | 0.362 | 0.00 |
| `floor_ma_c_k10` | 5 | 128 | 0.008 | 0.008 | 0.240 | 0.00 |

## Floor — `ucr_split_w2p`

| arm | n righe | W | VUS-PR | AUPRC | AUROC | paper top-1 |
|---|---:|---:|---:|---:|---:|---:|
| `floor_pca_K8_gamma0.05__central__w408` | 5 | 408 | 0.187 | 0.178 | 0.948 | 0.00 |
| `floor_pca_K8_gamma0.05__fed_exact__w408` | 5 | 408 | 0.187 | 0.178 | 0.948 | 0.00 |
| `floor_pca_K8_gamma0.05__fed_fedavg_uniform__w408` | 5 | 408 | 0.187 | 0.177 | 0.947 | 0.00 |
| `floor_pca_K8_gamma0.05__fed_fedavg__w408` | 5 | 408 | 0.187 | 0.177 | 0.948 | 0.00 |
| `floor_pca_K8_gamma0.05__fed_oneclient__w408` | 5 | 408 | 0.186 | 0.175 | 0.947 | 0.00 |
| `floor_pca_K8_gamma0.05_cap2__central_capN__w408` | 5 | 408 | 0.186 | 0.176 | 0.946 | 0.00 |
| `floor_pca_K8_gamma0.05__w408` | 5 | 408 | 0.159 | 0.150 | 0.933 | 0.00 |
| `floor_pca_K8_gamma0.05__fed_naive_aligned__w408` | 5 | 408 | 0.128 | 0.124 | 0.939 | 0.00 |
| `floor_pca_K8_gamma0.05__fed_naive__w408` | 5 | 408 | 0.091 | 0.089 | 0.926 | 0.00 |
| `floor_ar_p32_lam0.0001_tau16_rounds30__fed_localgd__w408` | 5 | 408 | 0.015 | 0.015 | 0.556 | 0.00 |
| `floor_ar_p32_lam0.0001__central__w408` | 5 | 408 | 0.011 | 0.011 | 0.404 | 0.00 |
| `floor_ar_p32_lam0.0001__fed_exact__w408` | 5 | 408 | 0.011 | 0.011 | 0.404 | 0.00 |
| `floor_ar_p32_lam0.0001__fed_fedavg__w408` | 5 | 408 | 0.011 | 0.010 | 0.403 | 0.00 |
| `floor_ar_p32_lam0.0001__fed_oneclient__w408` | 5 | 408 | 0.011 | 0.010 | 0.401 | 0.00 |
| `floor_ar_p32_lam0.0001__fed_fedavg_uniform__w408` | 5 | 408 | 0.011 | 0.010 | 0.401 | 0.00 |
| `floor_ar_p32_lam0.0001_mu0.1_rounds30__fed_prox__w408` | 5 | 408 | 0.011 | 0.010 | 0.400 | 0.00 |
| `floor_ar_p32_lam0.0001__fed_scaleonly__w408` | 5 | 408 | 0.011 | 0.010 | 0.398 | 0.00 |
| `floor_ar_p32_lam0.0001__w408` | 5 | 408 | 0.011 | 0.010 | 0.398 | 0.00 |
| `floor_ar_p32_lam0.0001_cap2__central_capN__w408` | 5 | 408 | 0.011 | 0.010 | 0.393 | 0.00 |
| `floor_ma_c_k10__w408` | 5 | 408 | 0.008 | 0.007 | 0.075 | 0.00 |
| `floor_ma_causal_k10__w408` | 5 | 408 | 0.008 | 0.007 | 0.115 | 0.00 |

## Ricostruzione contro rilevamento — il protocollo di stop guarda la cosa sbagliata

| build | arm | merge | val_loss s1 al best | perplexity | VUS-PR |
|---|---|---|---:|---:|---:|
| `ucr_split` | `federated_cb_only_ema` | suffstat | 0.0698 | 20/64 | 0.315 |
| `ucr_split` | `federated_cb_only` | suffstat | 0.0632 | 19/64 | 0.404 |
| `ucr_split` | `federated_fedavg_cb_only` | fedavg | 0.2875 | 59/64 | 0.230 |
| `ucr_split` | `federated_shared` | suffstat | 0.0525 | 18/64 | 0.280 |
| `ucr_split` | `federated_fedavg_cb_sharedprior` | fedavg | 0.0226 | 58/64 | 0.373 |
| `ucr_split` | `federated` | suffstat | 0.0547 | 16/64 | 0.293 |
| `ucr_split_w2p` | `federated_cb_only_ema` | suffstat | 0.3665 | 4/64 | 0.422 |
| `ucr_split_w2p` | `federated_cb_only` | suffstat | 0.2404 | 9/64 | 0.461 |
| `ucr_split_w2p` | `federated_fedavg_cb_only` | fedavg | 0.2254 | 57/64 | 0.418 |
| `ucr_split_w2p` | `federated_shared` | suffstat | 0.2507 | 11/64 | 0.177 |
| `ucr_split_w2p` | `federated_fedavg_cb_sharedprior` | fedavg | 0.2150 | 43/64 | 0.027 |
| `ucr_split_w2p` | `federated` | suffstat | 0.2545 | 10/64 | 0.257 |

Su questa serie val_loss e VUS-PR sono **scorrelate** (Pearson **-0.08** su 12 arm). Attesa ingenua: correlazione negativa forte — ricostruire meglio dovrebbe rilevare meglio.

🔴 **Precedente registrato (misurato su `ucr_001`, 2026-07-31).** Lì la ricostruzione **anti-prediceva** il rilevamento: `federated_fedavg_cb_sharedprior` a W=408 aveva la val_loss di stage 1 migliore del suo build (0,2150) e il detector peggiore di 17× (0,027).

❌ **Ipotesi scartata lì, da non riciclare qui: non è la perplexity.** Sembrava che un dizionario stretto rilevasse meglio (suff-stat 4–9 codeword contro FedAvg 43). `federated_fedavg_cb_only` la refuta: perplexity **57** e VUS-PR **0,418** sano, mentre l'arm collassato usava *meno* codeword (43). La perplexity non ordina i risultati.

⚠️ **Conseguenza sul protocollo.** `--protocol converged` ferma il training sulla val_loss di RICOSTRUZIONE, ma la metrica riportata è il RILEVAMENTO. Un arm può essere «convergito» a pieno titolo e inutile come detector, e la tabella non lo distingue da un arm fermato male. Questo vale per ogni riga, non solo per quelle FedAvg.


## Il fattoriale 2×2 — primitiva del codebook × condivisione del prior

**`ucr_split` (W=128)** — VUS-PR, media sui 5 client

| merge del codebook | prior LOCALE | prior CONDIVISO | effetto del prior |
|---|---:|---:|---:|
| **suff-stat** (Prop. 1) | 0.404 | 0.280 | **-0.124** |
| FedAvg (media pesata) | 0.230 | 0.373 | **+0.143** |
| **effetto del merge** | **+0.174** | **-0.093** | |

**`ucr_split_w2p` (W=408)** — VUS-PR, media sui 5 client

| merge del codebook | prior LOCALE | prior CONDIVISO | effetto del prior |
|---|---:|---:|---:|
| **suff-stat** (Prop. 1) | 0.461 | 0.177 | **-0.284** |
| FedAvg (media pesata) | 0.418 | 0.027 | **-0.391** |
| **effetto del merge** | **+0.043** | **+0.150** | |

Riferimenti nella stessa colonna: `local` e `centralized` in cima alla sezione del build. Un arm federato che sta sotto `local` non sta pagando la federazione — sta peggiorando rispetto a non federare affatto.

⚠️ Leggere insieme al controllo dei gemelli qui sotto: dove la riga FedAvg ha una divergenza di stage 1 grande, la sua cella è **un sorteggio**, e l'interazione fra i due assi non è interpretabile a un seed solo.


## Controllo di riproducibilità — gemelli a stage 1 identico

| build | coppia | merge | val_loss di stage 1 all'ultimo round comune | divergenza |
|---|---|---|---|---:|
| `ucr_split` | `federated_cb_only` / `federated_shared` | suff-stat | r26: 0.0669 / 0.0589 | ok **1.1×** |
| `ucr_split` | `federated_fedavg_cb_only` / `federated_fedavg_cb_sharedprior` | FedAvg | r12: 0.3039 / 0.0555 | 🔴 **5.5×** |
| `ucr_split_w2p` | `federated_cb_only` / `federated_shared` | suff-stat | r20: 0.2709 / 0.2636 | ok **1.0×** |
| `ucr_split_w2p` | `federated_fedavg_cb_only` / `federated_fedavg_cb_sharedprior` | FedAvg | r23: 0.2530 / 0.2702 | ok **1.1×** |

Le due colonne di ogni coppia **dovrebbero coincidere**: stesso seed, stessa configurazione di stage 1, e la seeding per round/client è esplicita (`_round_seed(seed, round, client, stage)`). Non coincidono perché `cfg.deterministic = False` (config.py:281), `torch.backends.cudnn.benchmark = True` (federated.py:1290) e l'AMP gira in **float16**: la selezione dei kernel e le riduzioni non deterministiche fanno divergere le traiettorie da differenze alla quinta cifra.

🔴 **Conseguenza da non aggirare.** Dove la divergenza è grande, un contrasto a UN SOLO SEED fra due arm non misura l'effetto dell'arm: misura il sorteggio. Peggio, l'early stopping lo cristallizza — un arm che pianeggia per 6 round viene fermato e il suo plateau diventa «il risultato». Prima di mettere in tabella una differenza fra arm servono **più seed**, oppure `AMP=0` + `deterministic=True` per togliere la sorgente di rumore.


## Ablazione sulla finestra — W=128 contro W=2P (408), rapporto 3.19×

Stessa serie, stessi 5 client, stesso protocollo: **unica variabile W**. Solo gli arm chiusi su ENTRAMBI i build compaiono qui — un arm a metà non è un confronto appaiato.

| arm | VUS-PR | AUPRC | AUROC |
|---|---:|---:|---:|
| `local` | 0.353 → 0.468 (**+0.115**) | 0.374 → 0.474 (**+0.100**) | 0.832 → 0.938 (**+0.106**) |
| `centralized` | 0.556 → 0.618 (**+0.063**) | 0.593 → 0.646 (**+0.053**) | 0.832 → 0.955 (**+0.123**) |
| `federated_cb_only_ema` | 0.315 → 0.422 (**+0.106**) | 0.348 → 0.442 (**+0.093**) | 0.823 → 0.953 (**+0.129**) |
| `federated_cb_only` | 0.404 → 0.461 (**+0.056**) | 0.434 → 0.481 (**+0.048**) | 0.827 → 0.949 (**+0.122**) |
| `federated_fedavg_cb_only` | 0.230 → 0.418 (**+0.188**) | 0.239 → 0.432 (**+0.193**) | 0.809 → 0.936 (**+0.128**) |
| `federated_shared` | 0.280 → 0.177 (**-0.103**) | 0.312 → 0.181 (**-0.131**) | 0.803 → 0.886 (**+0.083**) |
| `federated_fedavg_cb_sharedprior` | 0.373 → 0.027 (**-0.346**) | 0.381 → 0.026 (**-0.356**) | 0.827 → 0.619 (**-0.209**) |
| `federated` | 0.293 → 0.257 (**-0.037**) | 0.323 → 0.256 (**-0.067**) | 0.778 → 0.908 (**+0.130**) |

8/8 arm appaiati. Un delta positivo dice che agganciare la finestra al periodo (`T = 2 × periodo`, come il paper) batte la nostra W=128 fissa.

⚠️ Due avvertenze sul segno. (i) Le metriche **a soglia** non seguono: su `centralized` affiliation-F1 fa 0,827 → 0,671 e F1 0,589 → 0,459. La finestra larga **ordina** meglio, ma la soglia al quantile 0,99 le va peggio — è una storia sul ranking, non su un detector calibrato. (ii) La frazione di rete federata cambia con W (encoder 44,9 % a W=128, 40,1 % a W=512), quindi parte del delta sugli arm federati è cambio di perimetro, non di finestra: vedi `UCR_WINDOW_ABLATION_SET.md` §5.1.


## Deep contro floor

| build | miglior floor (VUS-PR) | miglior deep (VUS-PR) | rapporto |
|---|---:|---:|---:|
| `ucr_split` | 0.016 | 0.556 | 34x |
| `ucr_split_w2p` | 0.187 | 0.618 | 3x |

Un rapporto ≫ 1 dice che il deep stacca il floor su questa serie. È il **contrario** di quanto misurato su `wsd_fed`, dove `movavg10` a zero parametri sta a 0,507 e batte il trio `enc_*`. Una serie non è una tendenza: va confermato sulle altre 9 prima di scriverlo.

