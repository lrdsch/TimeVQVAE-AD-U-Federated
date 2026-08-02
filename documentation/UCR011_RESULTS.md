# `ucr_011` — floor + tutti gli arm deep

Generato da `scripts/ucr_series_report.py ucr_011` il **2026-08-01 05:21**. Rieseguibile: rilegge il disco.

| | |
|---|---|
| coorte | `ucr011`, fingerprint **`f321bb77e6ef3601`** |
| finestre | `ucr_split` W=128 · `ucr_split_w2p` W=182 |
| tag deep | `ucr011_v1` — 16/16 completi |
| tag floor | `ucr011_floor` |
| protocollo | `converged`, s1_rounds=300, s2_rounds=300, patience=6 |
| client | 5 per build, quantity-skew `[10,10,20,20,30] %` |
| commit | `a70668883297` ⚠️ **worktree dirty** |

⚠️ **Le tabelle qui girano su VUS-PR**, legittimo su **una** serie ma [non aggregabile fra serie UCR](UCR_WINDOW_ABLATION_SET.md). Il top-1 il deep lo calcola — sta in `<arm>/<client>/report.json` — solo che `METRIC_KEYS` non lo promuove al summary che questo report legge. Per la metrica ufficiale UCR: `scripts/topk_table.py`.


## Deep — `ucr_split` (W=128)

| arm | stato | round | best/ultimo | VUS-PR | AUPRC | AUROC | PATE-F1 | sd fra client |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `local` | OK | — | — | 0.410 | 0.448 | 0.981 | 0.452 | 0.208 |
| `centralized` | OK | — | — | 0.968 | 0.981 | 1.000 | 0.981 | 0.004 |
| `federated_cb_only_ema` | OK | 38 | r32 / 38 | 0.303 | 0.330 | 0.959 | 0.347 | 0.142 |
| `federated_cb_only` | OK | 32 | r26 / 32 | 0.337 | 0.362 | 0.967 | 0.384 | 0.166 |
| `federated_fedavg_cb_only` | OK | 31 | r25 / 31 | 0.467 | 0.497 | 0.984 | 0.503 | 0.246 |
| `federated_shared` | OK | 50 | r44 / 50 | 0.309 | 0.326 | 0.954 | 0.333 | 0.208 |
| `federated_fedavg_cb_sharedprior` | OK | 45 | r39 / 45 | 0.405 | 0.409 | 0.980 | 0.415 | 0.158 |
| `federated` | OK | 48 | r42 / 48 | 0.322 | 0.359 | 0.937 | 0.369 | 0.184 |

## Deep — `ucr_split_w2p` (W=182)

| arm | stato | round | best/ultimo | VUS-PR | AUPRC | AUROC | PATE-F1 | sd fra client |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `local` | OK | — | — | 0.488 | 0.529 | 0.985 | 0.541 | 0.241 |
| `centralized` | OK | — | — | 0.969 | 0.985 | 1.000 | 0.993 | 0.003 |
| `federated_cb_only_ema` | OK | 68 | r62 / 68 | 0.392 | 0.418 | 0.981 | 0.427 | 0.203 |
| `federated_cb_only` | OK | 87 | r81 / 87 | 0.505 | 0.541 | 0.987 | 0.555 | 0.227 |
| `federated_fedavg_cb_only` | OK | 41 | r35 / 41 | 0.610 | 0.647 | 0.990 | 0.674 | 0.192 |
| `federated_shared` | OK | 59 | r53 / 59 | 0.381 | 0.402 | 0.962 | 0.418 | 0.198 |
| `federated_fedavg_cb_sharedprior` | OK | 41 | r35 / 41 | 0.578 | 0.598 | 0.970 | 0.621 | 0.221 |
| `federated` | OK | 86 | r80 / 86 | 0.330 | 0.347 | 0.967 | 0.358 | 0.164 |

## Floor — `ucr_split`

| arm | n righe | W | VUS-PR | AUPRC | AUROC | paper top-1 |
|---|---:|---:|---:|---:|---:|---:|
| `floor_ar_p32_lam0.0001_tau16_rounds30__fed_localgd` | 5 | 128 | 0.207 | 0.237 | 0.969 | 1.00 |
| `floor_ar_p32_lam0.0001_cap2__central_capN` | 5 | 128 | 0.180 | 0.207 | 0.963 | 1.00 |
| `floor_ar_p32_lam0.0001__fed_oneclient` | 5 | 128 | 0.176 | 0.204 | 0.962 | 1.00 |
| `floor_ar_p32_lam0.0001__central` | 5 | 128 | 0.175 | 0.198 | 0.962 | 1.00 |
| `floor_ar_p32_lam0.0001__fed_exact` | 5 | 128 | 0.175 | 0.198 | 0.962 | 1.00 |
| `floor_ar_p32_lam0.0001__fed_fedavg_uniform` | 5 | 128 | 0.175 | 0.199 | 0.962 | 1.00 |
| `floor_ar_p32_lam0.0001` | 5 | 128 | 0.175 | 0.200 | 0.962 | 1.00 |
| `floor_ar_p32_lam0.0001__fed_scaleonly` | 5 | 128 | 0.175 | 0.200 | 0.962 | 1.00 |
| `floor_ar_p32_lam0.0001_mu0.1_rounds30__fed_prox` | 5 | 128 | 0.174 | 0.198 | 0.962 | 1.00 |
| `floor_ar_p32_lam0.0001__fed_fedavg` | 5 | 128 | 0.174 | 0.198 | 0.962 | 1.00 |
| `floor_pca_K8_gamma0.05_cap2__central_capN` | 5 | 128 | 0.112 | 0.133 | 0.940 | 1.00 |
| `floor_pca_K8_gamma0.05__fed_fedavg_uniform` | 5 | 128 | 0.111 | 0.133 | 0.939 | 1.00 |
| `floor_pca_K8_gamma0.05__fed_oneclient` | 5 | 128 | 0.111 | 0.132 | 0.939 | 1.00 |
| `floor_pca_K8_gamma0.05__fed_naive_aligned` | 5 | 128 | 0.111 | 0.132 | 0.940 | 1.00 |
| `floor_pca_K8_gamma0.05__fed_fedavg` | 5 | 128 | 0.109 | 0.130 | 0.938 | 1.00 |
| `floor_pca_K8_gamma0.05` | 5 | 128 | 0.105 | 0.126 | 0.935 | 1.00 |
| `floor_pca_K8_gamma0.05__fed_naive` | 5 | 128 | 0.105 | 0.124 | 0.935 | 1.00 |
| `floor_pca_K8_gamma0.05__central` | 5 | 128 | 0.092 | 0.113 | 0.924 | 1.00 |
| `floor_pca_K8_gamma0.05__fed_exact` | 5 | 128 | 0.092 | 0.113 | 0.924 | 1.00 |
| `floor_ma_causal_k10` | 5 | 128 | 0.075 | 0.115 | 0.880 | 1.00 |
| `floor_ma_c_k10` | 5 | 128 | 0.062 | 0.084 | 0.793 | 1.00 |

## Floor — `ucr_split_w2p`

| arm | n righe | W | VUS-PR | AUPRC | AUROC | paper top-1 |
|---|---:|---:|---:|---:|---:|---:|
| `floor_ar_p32_lam0.0001_tau16_rounds30__fed_localgd__w182` | 5 | 182 | 0.199 | 0.226 | 0.967 | 1.00 |
| `floor_ar_p32_lam0.0001_cap2__central_capN__w182` | 5 | 182 | 0.173 | 0.195 | 0.959 | 1.00 |
| `floor_ar_p32_lam0.0001__fed_oneclient__w182` | 5 | 182 | 0.169 | 0.193 | 0.958 | 1.00 |
| `floor_ar_p32_lam0.0001__fed_fedavg_uniform__w182` | 5 | 182 | 0.168 | 0.188 | 0.958 | 1.00 |
| `floor_ar_p32_lam0.0001__fed_scaleonly__w182` | 5 | 182 | 0.167 | 0.189 | 0.958 | 1.00 |
| `floor_ar_p32_lam0.0001__w182` | 5 | 182 | 0.167 | 0.189 | 0.958 | 1.00 |
| `floor_ar_p32_lam0.0001__fed_fedavg__w182` | 5 | 182 | 0.167 | 0.187 | 0.958 | 1.00 |
| `floor_ar_p32_lam0.0001__central__w182` | 5 | 182 | 0.167 | 0.187 | 0.958 | 1.00 |
| `floor_ar_p32_lam0.0001__fed_exact__w182` | 5 | 182 | 0.167 | 0.187 | 0.958 | 1.00 |
| `floor_ar_p32_lam0.0001_mu0.1_rounds30__fed_prox__w182` | 5 | 182 | 0.167 | 0.187 | 0.958 | 1.00 |
| `floor_pca_K8_gamma0.05__fed_naive__w182` | 5 | 182 | 0.092 | 0.112 | 0.924 | 1.00 |
| `floor_pca_K8_gamma0.05__fed_naive_aligned__w182` | 5 | 182 | 0.090 | 0.111 | 0.919 | 1.00 |
| `floor_pca_K8_gamma0.05__fed_fedavg__w182` | 5 | 182 | 0.089 | 0.110 | 0.916 | 1.00 |
| `floor_pca_K8_gamma0.05__central__w182` | 5 | 182 | 0.089 | 0.109 | 0.915 | 1.00 |
| `floor_pca_K8_gamma0.05__fed_exact__w182` | 5 | 182 | 0.089 | 0.109 | 0.915 | 1.00 |
| `floor_pca_K8_gamma0.05__fed_fedavg_uniform__w182` | 5 | 182 | 0.088 | 0.110 | 0.916 | 1.00 |
| `floor_pca_K8_gamma0.05_cap2__central_capN__w182` | 5 | 182 | 0.088 | 0.109 | 0.916 | 1.00 |
| `floor_pca_K8_gamma0.05__fed_oneclient__w182` | 5 | 182 | 0.088 | 0.109 | 0.916 | 1.00 |
| `floor_pca_K8_gamma0.05__w182` | 5 | 182 | 0.087 | 0.108 | 0.915 | 1.00 |
| `floor_ma_causal_k10__w182` | 5 | 182 | 0.072 | 0.107 | 0.874 | 1.00 |
| `floor_ma_c_k10__w182` | 5 | 182 | 0.071 | 0.092 | 0.883 | 1.00 |

## Ricostruzione contro rilevamento — il protocollo di stop guarda la cosa sbagliata

| build | arm | merge | val_loss s1 al best | perplexity | VUS-PR |
|---|---|---|---:|---:|---:|
| `ucr_split` | `federated_cb_only_ema` | suffstat | 0.1899 | 10/64 | 0.303 |
| `ucr_split` | `federated_cb_only` | suffstat | 0.1880 | 16/64 | 0.337 |
| `ucr_split` | `federated_fedavg_cb_only` | fedavg | 0.1493 | 51/64 | 0.467 |
| `ucr_split` | `federated_shared` | suffstat | 0.1574 | 20/64 | 0.309 |
| `ucr_split` | `federated_fedavg_cb_sharedprior` | fedavg | 0.1339 | 54/64 | 0.405 |
| `ucr_split` | `federated` | suffstat | 0.1334 | 17/64 | 0.322 |
| `ucr_split_w2p` | `federated_cb_only_ema` | suffstat | 0.2149 | 12/64 | 0.392 |
| `ucr_split_w2p` | `federated_cb_only` | suffstat | 0.1895 | 24/64 | 0.505 |
| `ucr_split_w2p` | `federated_fedavg_cb_only` | fedavg | 0.1951 | 51/64 | 0.610 |
| `ucr_split_w2p` | `federated_shared` | suffstat | 0.2028 | 16/64 | 0.381 |
| `ucr_split_w2p` | `federated_fedavg_cb_sharedprior` | fedavg | 0.1899 | 49/64 | 0.578 |
| `ucr_split_w2p` | `federated` | suffstat | 0.1997 | 24/64 | 0.330 |

Su questa serie val_loss e VUS-PR sono **scorrelate** (Pearson **+0.18** su 12 arm). Attesa ingenua: correlazione negativa forte — ricostruire meglio dovrebbe rilevare meglio.

🔴 **Precedente registrato (misurato su `ucr_001`, 2026-07-31).** Lì la ricostruzione **anti-prediceva** il rilevamento: `federated_fedavg_cb_sharedprior` a W=408 aveva la val_loss di stage 1 migliore del suo build (0,2150) e il detector peggiore di 17× (0,027).

❌ **Ipotesi scartata lì, da non riciclare qui: non è la perplexity.** Sembrava che un dizionario stretto rilevasse meglio (suff-stat 4–9 codeword contro FedAvg 43). `federated_fedavg_cb_only` la refuta: perplexity **57** e VUS-PR **0,418** sano, mentre l'arm collassato usava *meno* codeword (43). La perplexity non ordina i risultati.

⚠️ **Conseguenza sul protocollo.** `--protocol converged` ferma il training sulla val_loss di RICOSTRUZIONE, ma la metrica riportata è il RILEVAMENTO. Un arm può essere «convergito» a pieno titolo e inutile come detector, e la tabella non lo distingue da un arm fermato male. Questo vale per ogni riga, non solo per quelle FedAvg.


## Il fattoriale 2×2 — primitiva del codebook × condivisione del prior

**`ucr_split` (W=128)** — VUS-PR, media sui 5 client

| merge del codebook | prior LOCALE | prior CONDIVISO | effetto del prior |
|---|---:|---:|---:|
| **suff-stat** (Prop. 1) | 0.337 | 0.309 | **-0.028** |
| FedAvg (media pesata) | 0.467 | 0.405 | **-0.062** |
| **effetto del merge** | **-0.130** | **-0.096** | |

**`ucr_split_w2p` (W=182)** — VUS-PR, media sui 5 client

| merge del codebook | prior LOCALE | prior CONDIVISO | effetto del prior |
|---|---:|---:|---:|
| **suff-stat** (Prop. 1) | 0.505 | 0.381 | **-0.124** |
| FedAvg (media pesata) | 0.610 | 0.578 | **-0.033** |
| **effetto del merge** | **-0.106** | **-0.197** | |

Riferimenti nella stessa colonna: `local` e `centralized` in cima alla sezione del build. Un arm federato che sta sotto `local` non sta pagando la federazione — sta peggiorando rispetto a non federare affatto.

⚠️ Leggere insieme al controllo dei gemelli qui sotto: dove la riga FedAvg ha una divergenza di stage 1 grande, la sua cella è **un sorteggio**, e l'interazione fra i due assi non è interpretabile a un seed solo.


## Controllo di riproducibilità — gemelli a stage 1 identico

| build | coppia | merge | val_loss di stage 1 all'ultimo round comune | divergenza |
|---|---|---|---|---:|
| `ucr_split` | `federated_cb_only` / `federated_shared` | suff-stat | r32: 0.1947 / 0.1797 | ok **1.1×** |
| `ucr_split` | `federated_fedavg_cb_only` / `federated_fedavg_cb_sharedprior` | FedAvg | r31: 0.1523 / 0.1557 | ok **1.0×** |
| `ucr_split_w2p` | `federated_cb_only` / `federated_shared` | suff-stat | r59: 0.1997 / 0.2051 | ok **1.0×** |
| `ucr_split_w2p` | `federated_fedavg_cb_only` / `federated_fedavg_cb_sharedprior` | FedAvg | r41: 0.2014 / 0.2049 | ok **1.0×** |

Le due colonne di ogni coppia **dovrebbero coincidere**: stesso seed, stessa configurazione di stage 1, e la seeding per round/client è esplicita (`_round_seed(seed, round, client, stage)`). Non coincidono perché `cfg.deterministic = False` (config.py:281), `torch.backends.cudnn.benchmark = True` (federated.py:1290) e l'AMP gira in **float16**: la selezione dei kernel e le riduzioni non deterministiche fanno divergere le traiettorie da differenze alla quinta cifra.

🔴 **Conseguenza da non aggirare.** Dove la divergenza è grande, un contrasto a UN SOLO SEED fra due arm non misura l'effetto dell'arm: misura il sorteggio. Peggio, l'early stopping lo cristallizza — un arm che pianeggia per 6 round viene fermato e il suo plateau diventa «il risultato». Prima di mettere in tabella una differenza fra arm servono **più seed**, oppure `AMP=0` + `deterministic=True` per togliere la sorgente di rumore.


## Ablazione sulla finestra — W=128 contro W=2P (182), rapporto 1.42×

Stessa serie, stessi 5 client, stesso protocollo: **unica variabile W**. Solo gli arm chiusi su ENTRAMBI i build compaiono qui — un arm a metà non è un confronto appaiato.

| arm | VUS-PR | AUPRC | AUROC |
|---|---:|---:|---:|
| `local` | 0.410 → 0.488 (**+0.078**) | 0.448 → 0.529 (**+0.081**) | 0.981 → 0.985 (**+0.004**) |
| `centralized` | 0.968 → 0.969 (**+0.002**) | 0.981 → 0.985 (**+0.005**) | 1.000 → 1.000 (**+0.000**) |
| `federated_cb_only_ema` | 0.303 → 0.392 (**+0.090**) | 0.330 → 0.418 (**+0.088**) | 0.959 → 0.981 (**+0.022**) |
| `federated_cb_only` | 0.337 → 0.505 (**+0.168**) | 0.362 → 0.541 (**+0.179**) | 0.967 → 0.987 (**+0.020**) |
| `federated_fedavg_cb_only` | 0.467 → 0.610 (**+0.144**) | 0.497 → 0.647 (**+0.150**) | 0.984 → 0.990 (**+0.006**) |
| `federated_shared` | 0.309 → 0.381 (**+0.072**) | 0.326 → 0.402 (**+0.076**) | 0.954 → 0.962 (**+0.008**) |
| `federated_fedavg_cb_sharedprior` | 0.405 → 0.578 (**+0.173**) | 0.409 → 0.598 (**+0.189**) | 0.980 → 0.970 (**-0.010**) |
| `federated` | 0.322 → 0.330 (**+0.007**) | 0.359 → 0.347 (**-0.013**) | 0.937 → 0.967 (**+0.031**) |

8/8 arm appaiati. Un delta positivo dice che agganciare la finestra al periodo (`T = 2 × periodo`, come il paper) batte la nostra W=128 fissa.

⚠️ Due avvertenze sul segno. (i) Le metriche **a soglia** non seguono: su `centralized` affiliation-F1 fa 0,827 → 0,671 e F1 0,589 → 0,459. La finestra larga **ordina** meglio, ma la soglia al quantile 0,99 le va peggio — è una storia sul ranking, non su un detector calibrato. (ii) La frazione di rete federata cambia con W (encoder 44,9 % a W=128, 40,1 % a W=512), quindi parte del delta sugli arm federati è cambio di perimetro, non di finestra: vedi `UCR_WINDOW_ABLATION_SET.md` §5.1.


## Deep contro floor

| build | miglior floor (VUS-PR) | miglior deep (VUS-PR) | rapporto |
|---|---:|---:|---:|
| `ucr_split` | 0.207 | 0.968 | 5x |
| `ucr_split_w2p` | 0.199 | 0.969 | 5x |

Un rapporto ≫ 1 dice che il deep stacca il floor su questa serie. È il **contrario** di quanto misurato su `wsd_fed`, dove `movavg10` a zero parametri sta a 0,507 e batte il trio `enc_*`. Una serie non è una tendenza: va confermato sulle altre 9 prima di scriverlo.

