# Tabella paper — `paper_top1`, aggregata fra serie

Metrica ufficiale UCR: **1 = anomalia trovata, 0 = mancata**. Ogni cella è la media sui 5 client di quella serie — i client condividono il test set (ICC 0.82), quindi la media è l'accuratezza attesa di un client a caso, e **l'unità per qualunque statistica fra serie è la serie, non il client**.

VUS-PR non compare apposta: [non è aggregabile fra serie UCR](UCR_WINDOW_ABLATION_SET.md). Letti da `report.json`, non ricalcolati.


## `ucr_split` — tolleranza 64

| arm | `ucr_001` | `ucr_011` | media | n |
|---|---:|---:|---:|---:|
| `local` | 0.60 | 0.80 | **0.700** | 2 |
| `centralized` | 1.00 | 1.00 | **1.000** | 2 |
| `federated_cb_only` | 0.80 | 0.60 | **0.700** | 2 |
| `federated_cb_only [K=128]` | 0.60 | 0.80 | **0.700** | 2 |
| `federated_cb_only_ema` | 0.80 | 0.80 | **0.800** | 2 |
| `federated_fedavg_cb_only` | 0.40 | 0.80 | **0.600** | 2 |
| `federated_fedavg_cb_only [K=128]` | 0.60 | 0.60 | **0.600** | 2 |
| `federated` | 0.80 | 0.80 | **0.800** | 2 |
| `federated_shared` | 0.60 | 0.60 | **0.600** | 2 |
| `federated_fedavg_cb_sharedprior` | 0.40 | 0.40 | **0.400** | 2 |
| `federated_enc_commoninit` | 0.40 | 0.80 | **0.600** | 2 |
| `federated_enc_fedavg` | 0.80 | 1.00 | **0.900** | 2 |
| `federated_enc_fedprox [mu0.01]` | 0.80 | 1.00 | **0.900** | 2 |
| `federated_enc_fedproto [lam1_count agg=count]` | 0.60 | 0.60 | **0.600** | 2 |
| `federated_enc_fedproto [lam1_uniform]` | 0.80 | 0.60 | **0.700** | 2 |

⚠️ **2 serie**: la colonna «media» è una media di 2 punti, non una stima. Nessun ordinamento fra arm è distinguibile dal rumore a questo n.

## `ucr_split_w2p` — tolleranza 64

| arm | `ucr_001` | `ucr_011` | media | n |
|---|---:|---:|---:|---:|
| `local` | 0.60 | 1.00 | **0.800** | 2 |
| `centralized` | 1.00 | 1.00 | **1.000** | 2 |
| `federated_cb_only` | 0.80 | 1.00 | **0.900** | 2 |
| `federated_cb_only [K=128]` | 0.60 | 0.60 | **0.600** | 2 |
| `federated_cb_only_ema` | 0.60 | 0.60 | **0.600** | 2 |
| `federated_fedavg_cb_only` | 0.60 | 1.00 | **0.800** | 2 |
| `federated_fedavg_cb_only [K=128]` | 0.40 | 0.80 | **0.600** | 2 |
| `federated` | 0.20 | 0.60 | **0.400** | 2 |
| `federated_shared` | 0.20 | 0.60 | **0.400** | 2 |
| `federated_fedavg_cb_sharedprior` | 0.00 | 0.80 | **0.400** | 2 |
| `federated_enc_commoninit` | 0.60 | 1.00 | **0.800** | 2 |
| `federated_enc_fedavg` | 0.20 | 0.60 | **0.400** | 2 |
| `federated_enc_fedprox [mu0.01]` | 0.80 | 1.00 | **0.900** | 2 |
| `federated_enc_fedproto [lam1_count agg=count]` | 0.60 | 0.80 | **0.700** | 2 |
| `federated_enc_fedproto [lam1_uniform]` | 0.40 | 0.60 | **0.500** | 2 |

⚠️ **2 serie**: la colonna «media» è una media di 2 punti, non una stima. Nessun ordinamento fra arm è distinguibile dal rumore a questo n.

---

## `ucr_split` — tolleranza 64

| arm | `ucr_001` | `ucr_011` | media | n |
|---|---:|---:|---:|---:|
| `local` | +0.20 | +0.00 | **+0.100** | 2 |
| `centralized` | +0.60 | +0.20 | **+0.400** | 2 |
| `federated_cb_only` | +0.40 | -0.20 | **+0.100** | 2 |
| `federated_cb_only [K=128]` | +0.20 | +0.00 | **+0.100** | 2 |
| `federated_cb_only_ema` | +0.40 | +0.00 | **+0.200** | 2 |
| `federated_fedavg_cb_only` | +0.00 | +0.00 | **+0.000** | 2 |
| `federated_fedavg_cb_only [K=128]` | +0.20 | -0.20 | **-0.000** | 2 |
| `federated` | +0.40 | +0.00 | **+0.200** | 2 |
| `federated_shared` | +0.20 | -0.20 | **-0.000** | 2 |
| `federated_fedavg_cb_sharedprior` | +0.00 | -0.40 | **-0.200** | 2 |
| `federated_enc_fedavg` | +0.40 | +0.20 | **+0.300** | 2 |
| `federated_enc_fedprox [mu0.01]` | +0.40 | +0.20 | **+0.300** | 2 |
| `federated_enc_fedproto [lam1_count agg=count]` | +0.20 | -0.20 | **-0.000** | 2 |
| `federated_enc_fedproto [lam1_uniform]` | +0.40 | -0.20 | **+0.100** | 2 |

⚠️ **2 serie**: la colonna «media» è una media di 2 punti, non una stima. Nessun ordinamento fra arm è distinguibile dal rumore a questo n.

## `ucr_split_w2p` — tolleranza 64

| arm | `ucr_001` | `ucr_011` | media | n |
|---|---:|---:|---:|---:|
| `local` | +0.00 | +0.00 | **+0.000** | 2 |
| `centralized` | +0.40 | +0.00 | **+0.200** | 2 |
| `federated_cb_only` | +0.20 | +0.00 | **+0.100** | 2 |
| `federated_cb_only [K=128]` | +0.00 | -0.40 | **-0.200** | 2 |
| `federated_cb_only_ema` | +0.00 | -0.40 | **-0.200** | 2 |
| `federated_fedavg_cb_only` | +0.00 | +0.00 | **+0.000** | 2 |
| `federated_fedavg_cb_only [K=128]` | -0.20 | -0.20 | **-0.200** | 2 |
| `federated` | -0.40 | -0.40 | **-0.400** | 2 |
| `federated_shared` | -0.40 | -0.40 | **-0.400** | 2 |
| `federated_fedavg_cb_sharedprior` | -0.60 | -0.20 | **-0.400** | 2 |
| `federated_enc_fedavg` | -0.40 | -0.40 | **-0.400** | 2 |
| `federated_enc_fedprox [mu0.01]` | +0.20 | +0.00 | **+0.100** | 2 |
| `federated_enc_fedproto [lam1_count agg=count]` | +0.00 | -0.20 | **-0.100** | 2 |
| `federated_enc_fedproto [lam1_uniform]` | -0.20 | -0.40 | **-0.300** | 2 |

⚠️ **2 serie**: la colonna «media» è una media di 2 punti, non una stima. Nessun ordinamento fra arm è distinguibile dal rumore a questo n.

---

## `ucr_split` — tolleranza 64

| arm | `ucr_001` | `ucr_011` | media | n |
|---|---:|---:|---:|---:|
| `local` | 0.80 | 1.00 | **0.900** | 2 |
| `centralized` | 1.00 | 1.00 | **1.000** | 2 |
| `federated_cb_only` | 1.00 | 1.00 | **1.000** | 2 |
| `federated_cb_only [K=128]` | 1.00 | 1.00 | **1.000** | 2 |
| `federated_cb_only_ema` | 1.00 | 0.80 | **0.900** | 2 |
| `federated_fedavg_cb_only` | 0.60 | 1.00 | **0.800** | 2 |
| `federated_fedavg_cb_only [K=128]` | 0.60 | 0.60 | **0.600** | 2 |
| `federated` | 1.00 | 1.00 | **1.000** | 2 |
| `federated_shared` | 1.00 | 0.80 | **0.900** | 2 |
| `federated_fedavg_cb_sharedprior` | 0.80 | 0.40 | **0.600** | 2 |
| `federated_enc_commoninit` | 0.40 | 1.00 | **0.700** | 2 |
| `federated_enc_fedavg` | 0.80 | 1.00 | **0.900** | 2 |
| `federated_enc_fedprox [mu0.01]` | 0.80 | 1.00 | **0.900** | 2 |
| `federated_enc_fedproto [lam1_count agg=count]` | 0.80 | 0.80 | **0.800** | 2 |
| `federated_enc_fedproto [lam1_uniform]` | 0.80 | 1.00 | **0.900** | 2 |

⚠️ **2 serie**: la colonna «media» è una media di 2 punti, non una stima. Nessun ordinamento fra arm è distinguibile dal rumore a questo n.

## `ucr_split_w2p` — tolleranza 64

| arm | `ucr_001` | `ucr_011` | media | n |
|---|---:|---:|---:|---:|
| `local` | 1.00 | 1.00 | **1.000** | 2 |
| `centralized` | 1.00 | 1.00 | **1.000** | 2 |
| `federated_cb_only` | 1.00 | 1.00 | **1.000** | 2 |
| `federated_cb_only [K=128]` | 1.00 | 0.80 | **0.900** | 2 |
| `federated_cb_only_ema` | 1.00 | 0.60 | **0.800** | 2 |
| `federated_fedavg_cb_only` | 0.80 | 1.00 | **0.900** | 2 |
| `federated_fedavg_cb_only [K=128]` | 0.60 | 1.00 | **0.800** | 2 |
| `federated` | 0.40 | 0.80 | **0.600** | 2 |
| `federated_shared` | 0.40 | 1.00 | **0.700** | 2 |
| `federated_fedavg_cb_sharedprior` | 0.00 | 1.00 | **0.500** | 2 |
| `federated_enc_commoninit` | 0.80 | 1.00 | **0.900** | 2 |
| `federated_enc_fedavg` | 0.80 | 0.60 | **0.700** | 2 |
| `federated_enc_fedprox [mu0.01]` | 1.00 | 1.00 | **1.000** | 2 |
| `federated_enc_fedproto [lam1_count agg=count]` | 0.80 | 0.80 | **0.800** | 2 |
| `federated_enc_fedproto [lam1_uniform]` | 0.80 | 1.00 | **0.900** | 2 |

⚠️ **2 serie**: la colonna «media» è una media di 2 punti, non una stima. Nessun ordinamento fra arm è distinguibile dal rumore a questo n.

---

## `ucr_split` — tolleranza 64

| arm | `ucr_001` | `ucr_011` | media | n |
|---|---:|---:|---:|---:|
| `local` | 0.80 | 1.00 | **0.900** | 2 |
| `centralized` | 1.00 | 1.00 | **1.000** | 2 |
| `federated_cb_only` | 1.00 | 1.00 | **1.000** | 2 |
| `federated_cb_only [K=128]` | 1.00 | 1.00 | **1.000** | 2 |
| `federated_cb_only_ema` | 1.00 | 0.80 | **0.900** | 2 |
| `federated_fedavg_cb_only` | 0.60 | 1.00 | **0.800** | 2 |
| `federated_fedavg_cb_only [K=128]` | 0.80 | 0.80 | **0.800** | 2 |
| `federated` | 1.00 | 1.00 | **1.000** | 2 |
| `federated_shared` | 1.00 | 0.80 | **0.900** | 2 |
| `federated_fedavg_cb_sharedprior` | 0.80 | 0.60 | **0.700** | 2 |
| `federated_enc_commoninit` | 0.60 | 1.00 | **0.800** | 2 |
| `federated_enc_fedavg` | 1.00 | 1.00 | **1.000** | 2 |
| `federated_enc_fedprox [mu0.01]` | 0.80 | 1.00 | **0.900** | 2 |
| `federated_enc_fedproto [lam1_count agg=count]` | 1.00 | 1.00 | **1.000** | 2 |
| `federated_enc_fedproto [lam1_uniform]` | 1.00 | 1.00 | **1.000** | 2 |

⚠️ **2 serie**: la colonna «media» è una media di 2 punti, non una stima. Nessun ordinamento fra arm è distinguibile dal rumore a questo n.

## `ucr_split_w2p` — tolleranza 64

| arm | `ucr_001` | `ucr_011` | media | n |
|---|---:|---:|---:|---:|
| `local` | 1.00 | 1.00 | **1.000** | 2 |
| `centralized` | 1.00 | 1.00 | **1.000** | 2 |
| `federated_cb_only` | 1.00 | 1.00 | **1.000** | 2 |
| `federated_cb_only [K=128]` | 1.00 | 1.00 | **1.000** | 2 |
| `federated_cb_only_ema` | 1.00 | 1.00 | **1.000** | 2 |
| `federated_fedavg_cb_only` | 0.80 | 1.00 | **0.900** | 2 |
| `federated_fedavg_cb_only [K=128]` | 0.80 | 1.00 | **0.900** | 2 |
| `federated` | 0.60 | 0.80 | **0.700** | 2 |
| `federated_shared` | 0.60 | 1.00 | **0.800** | 2 |
| `federated_fedavg_cb_sharedprior` | 0.00 | 1.00 | **0.500** | 2 |
| `federated_enc_commoninit` | 1.00 | 1.00 | **1.000** | 2 |
| `federated_enc_fedavg` | 0.80 | 0.80 | **0.800** | 2 |
| `federated_enc_fedprox [mu0.01]` | 1.00 | 1.00 | **1.000** | 2 |
| `federated_enc_fedproto [lam1_count agg=count]` | 1.00 | 0.80 | **0.900** | 2 |
| `federated_enc_fedproto [lam1_uniform]` | 0.80 | 1.00 | **0.900** | 2 |

⚠️ **2 serie**: la colonna «media» è una media di 2 punti, non una stima. Nessun ordinamento fra arm è distinguibile dal rumore a questo n.
