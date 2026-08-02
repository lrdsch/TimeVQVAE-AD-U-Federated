# `ucr_011` — paper top-1 / top-3 / top-5, per client

Metrica ufficiale UCR. **1 = anomalia trovata, 0 = mancata.** Top-1 = l'argmax dello score; top-K = i K massimi locali distanti almeno `tol`; hit se una predizione cade entro `tol` dal segmento anomalo.

Letti dove la pipeline li ha scritti — `report.json` per client (deep) e i record del floor — non ricalcolati. Entrambi da `detect._paper_metrics`.


## `ucr_split` — W=128, tolleranza=64

| arm | k | p0 | p1 | p2 | p3 | p4 | media |
|---|---|---:|---:|---:|---:|---:|---:|
| `local` | top-1 | 0 | **1** | **1** | **1** | **1** | **0.80** |
| `local` | top-3 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `local` | top-5 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `centralized` | top-1 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `centralized` | top-3 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `centralized` | top-5 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `federated_cb_only` | top-1 | 0 | **1** | **1** | 0 | **1** | **0.60** |
| `federated_cb_only` | top-3 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `federated_cb_only` | top-5 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `federated_cb_only_ema` | top-1 | **1** | 0 | **1** | **1** | **1** | **0.80** |
| `federated_cb_only_ema` | top-3 | **1** | 0 | **1** | **1** | **1** | **0.80** |
| `federated_cb_only_ema` | top-5 | **1** | 0 | **1** | **1** | **1** | **0.80** |
| `federated_fedavg_cb_only` | top-1 | **1** | **1** | **1** | 0 | **1** | **0.80** |
| `federated_fedavg_cb_only` | top-3 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `federated_fedavg_cb_only` | top-5 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `federated` | top-1 | **1** | 0 | **1** | **1** | **1** | **0.80** |
| `federated` | top-3 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `federated` | top-5 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `federated_shared` | top-1 | 0 | **1** | **1** | **1** | 0 | **0.60** |
| `federated_shared` | top-3 | 0 | **1** | **1** | **1** | **1** | **0.80** |
| `federated_shared` | top-5 | 0 | **1** | **1** | **1** | **1** | **0.80** |
| `federated_fedavg_cb_sharedprior` | top-1 | 0 | **1** | **1** | 0 | 0 | **0.40** |
| `federated_fedavg_cb_sharedprior` | top-3 | 0 | **1** | **1** | 0 | 0 | **0.40** |
| `federated_fedavg_cb_sharedprior` | top-5 | **1** | **1** | **1** | 0 | 0 | **0.60** |
| **floor migliore** (`ar_p32_lam0.0001`) | top-1 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| ↑ | top-3 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| ↑ | top-5 | **1** | **1** | **1** | **1** | **1** | **1.00** |

*floor: mostrato l'arm migliore su top-1 fra 21. Arm del floor che trovano l'anomalia in top-5 su almeno un client: **21/21**.*

## `ucr_split_w2p` — W=182, tolleranza=64

| arm | k | p0 | p1 | p2 | p3 | p4 | media |
|---|---|---:|---:|---:|---:|---:|---:|
| `local` | top-1 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `local` | top-3 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `local` | top-5 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `centralized` | top-1 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `centralized` | top-3 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `centralized` | top-5 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `federated_cb_only` | top-1 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `federated_cb_only` | top-3 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `federated_cb_only` | top-5 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `federated_cb_only_ema` | top-1 | 0 | **1** | **1** | 0 | **1** | **0.60** |
| `federated_cb_only_ema` | top-3 | 0 | **1** | **1** | 0 | **1** | **0.60** |
| `federated_cb_only_ema` | top-5 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `federated_fedavg_cb_only` | top-1 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `federated_fedavg_cb_only` | top-3 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `federated_fedavg_cb_only` | top-5 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `federated` | top-1 | 0 | **1** | **1** | 0 | **1** | **0.60** |
| `federated` | top-3 | **1** | **1** | **1** | 0 | **1** | **0.80** |
| `federated` | top-5 | **1** | **1** | **1** | 0 | **1** | **0.80** |
| `federated_shared` | top-1 | **1** | **1** | **1** | 0 | 0 | **0.60** |
| `federated_shared` | top-3 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `federated_shared` | top-5 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `federated_fedavg_cb_sharedprior` | top-1 | 0 | **1** | **1** | **1** | **1** | **0.80** |
| `federated_fedavg_cb_sharedprior` | top-3 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| `federated_fedavg_cb_sharedprior` | top-5 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| **floor migliore** (`ar_p32_lam0.0001__cent`) | top-1 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| ↑ | top-3 | **1** | **1** | **1** | **1** | **1** | **1.00** |
| ↑ | top-5 | **1** | **1** | **1** | **1** | **1** | **1.00** |

*floor: mostrato l'arm migliore su top-1 fra 21. Arm del floor che trovano l'anomalia in top-5 su almeno un client: **21/21**.*
