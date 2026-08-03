# Tabella finale — 9 serie UCR, TimeVQVAE-AD federato

Unita' = **serie** (media delle 2 finestre, media dei 5 client). Soglie = 2 SD della differenza appaiata, da 3 replicati su hardware identico.

⚠️ **3 serie su 9 non discriminano** e contribuiscono 0 a ogni Δ: `ucr_229` (pavimento (tutti 0,00)), `ucr_083` (soffitto (local gia' 1,00)), `ucr_086` (punteggi dipendenti dallo shard di train (difetto di scoring, dimostrato)). Le colonne «senza cieche» usano le 6 restanti.

## top-1 — ogni arm contro `local`

| arm | Δ vs local | n | soglia | concordi | esito | per serie |
|---|---:|---:|---:|---:|---|---|
| `centralized` | **+0.189** | 9 | ±0.097 | 5/5 (+4 pari) | 🟢 sopra il rumore | +0.40 +0.10 +0.70 +0.30 +0.00 +0.20 +0.00 +0.00 +0.00 |
| `federated_cb_only` | **-0.089** | 9 | ±0.097 | 5/8 (+1 pari) | dentro il rumore | +0.20 -0.10 -0.20 -0.20 -0.60 +0.30 -0.30 +0.10 +0.00 |
| `federated_cb_only_ema` | **-0.078** | 9 | ±0.097 | 4/7 (+2 pari) | dentro il rumore | +0.10 -0.20 -0.10 +0.00 -0.60 +0.20 -0.40 +0.30 +0.00 |
| `federated_fedavg_cb_only` | **-0.022** | 9 | ±0.097 | 3/6 (+3 pari) | dentro il rumore | -0.10 +0.00 -0.20 +0.10 +0.00 +0.20 -0.30 +0.10 +0.00 |
| `federated` | **-0.111** | 9 | ±0.097 | 5/7 (+2 pari) | 🟢 sopra il rumore | -0.10 -0.20 +0.00 -0.30 -0.60 +0.20 -0.30 +0.30 +0.00 |
| `federated_shared` | **-0.144** | 9 | ±0.097 | 5/7 (+2 pari) | 🟢 sopra il rumore | -0.20 -0.30 +0.00 -0.20 -0.70 +0.10 -0.30 +0.30 +0.00 |
| `federated_fedavg_cb_sharedprior` | **-0.189** | 9 | ±0.097 | 6/7 (+2 pari) | 🟢 sopra il rumore | -0.40 -0.30 -0.20 -0.10 -0.50 +0.00 -0.30 +0.10 +0.00 |

## VUS-PR — ogni arm contro `local`

| arm | Δ vs local | n | soglia | concordi | esito | per serie |
|---|---:|---:|---:|---:|---|---|
| `centralized` | **+0.168** | 9 | ±0.072 | 8/9 | 🟢 sopra il rumore | +0.18 +0.52 +0.46 +0.20 +0.01 +0.10 -0.01 +0.04 +0.02 |
| `federated_cb_only` | **-0.023** | 9 | ±0.072 | 5/9 | dentro il rumore | +0.02 -0.03 -0.04 -0.03 -0.12 +0.06 -0.10 +0.03 +0.00 |
| `federated_cb_only_ema` | **-0.042** | 9 | ±0.072 | 5/9 | dentro il rumore | -0.04 -0.10 -0.03 +0.00 -0.11 +0.02 -0.15 +0.03 +0.00 |
| `federated_fedavg_cb_only` | **-0.009** | 9 | ±0.072 | 3/9 | dentro il rumore | -0.09 +0.09 -0.06 +0.05 +0.00 +0.05 -0.13 +0.01 +0.00 |
| `federated` | **-0.069** | 9 | ±0.072 | 6/9 | dentro il rumore | -0.14 -0.12 -0.06 -0.03 -0.13 +0.00 -0.17 +0.02 +0.01 |
| `federated_shared` | **-0.069** | 9 | ±0.072 | 5/9 | dentro il rumore | -0.18 -0.10 -0.13 +0.04 -0.13 +0.02 -0.15 +0.01 +0.00 |
| `federated_fedavg_cb_sharedprior` | **-0.065** | 9 | ±0.072 | 6/9 | dentro il rumore | -0.21 +0.04 -0.16 +0.03 -0.09 -0.05 -0.14 -0.01 +0.00 |

## Codebook: K=128 contro K=64

| arm | metrica | Δ | n | soglia | concordi | esito |
|---|---|---:|---:|---:|---:|---|
| `federated_cb_only` | top-1 | +0.011 | 9 | ±0.097 | 3/6 | dentro il rumore |
| `federated_cb_only` | VUS-PR | -0.010 | 9 | ±0.072 | 5/9 | dentro il rumore |
| `federated_fedavg_cb_only` | top-1 | -0.011 | 9 | ±0.097 | 3/6 | dentro il rumore |
| `federated_fedavg_cb_only` | VUS-PR | +0.011 | 9 | ±0.072 | 3/9 | dentro il rumore |
| `local` | top-1 | +0.033 | 9 | ±0.097 | 4/6 | dentro il rumore |
| `local` | VUS-PR | +0.018 | 9 | ±0.072 | 5/9 | dentro il rumore |
| `centralized` | top-1 | +0.000 | 9 | ±0.097 | 2/3 | dentro il rumore |
| `centralized` | VUS-PR | +0.012 | 9 | ±0.072 | 6/9 | dentro il rumore |

## Federazione dell'encoder — ogni arm contro `local`

Il floor `movavg10` batte gia' questo blocco (mediana 0,507 a zero parametri): il confronto che conta e' contro `local`, non fra gli arm.

| arm | metrica | Δ vs local | n | soglia | concordi | esito |
|---|---|---:|---:|---:|---:|---|
| `federated_enc_fedavg` | top-1 | -0.089 | 9 | ±0.097 | 4/5 | dentro il rumore |
| `federated_enc_fedavg` | VUS-PR | -0.039 | 9 | ±0.072 | 7/9 | dentro il rumore |
| `federated_enc_fedprox_mu0.01` | top-1 | -0.122 | 9 | ±0.097 | 4/7 | 🟢 sopra |
| `federated_enc_fedprox_mu0.01` | VUS-PR | -0.058 | 9 | ±0.072 | 8/9 | dentro il rumore |
| `federated_enc_fedproto_lam1_uniform` | top-1 | -0.122 | 9 | ±0.097 | 5/8 | 🟢 sopra |
| `federated_enc_fedproto_lam1_uniform` | VUS-PR | -0.021 | 9 | ±0.072 | 5/9 | dentro il rumore |
| `federated_enc_commoninit` | top-1 | -0.067 | 9 | ±0.097 | 4/6 | dentro il rumore |
| `federated_enc_commoninit` | VUS-PR | -0.040 | 9 | ±0.072 | 5/9 | dentro il rumore |

## FedProto: aggregazione `count` contro `uniform`

| metrica | Δ (count − uniform) | n | soglia | concordi | esito |
|---|---:|---:|---:|---:|---|
| top-1 | -0.022 | 9 | ±0.097 | 4/6 | dentro il rumore |
| VUS-PR | -0.020 | 9 | ±0.072 | 5/9 | dentro il rumore |

## Finestra: W=2P contro W=128

| arm | metrica | Δ | n | soglia | concordi | esito |
|---|---|---:|---:|---:|---:|---|
| `local` | top-1 | -0.156 | 9 | ±0.097 | 3/4 | 🟢 sopra |
| `local` | VUS-PR | -0.012 | 9 | ±0.072 | 5/9 | dentro il rumore |
| `centralized` | top-1 | -0.044 | 9 | ±0.097 | 1/2 | dentro il rumore |
| `centralized` | VUS-PR | +0.061 | 9 | ±0.072 | 7/9 | dentro il rumore |
| `federated_cb_only` | top-1 | -0.156 | 9 | ±0.097 | 5/7 | 🟢 sopra |
| `federated_cb_only` | VUS-PR | -0.017 | 9 | ±0.072 | 3/9 | dentro il rumore |
| `federated_cb_only_ema` | top-1 | -0.267 | 9 | ±0.097 | 6/6 | 🟢 sopra |
| `federated_cb_only_ema` | VUS-PR | -0.025 | 9 | ±0.072 | 6/9 | dentro il rumore |
| `federated_fedavg_cb_only` | top-1 | -0.022 | 9 | ±0.097 | 2/6 | dentro il rumore |
| `federated_fedavg_cb_only` | VUS-PR | +0.011 | 9 | ±0.072 | 6/9 | dentro il rumore |
| `federated` | top-1 | -0.289 | 9 | ±0.097 | 6/7 | 🟢 sopra |
| `federated` | VUS-PR | -0.008 | 9 | ±0.072 | 5/9 | dentro il rumore |
| `federated_shared` | top-1 | -0.222 | 9 | ±0.097 | 6/7 | 🟢 sopra |
| `federated_shared` | VUS-PR | -0.025 | 9 | ±0.072 | 6/9 | dentro il rumore |
| `federated_fedavg_cb_sharedprior` | top-1 | -0.222 | 9 | ±0.097 | 5/7 | 🟢 sopra |
| `federated_fedavg_cb_sharedprior` | VUS-PR | -0.055 | 9 | ±0.072 | 7/9 | dentro il rumore |

## Contro il numero pubblicato (TimeVQVAE-AD: paper_top1 = 0,708)

- **tutte le serie** (n=9): 0.600, IC95% [0.315, 0.885]
- **senza le cieche** (n=6): 0.700, IC95% [0.384, 1.016]

`centralized` e' l'analogo strutturale del loro setting: il test e' condiviso e identico fra i 5 client, quindi addestra sull'intero train della serie e valuta sull'intero test. Periodi verificati identici ai loro su tutte e 250 le serie, e ±64 contro ±100 non cambia nulla (0 celle su 960).
