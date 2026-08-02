# `ucr_split` vs `ucr_split_w2p` — accoppiamento

Generato da `scripts/ucr_split_pair_manifest.py`. Stesso protocollo, unica
differenza la finestra a cui sono agganciate le soglie di eleggibilità.

| | cluster |
|---|---|
| `ucr_split` (W=128) | 226 |
| `ucr_split_w2p` (W=2·periodo) | 180 |
| **comparabili (intersezione)** | **177** |
| solo `ucr_split` | 49 |
| solo `ucr_split_w2p` | 3 |

⚠️ Un test appaiato gira **solo** sull'intersezione. Le serie fuori non hanno
controparte: vanno riportate come copertura, mai come risultato.

Intersezione byte-identica: **sì** (240/240 file verificati).

## Solo in `ucr_split` — scartate dal build 2P (49)

| serie | W=2P | perché fuori dal 2P |
|---|---|---|
| `ucr_015` | — | smallest client slice 500 < 1916 (W=958) |
| `ucr_016` | — | smallest client slice 500 < 664 (W=332) |
| `ucr_017` | — | smallest client slice 500 < 664 (W=332) |
| `ucr_019` | — | smallest client slice 500 < 876 (W=438) |
| `ucr_020` | — | smallest client slice 500 < 876 (W=438) |
| `ucr_021` | — | smallest client slice 500 < 876 (W=438) |
| `ucr_022` | — | smallest client slice 400 < 860 (W=430) |
| `ucr_023` | — | smallest client slice 500 < 868 (W=434) |
| `ucr_024` | — | smallest client slice 320 < 612 (W=306) |
| `ucr_025` | — | smallest client slice 280 < 608 (W=304) |
| `ucr_030` | — | smallest client slice 300 < 732 (W=366) |
| `ucr_031` | — | smallest client slice 270 < 732 (W=366) |
| `ucr_033` | — | smallest client slice 400 < 700 (W=350) |
| `ucr_036` | — | smallest client slice 420 < 700 (W=350) |
| `ucr_048` | — | smallest client slice 350 < 388 (W=194) |
| `ucr_049` | — | smallest client slice 350 < 388 (W=194) |
| `ucr_050` | — | smallest client slice 350 < 388 (W=194) |
| `ucr_051` | — | smallest client slice 350 < 388 (W=194) |
| `ucr_052` | — | smallest client slice 350 < 388 (W=194) |
| `ucr_054` | — | smallest client slice 270 < 456 (W=228) |
| `ucr_074` | — | smallest client slice 400 < 824 (W=412) |
| `ucr_075` | — | smallest client slice 400 < 824 (W=412) |
| `ucr_096` | — | smallest client slice 500 < 668 (W=334) |
| `ucr_097` | — | smallest client slice 500 < 876 (W=438) |
| `ucr_103` | — | smallest client slice 350 < 392 (W=196) |
| `ucr_123` | — | smallest client slice 500 < 1900 (W=950) |
| `ucr_124` | — | smallest client slice 500 < 664 (W=332) |
| `ucr_125` | — | smallest client slice 500 < 664 (W=332) |
| `ucr_127` | — | smallest client slice 500 < 876 (W=438) |
| `ucr_128` | — | smallest client slice 500 < 876 (W=438) |
| `ucr_129` | — | smallest client slice 500 < 876 (W=438) |
| `ucr_130` | — | smallest client slice 400 < 876 (W=438) |
| `ucr_131` | — | smallest client slice 500 < 876 (W=438) |
| `ucr_132` | — | smallest client slice 320 < 620 (W=310) |
| `ucr_133` | — | smallest client slice 280 < 620 (W=310) |
| `ucr_138` | — | smallest client slice 300 < 728 (W=364) |
| `ucr_139` | — | smallest client slice 270 < 728 (W=364) |
| `ucr_141` | — | smallest client slice 400 < 708 (W=354) |
| `ucr_144` | — | smallest client slice 420 < 700 (W=350) |
| `ucr_156` | — | smallest client slice 350 < 392 (W=196) |
| `ucr_157` | — | smallest client slice 350 < 392 (W=196) |
| `ucr_158` | — | smallest client slice 350 < 392 (W=196) |
| `ucr_159` | — | smallest client slice 350 < 392 (W=196) |
| `ucr_160` | — | smallest client slice 350 < 392 (W=196) |
| `ucr_162` | — | smallest client slice 270 < 452 (W=226) |
| `ucr_182` | — | smallest client slice 400 < 820 (W=410) |
| `ucr_183` | — | smallest client slice 400 < 732 (W=366) |
| `ucr_213` | — | smallest client slice 3321 < 3512 (W=1756) |
| `ucr_214` | — | smallest client slice 3421 < 3552 (W=1776) |

## Solo in `ucr_split_w2p` — scartate dal build W=128 (3)

| serie | W=2P | perché fuori dal 128 |
|---|---|---|
| `ucr_068` | 50 | smallest client slice 129 < 256 |
| `ucr_176` | 52 | smallest client slice 129 < 256 |
| `ucr_248` | 54 | smallest client slice 199 < 256 |
