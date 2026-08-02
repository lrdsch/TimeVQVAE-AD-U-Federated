# L'unico dato di varianza-da-seed raccolto (2026-07-31)

Il run a seed 1 e' stato fermato su richiesta dopo 1 job su 16: si resta a **un solo seed (0)**.
Sopravvive questo numero, che vale la pena non perdere.

| arm | build | seed 0 | seed 1 | delta |
|---|---|---:|---:|---:|
| `centralized` | `ucr_split_w2p` (W=408) | 0.6180 | 0.6157 | **-0.0023** |

VUS-PR, media sui 5 client di `ucr_001`. Coorti `ucr001` (fp c604c97f8001290c) e `ucr001s1`
(fp a5e457b6ce033f53, poi cancellata): identiche tranne il seed.

**A cosa serve.** Fissa la scala del rumore da run-a-run su un arm che non federa nulla:
**0,4%**. Quindi l'inversione di segno del contrasto suff-stat vs FedAvg fra `ucr_001`
(+0,174) e `ucr_011` (-0,130) e' **due ordini di grandezza** sopra questo rumore -- su
`centralized`. NON e' noto se gli arm federati siano altrettanto stabili: su `ucr_001` i
gemelli FedAvg a stage-1 identico divergevano di 5,5x, su `ucr_011` di 1,0x. Quella domanda
resta aperta e con un solo seed non e' rispondibile.

Il JSON accanto e' il risultato completo del job seed 1. Tutto il resto dell'albero
`ucr001_s1` / `ucr011_s1` e' stato cancellato perche' parziale (log pieni di `rc=143`, che
era il SIGTERM dell'interruzione, non guasti).
