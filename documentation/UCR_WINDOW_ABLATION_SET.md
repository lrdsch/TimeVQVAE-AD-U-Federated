# Set per l'ablazione sulla finestra — 10 serie UCR

**Deciso il 2026-07-30.** Selezione chiusa. Questo file è la motivazione: se qualcuno
(incluso me fra tre settimane) chiede «perché proprio queste dieci», la risposta è qui.

```
ucr_001,ucr_011,ucr_014,ucr_043,ucr_082,ucr_083,ucr_086,ucr_170,ucr_222,ucr_229
```

---

## 1. Il set

| serie | nome UCR | periodo | W a 2P | W(2P)/128 | h di test (2P) | h di test (128) |
|---|---|---:|---:|---:|---:|---:|
| `ucr_011` | DISTORTEDECG1 | 91 | 182 | 1,42× | 83 | 70 |
| `ucr_222` | mit14046longtermecg | 101 | 202 | 1,58× | 88 | 70 |
| `ucr_229` | mit14134longtermecg | 127 | 254 | 1,98× | 99 | 70 |
| `ucr_014` | DISTORTEDECG3 | 165 | 330 | 2,58× | 112 | 70 |
| **`ucr_001`** | **DISTORTED1sddb40** | **204** | **408** | **3,19×** | 125 | 70 |
| `ucr_043` | DISTORTEDMesoplodonDensirostris | 207 | 414 | 3,23× | 126 | 70 |
| `ucr_086` | DISTORTEDsddb49 | 268 | 536 | 4,19× | 143 | 70 |
| `ucr_170` | gaitHunt1 | 337 | 674 | 5,27× | 160 | 70 |
| `ucr_082` | DISTORTEDresperation4 | 446 | 892 | 6,97× | 185 | 70 |
| `ucr_083` | DISTORTEDresperation9 | 485 | 970 | 7,58× | 193 | 70 |

`ucr_001` è **richiesto dall'utente**, non emerso dal criterio; cade comunque a 3,19×,
in mezzo alla scala, quindi non è un'anomalia — vedi §4.

Tutte a 5 client, quantity-skew `[10, 10, 20, 20, 30] %`, split cronologico contiguo,
test e val condivisi dentro il cluster. Identico nei due build (verificato in
`data/raw/*/metadata.json`).

---

## 2. Che cosa deve dimostrare l'ablazione

Una sola cosa: **la finestra W=128 che usiamo in tutto il resto del lavoro è sbagliata,
e l'errore ha un verso e una taglia misurabili.**

Il paper originale aggancia la finestra al periodo, `T = 2 × periodo`; noi usiamo 128 fissa
su tutte le serie ([[window-128-wrong-on-every-ucr-series]]). La differenza non è cosmetica:
a W=128 il modello vede **70 h di test** (token latenti) su ogni serie, indipendentemente
dalla fisica; a 2P ne vede da 83 a 193, cioè da 1,2× a 2,8× più contesto, e — questo è il
punto — un contesto **allineato al ciclo del segnale**. Il gap dichiarato è paper top-1
0,708 contro il nostro 0,415.

Perciò il contrasto da misurare è **appaiato per serie**: stessa serie, stessi client,
stesso protocollo, stessi 8 arm, unica variabile W. Da qui discendono tutti i criteri sotto.

---

## 3. I criteri di selezione, in ordine di forza

### 3.1 Solo dall'intersezione dei due build — vincolo assoluto

`ucr_split` ha 226 cluster, `ucr_split_w2p` ne ha 180, e **non è un sottoinsieme**
([[ucr-split-two-builds-128-and-2p]]). Dentro la coorte `paper` (che applica
`--max-window 1024`) l'intersezione è **167 cluster**. Una serie fuori dall'intersezione
non ha controparte: entrerebbe nella tabella come copertura, mai come risultato appaiato.
Tutte e 10 sono nell'intersezione, ed è verificato che l'intersezione è **byte-identica**
fra i due build (240/240 file, `documentation/UCR_SPLIT_PAIRING.md`).

### 3.2 Solo W(2P) > 128 — il criterio che ha ucciso la selezione precedente

Delle 167 serie comparabili, **55 hanno W(2P) < 128** e 112 hanno W(2P) > 128. Nessuna
esattamente 128.

Questo è il criterio che conta di più, ed è quello che ha bocciato l'idea di «prendere le
prime 10 serie valide»: in quel set **7 su 10 avevano W = 48**. Su quelle serie l'ablazione
gira al contrario — non stai misurando «cosa guadagni allineando la finestra al periodo»,
stai misurando «cosa perdi restringendo la finestra a un terzo». Sono due domande diverse
con due meccanismi diversi (contesto insufficiente vs. contesto disallineato), e mescolarle
in una media rende il numero risultante illeggibile.

Con W(2P) > 128 su tutte e 10, il contrasto ha **un verso solo** e la direzione dell'effetto
è interpretabile.

### 3.3 Spaziatura logaritmica sul rapporto, non sul periodo

La grandezza che governa sia il costo sia la capacità è il **rapporto** W(2P)/128, non W in
assoluto. Le 10 sono log-spaziate su `1,42× … 7,58×`, che è **l'intero range disponibile**
sotto il cap (min 1,42×, max 7,58× sulle 112 candidate).

Questo serve a due cose:

- se l'effetto della finestra è **monotono**, dieci punti log-spaziati lo mostrano come una
  curva, non come due barre;
- se esiste un **ginocchio** (un rapporto oltre il quale il guadagno satura o si inverte),
  dieci punti hanno una possibilità di vederlo. Due punti non ne hanno nessuna.

### 3.4 Cap a W ≤ 1024 — ereditato dalla coorte `paper`, non una scelta nuova

Oltre 1024 il modello non sta in GPU ai default: a W=3028 servono 16,79 GB e restano 2 job
per scheda invece di 7 (§5.7 di `LAUNCH_RUNBOOK.md`). La selezione è stata fatta **dentro**
la coorte `paper`, che il cap lo applica già; non introduce un grado di libertà nuovo di cui
rendere conto. Sul set finale il vincolo è comunque **inattivo** — il massimo è 970 — quindi
la coorte `ucrwin` non ha bisogno di `--max-window`.

### 3.5 Diversità di dominio, per quanto UCR la conceda

Il set copre ECG (4: `ucr_011`, `ucr_014`, `ucr_222`, `ucr_229`), respirazione (2:
`ucr_082`, `ucr_083`), pressione arteriosa / sddb (2: `ucr_001`, `ucr_086`), acustica di
cetacei (1: `ucr_043`), andatura (1: `ucr_170`). Non è una scelta libera: **UCR lega il
dominio al periodo**, quindi la coda ad alto rapporto è quasi tutta respirazione (vedi §5.2).

---

## 4. Il doppione a 3,2× è voluto

`ucr_001` (408) e `ucr_043` (414) stanno praticamente allo stesso rapporto: 3,19× e 3,23×.
Un pignolo direbbe che è un piolo sprecato della scala.

Non lo è, ed è l'unica coppia del set con questa proprietà. Sono **due serie di dominio
completamente diverso — pressione arteriosa e clic di zifio — allo stesso rapporto di
finestra**. La differenza fra i loro due delta è una stima diretta della **varianza
serie-a-serie a rapporto fisso**, cioè esattamente il rumore contro cui va letta la
pendenza della curva. Senza almeno una coppia così, ogni scalino della scala è confondibile
con «è un'altra serie».

Il costo è un piolo su dieci. La resa è un controllo che nessun altro punto del set fornisce.

---

## 5. Quello che questo set NON risolve — da dichiarare nel paper

### 5.1 La frazione di rete federata cambia con W

L'encoder è il **44,9 %** dei parametri di Stage 1 a W=128, ma **40,1 % a W=512** e 59,8 %
a W=46 (§5.7 del runbook, conteggio tensori: 52 → 126). Le due colonne dell'ablazione
**non federano la stessa quota di modello**.

Non invalida il confronto — l'arm è definito per ruolo («l'encoder»), non per conteggio di
parametri, ed è il modo in cui chiunque lo implementerebbe. Ma va scritto: parte del delta
osservato è cambio di finestra e parte è cambio di perimetro federato, e questo set **non
li separa**. Separarli richiederebbe un terzo braccio a perimetro congelato, che non è nel
piano.

### 5.2 La coda alta è monocultura, e le anomalie lì sono di 2 punti

Tutte le 17 candidate fra 6× e 8× sono della famiglia `resperation` — è così che è fatto
l'archivio. Peggio: la maggior parte ha **1 o 2 punti anomali** nel test. `ucr_082` ne ha 2
su 125 250, `ucr_083` ne ha 101.

Con la metrica primaria — **`paper_top1`**, obbligatoria perché VUS-PR non è aggregabile fra
serie UCR ([[vus-pr-not-aggregable-across-ucr]]) — questo va benissimo: il protocollo UCR è
per costruzione «una anomalia iniettata per serie» e top-1 è la metrica ufficiale. Ma
**qualunque numero point-wise secondario (AUPRC, F1) sui due pioli alti è rumore** e non va
riportato per serie. Se serve un secondario, si aggrega e si dichiara.

> 🔴 **MISURATO 2026-08-01, e smentisce il «va benissimo» qui sopra.** Su `ucr_229`
> (anomalia di **11 punti**) **tutti** gli arm fanno `paper_top1` = **0,00**, `centralized`
> compreso — mentre l'AUROC di `centralized` è **0,993**. Un'anomalia corta non rende
> semplicemente rumorosi i secondari: **azzera la metrica primaria** e rende la serie
> incapace di distinguere qualunque coppia di metodi. Vedi §5.4.

Alternativa scartata: `ucr_076` a 7,17× ha 1181 punti anomali e sarebbe più robusto in
point-wise — ma è la stessa famiglia `resperation`, quindi non compra diversità, e sotto
`paper_top1` non compra nulla. Non vale il cambio.

### 5.3 Il costo dei pioli alti è una congettura, non una misura

Vedi §6.

### 5.4 🔴 La difficoltà è CONFUSA col rapporto di finestra (aggiunto 2026-08-01)

§5.2 registrava già le anomalie corte dei due pioli alti. Quello che **non** era stato fatto
è ordinare **tutte e 10** le serie per difficoltà e confrontarla con l'asse che il set vuole
misurare. Finestra di hit = anomalia estesa di ±64 (la tolleranza), cioè la frazione di serie
in cui l'argmax deve cadere:

| serie | rapporto W | anomalia | finestra hit | 1 su | stato |
|---|---:|---:|---:|---:|---|
| `ucr_082` | 6,97× | **2 pt** | 0,104 % | **963** | |
| `ucr_083` | 7,58× | 101 pt | 0,146 % | 686 | |
| `ucr_229` | 1,98× | 11 pt | 0,326 % | 307 | 🔄 |
| `ucr_222` | 1,58× | 501 pt | 0,405 % | 247 | 🔄 |
| `ucr_170` | 5,27× | 111 pt | 0,525 % | 190 | |
| `ucr_086` | 4,19× | 251 pt | 0,632 % | 158 | |
| `ucr_014` | 2,58× | 101 pt | 1,041 % | 96 | |
| `ucr_001` | 3,19× | 621 pt | 1,672 % | 60 | ✅ |
| `ucr_043` | 3,23× | 161 pt | 1,970 % | 51 | |
| `ucr_011` | 1,42× | 301 pt | 2,145 % | **47** | ✅ |

**Range di difficoltà 21×**, mai considerato in §3. E i due assi **non sono ortogonali**:
`ucr_011` ha il rapporto più basso *ed* è la serie più facile; `ucr_082` e `ucr_083` hanno i
rapporti più alti *ed* sono le due più dure. Quindi un Δ attribuito alla finestra può essere
difficoltà, e viceversa — **il set non separa le due cose**.

⚠️ **Le prime due serie girate, `ucr_001` e `ucr_011`, sono l'8ª e la 10ª su 10 per
difficoltà**, cioè le più facili del set. Non era una scelta: sono le prime in ordine di
indice. È il motivo per cui `centralized` sembrava perfetto (top-1 1,00) fino a quando
`ucr_222` e `ucr_229` non hanno restituito **0,00** — vedi [[ucr-easy-series-trap-2026-08-01]].

**Da decidere prima di scrivere**: o la difficoltà entra come covariata, o si scelgono serie
appaiate per rarità e diverse solo nel rapporto di finestra. Non cambiare la tolleranza a metà
corsa: invaliderebbe tutto il misurato.

---

## 6. Costo — e la sua incertezza vera

Modello: `round ≈ (5,78 + 0,483·n_client) · √(W/128)` minuti, 64 round mediani a
convergenza, 5 client, 8 arm, 14 slot (2 GPU × 7).

| | valore |
|---|---|
| job | 10 serie × 2 build × 8 arm = **160** |
| GPU-ore | **2013** |
| wall-clock a 14 slot | **6,0 giorni** |
| job più lungo | **24,1 h** (`ucr_083` a W=970) |

Ripartizione per serie (8 arm, entrambi i build): da 153 h (`ucr_011`) a 262 h (`ucr_083`).

🔴 **`√(W/128)` è l'unica congettura del conto, e non è mai stata misurata.** Il probe del
2026-07-30 è stato fermato prima di toccare un job a finestra grande. Se il costo scalasse
**lineare** in W invece che a radice, il totale sarebbe **3356 GPU-ore = 10,0 giorni** e
`ucr_083` da solo passerebbe da 24 a ~55 h.

Si chiude con **un job solo**: `ucr_001` a W=408, un paio d'ore, e il modello si calibra
prima di impegnare la settimana. Va fatto prima del lancio, non dopo.

Le parti misurate del conto (64 round mediani, 1,64 min per client-round) vengono da
[[launch-cost-reality-2026-07-30]] e restano verificabili in
`evidence/records_pre20260730.tar.gz`.

---

## 7. Prerequisito: il floor non esiste più

Il purge del 2026-07-30 ha svuotato `artifacts/` per intero
([[purge-2026-07-30-total-artifact-wipe]]). **Zero righe di floor esistono su disco, per
nessun dataset.** Il floor su queste 10 serie va rifatto prima di poter leggere qualunque
numero deep: senza di esso non c'è modo di dire se un arm batte `movavg10` a zero parametri
([[floor-baseline-movavg-kills-enc-federation]]).

⚠️ Nota da [[floor-ucr-probe-verdict]]: su `ucr_split` il controllo del gauge è
**irreplicabile** (PCA morta a 0,005–0,013) e il floor è già davanti al deep dappertutto.
Il floor qui serve come **calibrazione**, non come baseline che ci si aspetta di battere.

---

## 8. Comandi

Creazione della coorte (**non ancora eseguita**):

```bash
$PY scripts/cohort.py new ucrwin --datasets ucr_split,ucr_split_w2p \
    --clusters ucr_001,ucr_011,ucr_014,ucr_043,ucr_082,ucr_083,ucr_086,ucr_170,ucr_222,ucr_229

$PY scripts/cohort.py show ucrwin && $PY scripts/cohort.py verify ucrwin
```

Calibrazione del fattore finestra (da fare **prima**). `launch.sh` non ha un flag per
restringere a un singolo cluster — l'asse è pinnato dalla coorte e basta — quindi la
calibrazione è una coorte a una serie sola:

```bash
$PY scripts/cohort.py new ucrwin_cal --datasets ucr_split_w2p --clusters ucr_001
bash scripts/launch.sh --cohort ucrwin_cal --arms federated_cb_only --tag ucrwin_cal
```

Serve a misurare i minuti/round a W=408 e confrontarli con gli 8,2 misurati a W=128:
il rapporto osservato dice se `√(W/128)` (che predice 1,79×) regge o se lo scaling è lineare
(3,19×). Basta lasciarlo girare qualche round — non serve la convergenza.

Lancio pieno:

```bash
bash scripts/launch.sh --cohort ucrwin --arms paper --tag ucrwin_v1 --dry   # prima
bash scripts/launch.sh --cohort ucrwin --arms paper --tag ucrwin_v1         # poi
```

Il fingerprint della coorte va annotato in `LAUNCH_RUNBOOK.md` §5.0 appena creata: righe con
`cohort_fingerprint` diverso non sono confrontabili
([[cohort-launch-system-2026-07-29]]).

---

## 9. Provenienza dei numeri di questo file

| numero | fonte |
|---|---|
| finestre a 2P, appartenenza ai build | `cohorts/paper.json` |
| periodi, train/test/anom | `documentation/ucr_split_series_periods.csv` |
| 167 comparabili, 112 con W>128, 55 con W<128 | ricalcolato da `cohorts/paper.json` il 2026-07-30 |
| intersezione byte-identica 240/240 | `documentation/UCR_SPLIT_PAIRING.md` |
| 5 client, shares `[10,10,20,20,30]` | `data/raw/ucr_split*/metadata.json` |
| 44,9 % / 40,1 % encoder | `LAUNCH_RUNBOOK.md` §5.7 |
| 64 round, 1,64 min/client-round | `evidence/records_pre20260730.tar.gz` |
| paper top-1 0,708 vs 0,415 | `RESEARCH_LEDGER.md` |
