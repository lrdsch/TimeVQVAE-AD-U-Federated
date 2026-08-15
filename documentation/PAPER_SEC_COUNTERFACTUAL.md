# §Counterfactual quality — progetto (riscritto 2026-08-12)

## 0. La tesi precedente è CADUTA — verbale

La prima versione di questo documento sosteneva una **dissociazione**: la federazione allinea
ciò che entra nello score e non ciò che entra nella spiegazione, quindi i controfattuali di
cinque client con tokenizer bit-identico sarebbero rimasti divergenti.

Aveva scritta la sua condizione di morte (§3, C1: «falsificata se ρ_CF cresce come ρ_score»), e
l'ha incontrata. Sonda su `ucr_170`, 8 finestre di test, decodificando **gli stessi token** con
i cinque decoder di A2 — che non condividono **nemmeno un tensore** (0/179):

| | correlazione fra le forme d'onda | MAE a coppie | MAE di ricostruzione vs input |
|---|---:|---:|---:|
| A2, stessi token, 5 decoder | **0,9847** | 0,1338 | 0,1355 |

Correlati **0,985**, cioè esattamente il livello a cui sono correlati gli score. Differenza nei
parametri, stessa funzione: è il fenomeno del gauge che il paper già cita
(`entezari2022permutation`, `ainsworth2023gitrebasin`), applicato al decoder. **La dissociazione
non esiste.**

Resta vero, ma è un'altra cosa e molto più modesta, che il disaccordo fra due decoder (0,1338) è
grande quanto l'errore di ricostruzione di ciascuno (0,1355) — rapporto 0,99.

⚠️ La riga `local` di quella sonda (corr −0,15) **non è un risultato**: forzava gli stessi
indici su vocabolari diversi, cioè dava al decoder del client 3 simboli che per lui non
denotano niente. Cella non definita, non un dato.

---

## 1. La tesi nuova

> **I controfattuali del modello federato sono di qualità paragonabile a quelli di `local` e
> `centralized` su tutti e tre gli assi misurabili — accordo, plausibilità, specificità. La
> federazione non costa qualità della spiegazione; il suo vantaggio sta nella detection.**

È un enunciato di **equivalenza**, quindi va trattato come tale: non «non abbiamo trovato
differenze», ma «le differenze misurate sono piccole rispetto a una scala dichiarata». La scala
è il null stocastico intra-modello (§2), ed è per questo che è la misura centrale e non un
accessorio.

Non serve che la sezione porti un risultato positivo: serve che **non nasconda un costo**. Il
paper afferma in §Background che tenere il decoder locale «non entra nello score»; questa
sezione verifica che non entri nemmeno nella spiegazione.

## 2. Le tre misure, e perché proprio queste

Tutte appaiate: **stessa finestra, stessa maschera, varia solo il modello**. La maschera viene
dalle **etichette**, mai dal modello — così è identica fra arm e fra client e `|M|` è costante
per costruzione. Questo aggira il bloccante misurato: la soglia paper-style non è una valuta
confrontabile fra arm (frazione di test sopra soglia da 0,000 a 0,978 a seconda della cella),
quindi `validity` esce dal disegno e la selezione model-driven dei token con essa.

**ACCORDO** — MAE a coppie fra i controfattuali di client diversi, **in unità del null
stocastico intra-modello**: `K` campioni dello stesso client, stessa finestra, stessa maschera.
Rapporto ≈ 1 significa che due client federati differiscono fra loro quanto un singolo client
differisce da sé stesso ricampionando, cioè che la differenza non è attribuibile al modello.
Senza questo denominatore «i client divergono di 0,12» non ha scala.

**PLAUSIBILITÀ** — `Δ = [NLL(x) − NLL(x_cf)] / NLL(x)` sulle sole posizioni riscritte, sotto il
prior del client stesso, con `x_cf` **ri-codificato** dall'encoder (round-trip decode→encode).
Il round-trip non è un dettaglio: valutare i token campionati sarebbe **circolare**, perché
vengono per costruzione dal prior che poi li giudica. Si misura ciò che l'encoder legge davvero
nella forma d'onda prodotta. Il valore circolare è comunque riportato come diagnostica
(`delta_direct`), per separare «il prior campiona male» da «il round-trip perde la riparazione».
Le NLL assolute non sono confrontabili fra arm: il rapporto interno lo è.

**SPECIFICITÀ** — `G = Δ(anomalo) − Δ(normale)` con `|M|` appaiato. Senza il braccio normale,
«il controfattuale sembra normale» è vero per costruzione: se il generatore liscia tutto,
`G ≈ 0`. La versione ingenua di questo controllo è **dimostrabilmente rotta**: sul poolato di
`ucr_170` la distanza controfattuale-originale vale 0,069 sull'anomalia, 0,090 sul normale e
0,234 sul falso allarme — non separa, e ordina al contrario.

### Scartate, con motivo
`validity` (la soglia non è confrontabile — misurato); DTW (`dtaidistance` assente, e `cf_eval`
dichiara l'equivalenza a MAE); confronto con un segnale pulito (non esiste su UCR, e il
surrogato 1-NN nel train è circolare rispetto alla variabile che separa gli arm); accordo in
spazio-token fra arm con codebook diversi (non definito).

## 3. Vincoli che impediscono la galleria

1. Maschera dalle etichette, mai dal modello.
2. Finestre scelte da un algoritmo con seed fisso: le anomale campionate a passo costante lungo
   l'evento, con ≥10% di contesto normale; le normali a caso, con `|M|` pari alla mediana delle
   anomale, in un blocco contiguo a offset casuale.
3. Ogni divergenza riportata in unità del null intra-modello.
4. Le figure mostrano **tutti e 5** i client, mai un sottoinsieme.

## 4. Il controllo degenere da dichiarare

`centralized` è **un solo modello salvato cinque volte** (verificato: max |Δ| = 0,000e+00 su
tutti i tensori). Il suo accordo fra client è 1 per definizione e **non è una misura**: con lo
stesso seed i cinque output sono identici e il rapporto esce 0. Va marcato «—» in tabella, non
riportato come il valore migliore. Restano misurate, e valide, la sua plausibilità e la sua
specificità.

## 5. Cosa non si dirà

Che i controfattuali sono «migliori» in un arm: senza un segnale pulito di riferimento non
esiste una nozione di correttezza, solo accordo, plausibilità e specificità. Che un buon
controfattuale implichi una buona detection — su `ucr_170` la detection di A2 crolla, e la
sezione non deve suggerire il contrario. E niente di quantitativo sulle serie a pavimento, dove
non c'è un'anomalia da riparare.

## 6. Codice e artefatti

- `scripts/cf_quality.py` — driver, solo CPU. **Non** estende `pipeline/cf_eval.py`: quel file
  passa finestre grezze (`rec.X[s:s+W]`, riga 239) e non applica mai la z-norm per finestra, ed
  è quindi inservibile con i checkpoint post-2026-08-04. Il driver nuovo prende le finestre dal
  `SlidingWindowDataset`, che la z-norm la applica (`data.py:384`).
- `scripts/cf_figure.py` — figura.
- `evidence/cf_quality/` — JSON delle metriche, array per le figure, log.
