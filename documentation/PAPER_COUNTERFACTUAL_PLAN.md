# Sezione «Counterfactual quality» — piano (2026-08-12) — ⛔ SUPERATO

> **Questo file è storia.** La tesi che sostiene è stata falsificata il 2026-08-12 (dettagli e
> tesi sostitutiva in `PAPER_SEC_COUNTERFACTUAL.md`, §0). Contiene inoltre **due errori di
> fatto** che vanno segnalati perché il ragionamento ci si appoggiava:
>
> 1. **«Lo score percorre solo la lettura» è FALSO.** `score_tokens` chiama `_logits`
>    (`model/prior.py:310`, `:546`, `:774`), che somma `output_bias` e usa il channel embedding:
>    lo score passa **anche** dalle teste locali del prior. L'unica superficie esclusiva del
>    counterfactual è il **decoder** (più la refinement head), che nessuna configurazione federa.
> 2. Di conseguenza cade anche l'attribuzione a due sorgenti (§2, Q3): la testa di bias non è una
>    superficie di sola scrittura.

## 0. Verdetto di fattibilità

**Si può fare, interamente su CPU, senza rilanciare un training e senza toccare le GPU.** I
checkpoint necessari esistono già per tutti e 5 i client su tutte e 10 le serie di sviluppo
(`zn_main/{local,centralized,federated,federated_cb_only}`, `zn_a1`, `zn_a2`), e passano gli
assert di `stage2.counterfactual` (`shared_codebook_per_channel_vq` + `maskgit_3d_pos`).

**Ma NON con `pipeline/cf_eval.py` così com'è.** Due bloccanti, entrambi verificati di persona:

**B1 — `cf_eval.py` non applica la z-normalizzazione per finestra.** Riga 239 passa
`rec.X[s:s+W]` grezzo, e in tutto il file non compare mai `window_normalization` né `zscore`.
Ogni nostro checkpoint è addestrato con z-norm per finestra ⇒ il modello riceverebbe input fuori
distribuzione. È codice anteriore al cutoff del 2026-08-04. Qualunque numero prodotto oggi da
quel file è di un altro modello.

**B2 — la soglia paper-style non è una valuta confrontabile fra arm.** Frazione di timestep di
test sopra la soglia fittata sul train (misurata):

| `ucr_170` | p0 | p1 | p2 | p3 | p4 |
|---|---|---|---|---|---|
| local | **0,978** | 0,582 | 0,304 | 0,423 | 0,347 |
| centralized | 0,033 | 0,055 | 0,057 | 0,139 | 0,080 |
| A2 | **0,000** | 0,023 | 0,032 | 0,091 | 0,144 |

Su `local/p0` il 98% della serie è sopra soglia: nessuna riparazione potrà mai portare il picco
sotto. Su `A2/p0` non c'è niente sopra: è già soddisfatto senza fare nulla. ⇒ **la metrica
`validity` esce dal disegno.** Non è un bug: la soglia è fittata su ~1850 campioni di train e
applicata a un test di ~45500 con distribuzione diversa.

**Decisione: non si estende `cf_eval.py`.** Resta una cassetta di attrezzi da cui importare le
primitive già scritte (`_random_mask_matched`, `_bottom_mask_matched`, il blocco plausibilità,
`smoothness_delta`). Serve un driver nuovo, ~700 righe in quattro file:
`scripts/cf_cluster.py` (generazione), `cf_metrics.py`, `cf_aggregate.py`, `cf_figures.py`.

---

## 1. La domanda della sezione

Il perno è un fatto **strutturale e misurato** sui checkpoint, non un'ipotesi:

| tensori identici fra i 5 client (stage 1, `ucr_170`) | encoder | codebook | **decoder** |
|---|---:|---:|---:|
| **A2** | 160/177 | **6/6** | **0/179** |
| local | 0/177 | 3/6 | 0/179 |

I cinque client federati concordano **bit per bit** su come *leggere* il segnale — encoder,
dizionario, e in stage 2 il corpo del prior (55 chiavi condivise) — e **non condividono nemmeno
un tensore** su come *riscriverlo*. Lo score di anomalia percorre solo la lettura; il
counterfactual percorre anche la scrittura.

> **Tesi: la federazione allinea tutto ciò che entra nello score e nulla di ciò che entra nella
> spiegazione. Cinque client i cui profili di anomalia sono correlati 0,98 producono
> riparazioni che restano divergenti, e la quota dominante di quella divergenza viene dal
> decoder — l'unica superficie che nessuna configurazione federa, e che §Background dichiara
> gratuita perché «non entra nello score».**

È l'unico punto del paper in cui il costo nascosto di una scelta di design diventa misurabile:
la §Background afferma già che tenere il decoder locale «costa nulla in termini di detection»;
questa sezione mostra la fattura in termini di **spiegabilità**.

Tesi subordinata, più rischiosa e più interessante: dove il detector federato sbaglia — il falso
allarme unanime di `ucr_170` — i cinque prior potrebbero **non** concordare su cosa scrivere,
mentre dove ha ragione concordano. Se regge, la spiegazione porta un segnale di rigetto che lo
score, correlato 0,98, ha cancellato.

---

## 2. Disegno

**Il principio che rende tutto appaiato: stessa finestra e stessa `token_mask` per ogni arm e
ogni client; varia solo il modello.** La maschera si calcola una volta per finestra, dalle
etichette, mai dal modello. Questo aggira B2, elimina il confondente «ogni client ha una soglia
diversa quindi una maschera diversa», e tiene `|M|` costante per costruzione.

**Loci, selezionati algoritmicamente e pre-registrati** (nessuna scelta a mano — è la
contromisura al cherry-picking):
- **L-TP**: tutte le finestre che intersecano l'anomalia con ≥10% di contesto normale (139/172
  sulle 10 serie; le 33 sature sono escluse e dichiarate, perché lì il contesto è esso stesso
  anomalo);
- **L-FP**: i 3 picchi più alti dello score di A2 che **non** intersecano l'anomalia, definiti
  su A2 e poi valutati in tutti gli arm — va dichiarato così;
- **L-N**: 30 finestre normali a caso, seed fisso, lontane ≥W da anomalia e picchi.

### Misure quantitative

| | ipotesi | formula | falsificata se |
|---|---|---|---|
| **Q1** accordo score vs spiegazione | federare porta ρ_score→1 e ρ_CF molto meno | ρ_CF = media a coppie di corr fra `d_i = x_cf^i − x` | ρ_CF ≥ 0,95 come ρ_score ⇒ **la tesi cade** |
| **Q2** null intra-modello | i client divergono più di quanto lo stesso modello campioni | MAE fra 8 campioni stocastici dello **stesso** modello | il disaccordo fra client ≈ null ⇒ non divergono, campionano |
| **Q3** scomposizione decoder/token | in A2 la quota dominante è il **decoder** | griglia 5×5 `dec_d(refine_d(tok_t))`, ANOVA a due vie sui residui | quota decoder < 50% |
| **Q4** Δ-plausibilità + specificità | riscrivere il normale lo rende *meno* plausibile | `G = Δ̃(L-N) − Δ̃(L-TP)`, differenza interna allo stesso modello | G ≈ 0 |
| **Q5** distanza dal poolato | il federato sta più vicino al poolato o ai locali | MAE dal CF del poolato, in unità di Q2 | — |

Q2 non è opzionale: senza il null, «i client divergono» non ha scala. Q3 è ammissibile **solo**
sugli arm a codebook bit-identico: fra client `local` la frazione di token identici è ~0,009,
cioè il livello del caso, quindi lì la scomposizione non è definita.

**Il controllo di specificità è obbligatorio e ha tre livelli**, perché la versione ingenua è
dimostrabilmente rotta: sul poolato di `ucr_170` la MAE fra counterfactual e originale vale
0,069 su L-TP, 0,090 su L-N e 0,234 su L-FP — **non separa, e ordina al contrario**. Servono
(i) L-N con `|M|` appaiato, (ii) il gap Q4, (iii) la frazione di finestre normali su cui la
maschera model-driven è vuota.

### Figure

**F1** — `ucr_170`, due pannelli: l'anomalia vera e il falso allarme unanime, con i 5
counterfactual di A2 sovrapposti e quello del poolato in nero. Si vede a occhio che cinque
modelli con encoder, codebook e corpo del prior **bit-identici** riparano in cinque modi diversi.

**F2** — la scala di condivisione a sei pioli (`local` → `cb_only` → `federated` → A1 → A2 →
`centralized`) con ρ_score e ρ_CF affiancate: due curve che divergono. La legenda è la tabella
dei tensori identici, che rende la scala una proprietà **misurata** dei checkpoint.

### Scartate, con motivo

`validity` (B2); DTW (`dtaidistance` non installata, e `cf_eval:43-45` dichiara l'equivalenza a
MAE); riparazione contro `x_clean` (non esiste su UCR, e il surrogato 1-NN nel train è circolare
rispetto alla variabile che separa gli arm); accordo in token-space fra arm a codebook diverso;
channel-targeting (C=1).

---

## 3. Collocazione e costo

Sottosezione dei Results dopo il failure case, oppure sezione breve a sé. Due figure e una
tabella; in un IEEE a due colonne è circa una colonna e mezza.

Costo: tutto su CPU, nessuna GPU. L'ordine di esecuzione va dal più informativo al meno, così
fermarsi a metà lascia comunque una sezione: **(1)** `ucr_170` su local/A2/centralized — dà F1,
Q1, Q2 e la tesi principale; **(2)** le altre tre serie informative (011, 014, 043) — rende Q1
una mediana su 4 serie invece di un aneddoto; **(3)** gli arm intermedi per F2; **(4)** Q3 e Q5.

## 4. Rischi

Il primo è che la sezione diventi **una galleria di figure scelte a mano**. Contromisure: loci
selezionati algoritmicamente, maschera indipendente dal modello, null intra-modello che dà la
scala, e ogni numero riportato come mediana sulle serie — mai media sui client, perché l'unità
resta il cluster.

Il secondo è **confondere il predetto col misurato**. I valori attesi che circolano (ρ_CF
0,55→0,67, quota decoder ~99%, G > 0) vengono da sonde preliminari su una finestra sola: vanno
trattati come ipotesi da falsificare, non come risultati. La tabella sopra dice per ognuna cosa
la ucciderebbe.

Il terzo: `centralized` è **un solo modello salvato cinque volte** (i 5 checkpoint sono
identici). «Il poolato è coerente» è quindi definitorio, non misurato — solo la *distanza* da
esso è un numero. Va scritto in nota, altrimenti è una tautologia mascherata da risultato.

## 5. Cosa non si potrà dire

Che i counterfactual sono «migliori» in un arm che in un altro: senza un segnale pulito di
riferimento non esiste una nozione di correttezza, solo di accordo, plausibilità e specificità.
Che un counterfactual buono implica una detection buona: se la sezione riesce, dimostra il
contrario. E nulla di quantitativo sulle serie a pavimento, dove non c'è un'anomalia da riparare.
