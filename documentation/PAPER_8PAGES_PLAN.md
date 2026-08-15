# Ridurre `paper.tex` da 13 a 8 pagine — piano di taglio (2026-08-14)

Tutti i pesi qui sotto sono **misurati**, non stimati: ogni blocco è stato messo in un
`\vbox{\hsize=\columnwidth ...}` con il preambolo vero del paper e ne è stata letta l'altezza.
Unità = **punti-colonna (pt-col)**. Una pagina IEEE conference = 2 colonne × 672 pt = **1344 pt-col**.
Una figura `figure*` (larghezza piena) costa **il doppio** della sua altezza.

Riprodurre le misure: `/tmp/.../scratchpad/pagecalc/M_sec4.tex`, `M_blk.tex`, `M_floats.tex`.

---

## 0. Punto di partenza misurato

| voce | pt-col | pagine |
|---|---:|---:|
| Blocco titolo + 7 autori | 652 | 0,49 |
| **Testo** (tutte le sezioni, senza float) | **12 291** | **9,15** |
| **Float** (8 fra figure e tabelle) | **3 058** | **2,28** |
| Bibliografia (28 voci) | 950 | 0,71 |
| Perdita di impaginazione (colla dei float, ultima pagina non piena) | ~521 | 0,39 |
| **TOTALE** | **17 472** | **13,00** |

Peso dei singoli float:

| float | altezza | costo reale | pagine |
|---|---:|---:|---:|
| `fig:overview` (`figure*`) | 547 pt | **1 094** | **0,81** |
| `fig:framework` (1 col) | 475 pt | 475 | 0,35 |
| `tab:space` (1 col) — di cui **167 pt di sola nota a piè di tabella** | 467 pt | 467 | 0,35 |
| `fig:cf` (`figure*`) | 227 pt | 455 | 0,34 |
| `tab:main` | 186 pt | 186 | 0,14 |
| `tab:cf` | 144 pt | 144 | 0,11 |
| `tab:c50` | 136 pt | 136 | 0,10 |
| `tab:ablation` | 101 pt | 101 | 0,08 |

Peso del testo per sezione (le sei righe in grassetto sono metà del paper):

| sezione | righe | pt-col | pagine | parole |
|---|---|---:|---:|---:|
| Abstract | 63–84 | 239 | 0,18 | 227 |
| I. Introduction | 85–148 | 827 | 0,62 | 531 |
| II. Related Work | 149–204 | 682 | 0,51 | 467 |
| III. Background | 205–297 | 1 128 | 0,84 | 821 |
| IV. Federation Design | 298–423 | 1 531 | 1,14 | 761 |
| V. Experimental Protocol | 424–503 | 1 394 | 1,04 | 994 |
| VI-A Which series separate | 509–517 | 175 | 0,13 | 99 |
| VI-B Reference points | 518–528 | 403 | 0,30 | 291 |
| VI-C Main configuration | 529–569 | 403 | 0,30 | 281 |
| **VI-D Where the improvement comes from** | 570–645 | **1 255** | **0,93** | 977 |
| VI-E Stage-2 synchronization | 646–668 | 437 | 0,33 | 288 |
| VI-F Failure case + fusion | 669–679 | 533 | 0,40 | 393 |
| **VI-G Confirmation (n=48)** | 680–770 | **1 051** | **0,78** | 818 |
| **VI-H Counterfactual quality** | 771–858 | **691** | **0,51** | 508 |
| VI-I Limitations | 859–873 | 197 | 0,15 | 154 |
| VII. Conclusion | 874–908 | 533 | 0,40 | 404 |
| VIII. Future Work | 909–948 | 605 | 0,45 | 444 |
| Data + Ack | 949–955 | 120 | 0,09 | 58 |

---

## 1. Decisione 0 — da confermare prima di tagliare

**Le 8 pagine includono la bibliografia?** Molte call IEEE danno 8 pagine + 1 sola per i
riferimenti. Se i riferimenti sono esclusi il budget cresce di **950 pt-col = 0,71 pagine**, e
`fig:cf` (455) + `tab:cf` (144) possono restare senza altre contorsioni. Il piano sotto assume
il caso peggiore: **8 pagine tutto compreso**.

---

## 2. L'aritmetica

Budget: 8 pagine = 10 752 pt-col. Tengo 400 pt-col di riserva per la perdita di impaginazione
⇒ **budget di contenuto = 10 350 pt-col**.

```
10 350  budget
 −  360  titolo + autori compattati (Lotto 0)
 −  950  bibliografia
 − 1 203  float superstiti (Lotto 1)
 ────────
 = 7 837  pt-col disponibili per il TESTO   (oggi: 12 291)
```

**Il testo deve scendere del 36 %**, da ~8 800 a ~4 900 parole. I target per sezione al §5
sommano a 6 790 pt-col, cioè **~1 000 pt-col sotto il budget**: è margine voluto, e al §7 c'è
la lista di cosa rimettere dentro se avanza.

---

## 3. Lotto 0 — guadagno gratuito, zero contenuto perso (−292 pt-col, −0,22 pag)

Il blocco autori attuale impagina 7 `\IEEEauthorblockN` come 3+2+2 righe e occupa **326 pt a
larghezza piena** (l'abstract comincia a y=382 invece che a y≈236). Cinque autori su sette sono
UPM e due sono Federico II: **due blocchi per affiliazione invece di sette**.

Variante misurata (blocco che finisce a y≈184, risparmio **146 pt pieni = 292 pt-col**):

```latex
\author{\IEEEauthorblockN{Leonardo Schiavo, Donato Cerciello, \'Angel Panizo-Lledot,
Javier Huertas Tato, David Camacho}
\IEEEauthorblockA{\textit{Departamento de Sistemas Inform\'aticos},
\textit{Universidad Polit\'ecnica de Madrid}, Madrid, Spain \quad
\texttt{leonardo.schiavo@upm.es}}
\IEEEauthorblockN{Stefano Izzo, Fabio Giampaolo}
\IEEEauthorblockA{\textit{Department of Mathematics and Applications ``R. Caccioppoli''},
\textit{University of Naples Federico II}, Naples, Italy}
}
```

Restano tutti e sette i nomi, entrambe le affiliazioni e un contatto. È la forma raggruppata per
affiliazione, ammessa dal template.

---

## 4. Lotto 1 — float: −1 855 pt-col (−1,38 pagine)

Ordinato per rapporto risparmio/danno.

| # | azione | risparmio | perché si può |
|---|---|---:|---|
| 1 | **eliminare `fig:overview`** | **−1 094** | È la voce singola più cara del paper (0,81 pag). La didascalia stessa dichiara che **non sostiene nessuna affermazione**: «*Every configuration already localizes this series, so nothing here is a claim about the configurations*». Lo split è già descritto a parole in §V-B, il counterfactual in §VI-H. |
| 2 | **eliminare `fig:cf`** | −455 | Illustra un risultato che è **nullo** ed è già in `tab:cf`. Se §VI-H si riduce a un paragrafo (§5), la figura non ha più una sezione che la giustifichi. |
| 3 | `tab:ablation` → una frase | −101 | Sono 6 numeri (9/1/0, +0,069; 4/6/0, −0,001; 8/2/0, +0,136) già scritti per esteso nelle righe 575–577. La tabella è pura duplicazione. |
| 4 | potare la nota di `tab:space` (riga 413) e fondere le righe (j)–(m) | −120 | **167 pt della tabella sono la sola nota** a piè di pagina: tre note che rispiegano cose dette in §IV-C. Tenere `^a` (fusione = solo inferenza) e mezza riga di `^c` (BN locale ⇒ baseline è (j)); il resto va nel testo o sparisce. |
| 5 | ricomporre `fig_framework` più bassa (475 → ~390 pt) | −85 | Il PDF è generato da `scripts/`; 475 pt su una colonna sono 0,71 di colonna. Comprimere l'interlinea verticale del diagramma, non i contenuti. |

**Float dopo il lotto: 1 203 pt-col** (framework 390 + tab:space 347 + tab:main 186 +
tab:c50 136 + tab:cf 144 → 1 203, avendo tolto overview, cf e ablation).

---

## 5. Lotto 2 — le ridondanze globali

Queste sono la ragione per cui il paper è a 13 pagine. Ogni tesi centrale è enunciata da tre a
sette volte. **Regola: ogni affermazione vive per esteso in un solo posto; altrove è una
subordinata senza numeri.**

| # | tesi | dove è ripetuta | dove deve restare |
|---|---|---|---|
| 1 | «il prior condiviso serve solo dopo l'allineamento; κ=1,00 vs 0,07» | Abstract, Intro ¶3, Intro ¶5, §VI-D (579–595), Conclusion (878–887) | **§VI-D per esteso.** Abstract e Conclusion: una frase **senza** rifare i numeri. |
| 2 | «l'endpoint primario non separa niente, nemmeno il centralized» | Abstract, Intro ¶4, §VI-G ×3 (748–760), Conclusion, Future Work §3 | **§VI-G, una volta.** Abstract: mezza frase. |
| 3 | «un seme per cella» | §V-C (477), §VI intro (507), §VI-F (674), §VI-H (855), Limitations, Future Work ¶1 | **§V-C** + **Limitations**. |
| 4 | «il decoder non entra nello score ⇒ tenerlo locale è gratis» | §III-B (278), §IV apertura (301), §VI-H (774) | **§IV apertura.** |
| 5 | «le config diagnostiche non sono federation-legal, sono bound non metodi» | §IV-C (376), nota `tab:space`, §VI-F (674) | **§IV-C**, una frase. |
| 6 | «riportiamo l'endpoint fissato in anticipo e non rivendichiamo la lettura forte» | Intro 118–119, §VI-G 758–760, Conclusion 903–904 | **§VI-G.** |
| 7 | «la fusione è una riparazione, non il rivelatore principale» | Contributo 2, §IV-D, §VI-F ×2 (676, 678) | **§VI-F.** |
| 8 | «4 serie discriminanti, il sign test non arriva a 0,05» | §V-D (502), §VI-A (516), Limitations | **§VI-A.** |
| 9 | «κ misurato ≠ etichetta appiccicata all'arm» | §VI-D (579–580, 586–588), Conclusion (878–882) | Una volta, in §VI-D. |

**Due regole di stile che da sole valgono il 10-15 %:**

- **numeri in cifre, non in lettere**: «thirty-two of forty-eight» → «32/48», «nine of ten» →
  «9/10». Compare ~40 volte.
- **cancellare le frasi-annuncio**: «*The two diagnostic configurations locate the
  responsibility.*», «*The reason is visible in the tie counts rather than in the effect.*»,
  «*One limitation affects this comparison.*», «*The same holds for the codebook, with one
  difference worth stating.*». Annunciano la frase dopo senza aggiungere niente: ce ne sono
  ~15, ~12 pt-col l'una.
- **togliere gli incisi fra `---`**: il paper ne ha 23. La maggior parte sono qualificazioni
  difensive che il Lotto 2 rende ridondanti.

---

## 6. Lotto 3 — taglio sezione per sezione

Colonna «Δ» = risparmio misurato dei blocchi indicati.

### Abstract 239 → 180 (−59)
- Righe 75–77: la frase sulla fusione → subordinata finale di una riga.
- Riga 74: togliere l'inciso «*a metric on which pooled training… does not separate either*»
  (resta implicito in «matches pooled training»).

### I. Introduction 827 → 500 (−327)
| blocco | righe | Δ |
|---|---|---:|
| **eliminare l'`enumerate` dei contributi** | 129–147 | **−148** |
| ridurre il paragrafo di numeri a 2 frasi | 110–119 | −107 |
| comprimere il ¶1 di motivazione a 2 frasi | 87–94 | −45 |
| potare il ¶ sull'allineamento | 121–127 | −27 |

L'`enumerate` (193 pt) **ripete alla lettera** i ¶3 e ¶4 che lo precedono. Sostituirlo con una
frase: «*We contribute a coordinated federation strategy for tokenizer-based detectors, and a
characterization of a weight-space failure mode with a function-space alternative.*»

### II. Related Work 682 → 380 (−302)
- **Eliminare le 4 `\subsection`** e scrivere §II come 3 paragrafi: ogni titolo di
  sottosezione costa ~17 pt fra testo e colla → **−68**.
- La sottosezione VQ-VAE (179–187, 127 pt) **duplica §III**: ridurla a 2 frasi → −87.
- La coda auto-riassuntiva (197–200, 67 pt: «*That dependency is what we study… we measure its
  effect on detection, compare…*») è l'introduzione detta una terza volta → **−67**.
- Il ¶ FL (151–165): comprimere l'elenco SCAFFOLD/FedOpt/FedMA/FedPer/FedRep/LG-FedAvg in una
  frase con le citazioni raggruppate → −60.
- TSAD (167–177): −20.

### III. Background 1128 → 620 (−508)
| blocco | righe | Δ | nota |
|---|---|---:|---|
| ¶ kernel frequency-independent | 245 | −57 | serve solo al counterfactual |
| `eq:stage1` → in linea a parole | 228–235 | −38 | è la loss VQ-VAE standard |
| ¶ multi-rate κ + `eq:afinal` | 263–274 | −97 | tenere 3 frasi; `eq:afinal` in linea |
| ¶ «score dal solo prior» | 275–279 | −58 | vedi ridondanza 4 |
| ¶ config ereditata (T=2P, z-norm, n\_fft) | 283–289 | −72 | spostare in §V-C, una frase |
| ¶ coupling («a token is a pointer») | 291–296 | −31 | **è la tesi del paper: tenerlo, 2 frasi** |
| prosa residua di III-A e III-B | | −155 | |

### IV. Federation Design 1531 → 860 (−671)
| blocco | righe | Δ |
|---|---|---:|
| `eq:omega` → in linea `$\omega_j=n_j/\sum_i n_i$` | 303–307 | −47 |
| ¶ che annuncia la figura → nella didascalia | 309–312 | −57 |
| giustificazione anti-FedBN → 1 frase | 339–343 | −46 |
| ¶ motivazione del FedAvg sul codebook | 345 | −65 |
| ¶ «fissare il codebook non congela il tokenizer» | 356 | −43 |
| Prior federation ¶2–¶4 | 363–367 | −80 |
| ¶ config diagnostiche | 376 | −60 |
| ¶ (j)–(m) → resta nella nota di `tab:space` | 378 | −65 |
| Score fusion → 3 frasi | 420–423 | −91 |
| prosa residua | | −117 |

### V. Experimental Protocol 1394 → 700 (−694)
| blocco | righe | Δ | nota |
|---|---|---:|---|
| **¶ esclusione AUROC/soglia/VUS** | 485–498 | **−169** | tenere *una* frase: «*AUROC saturates and threshold-based scores are not comparable across arms; we therefore report accuracy@64 and AUPRC.*» I numeri 0,947–0,988 e 0,000–0,978 non servono |
| ¶ budget di passi che vincola (9 %, 15–25 %) | 475 | −122 | 1 frase: i baseline sono semmai svantaggiati |
| domini del dev set (4 ECG, 2 respirazione…) | 431–432 | −75 | «*selected by period and domain, not by difficulty*» |
| ¶ floor | 479 | −48 | |
| §V-B split federato | 435–471 | −107 | |
| ¶ unità di analisi + serie non discriminanti | 500–502 | −60 | vedi ridondanza 8 |
| prosa residua | | −113 | |

### VI. Results 5233 → 2 585 (−2 648)

| sotto-sezione | ora | target | come |
|---|---:|---:|---|
| intro §VI | 89 | 45 | |
| A. Which series separate | 175 | 90 | 3 frasi; l'elenco delle serie a soffitto/pavimento è in `tab:main` |
| B. Reference points | 403 | 180 | ¶ floor (527, 189 pt) → 2 frasi, via il caveat z-norm; ¶ max_j (523–525) → 1 frase |
| C. Main configuration | 403 | 210 | **via il ¶ 534–536 (117 pt): sono i numeri di `tab:main` riletti a voce**; ¶ budget del baseline locale (540) → 2 frasi |
| **D. Where the improvement comes from** | **1 255** | **600** | vedi sotto |
| **E. Stage-2 synchronization** | **437** | **110** | **è un nullo su una manopola che non è un contributo.** Tenere solo il nullo fuori campione (657–667) in 3 righe; togliere le 4 misure sul dev set (649–655, −225) |
| F. Failure case + fusion | 533 | 340 | ¶ diagnostiche (674) → 70 pt; ¶ 678 fuso nel 676 |
| **G. Confirmation** | **1 051** | **560** | il ¶ forense sui due run falliti (drift 1,39/1,58/3,03/3,60/8,89 a 3,5k…9,1k passi) → 3 righe: «*two runs aborted on non-finite codebook statistics; the failure tracks half-precision exposure, not difficulty; both series were at floor for the reference arms*». Poi ridondanza 2: la tesi «l'endpoint non ha risoluzione» compare 3 volte dentro la sotto-sezione, tenerne una |
| **H. Counterfactual quality** | **691** | **280** | ridurre a **un paragrafo + `tab:cf`**: che cosa si misura (Δ, G, maschera dalle label), i due risultati (Δ>0 in 15/15, G>0 in 14/15; specificità +0,114 vs local, +0,007 vs centralized) e il caveat «nullo, non equivalenza». Via il dettaglio del protocollo (8 finestre, stride costante, 10 % di contesto, round-trip 38–65 %) e via `fig:cf` |
| I. Limitations | 197 | 170 | assorbe il confondente hardware (623) |

**Dettaglio di §VI-D (1 255 → 600), la sotto-sezione più cara del paper:**

| blocco | righe | ora | target |
|---|---|---:|---:|
| ¶1–¶3 l'interazione (numeri di `tab:ablation`) | 573–577 | 350 | 200 (assorbe la tabella eliminata) |
| **¶ κ misurato** | 579–588 | 200 | 150 — **è il risultato di punta, non si tocca la sostanza** |
| ¶ meccanismi intermedi (κ=0,77 → 1,00; (m) a 0,04) | 590–595 | 130 | 90 |
| ¶ «(j)–(m) tengono il prior locale ⇒ la domanda è limitata» | 597–599 | **175** | **0** — è un'ammissione di ciò che non è stato misurato, basta una subordinata in Limitations |
| ¶ nullo FedProx/FedProto/init | 601 | 165 | 80 — tenere «nessuno cambia niente, il controllo a sola inizializzazione è indistinguibile»; via le due qualificazioni finali |
| ¶¶ regola del codebook (c)/(d) e (h)/(i) | 603–621 | 304 | 80 — **il paper stesso dice «*we do not present that choice as a contribution*»**: due frasi, nullo in aggregato con varianza per serie grande |
| ¶ confondente hardware | 623 | 69 | 0 — va in Limitations |

### VII. Conclusion 533 → 300 (−233)
Oggi ripete per esteso: κ=1,00 vs 0,07 (già §VI-D), il nullo FedProx/FedProto (già §VI-D), 32/48
e 37/48 (già §VI-G), la fusione (già §VI-F). Riscrivere in **un paragrafo di tesi + un paragrafo
di bilancio**, senza rinumerare nulla.

### VIII. Future Work 605 → 130 (−475)
**Eliminare la sezione.** Tre frasi in coda alla Conclusion: (i) più semi, (ii) meccanismi di
allineamento parziali (è il seguito naturale di κ), (iii) client eterogenei, dove BN condivisa e
codebook unico sono attesi invertirsi. Il punto sull'endpoint è già in §VI-G; il paragrafo sul
guard di finitezza è già in §VI-G.

---

## 7. Riserve e ripristini

**Se dopo la ricompilazione manca ancora spazio** (in quest'ordine):
1. bibliografia: 6 riferimenti puramente nominali (SCAFFOLD, FedOpt, FedPer, FedRep, LG-FedAvg,
   Orchestra) valgono **~204 pt-col**. Da fare solo se le frasi che li citano sono già cadute.
2. `tab:c50` (136) → le tre righe come testo: si perde molto in leggibilità, ultima risorsa.

**Se invece avanza spazio** (in quest'ordine, sono ~900 pt-col di margine previsto):
1. una `fig_overview` **ricomposta a una colonna**, solo i pannelli (a)+(b) → ~300 pt-col;
2. `fig:cf` → 455;
3. `tab:ablation` → 101.

---

## 8. Cosa non si tocca

- κ=1,00 su 10/10 vs mediana 0,07, **con codebook bit-identico da entrambi i lati**: è
  l'evidenza che regge il titolo.
- L'interazione $d_A - d_F$ e il fatto che il set di tensori condivisi sia identico nei due rami.
- L'ordine di riporto in §VI-G: **prima l'endpoint primario, che non conferma**; poi il
  secondario. Invertirlo è HARKing.
- `ucr_170` e la fusione: è il contributo 2.
- Le formulazioni difendibili già fissate negli audit (`publishability-audit-2026-08-09`): «A2 >
  local» dipende dal baseline, «centralized > local» regge, l'allineamento è una **precondizione**
  e non un effetto.
- Il confondente hardware fra il ramo allineato e quello non allineato: si accorcia, non si toglie.

---

## 9. Esecuzione e verifica

Un lotto alla volta, ricompilando dopo ciascuno:

```bash
bash scripts/build_paper.sh
grep -o 'Output written.*([0-9]* pages' documentation/.build/paper.log | grep -o '[0-9]* pages'
```

Traiettoria attesa:

| dopo | pagine attese |
|---|---:|
| partenza | 13 |
| Lotto 0 (autori) | 13 |
| Lotto 1 (float) | ~11 |
| Lotto 2 (ridondanze globali) | ~10 |
| Lotto 3 su III+IV+V | ~9 |
| Lotto 3 su VI+VII+VIII | **~8 con ~0,6 pag di margine** |

Alla fine controllare anche `scatole over/underfull` (oggi 10) e che nessun `\ref` resti appeso a
un float eliminato. I rinvii da sistemare sono esattamente cinque:

| riga | rinvio | dove va a finire |
|---|---|---|
| 467 | `Figure~\ref{fig:overview}` (§V-B) | cade con la frase |
| 586 | `Table~\ref{tab:ablation}` (§VI-D) | «the interaction above» |
| 823 | `Table~\ref{tab:cf}` (§VI-H) | resta, `tab:cf` sopravvive |
| 848 | `Figure~\ref{fig:cf}` (§VI-H) | cade con il paragrafo |
| 852 | `Figure~\ref{fig:overview}(c)` (§VI-H) | cade con il paragrafo |

## 10. Rischi

- **Il taglio di `fig:overview` è il singolo punto più discutibile**: è la figura che spiega lo
  split e il counterfactual a colpo d'occhio, e vale da sola 0,81 pagine. Se si vuole tenere una
  figura d'insieme, la versione a una colonna con due pannelli costa 300 invece di 1 094.
- **Ridurre §VI-H a un paragrafo** rende il counterfactual un risultato di contorno. È coerente
  con quello che è (un nullo su 5 serie e un seme), ma toglie l'unica parte che usa il decoder.
- Con §VIII eliminata bisogna verificare che la call non richieda una sezione di lavori futuri.
