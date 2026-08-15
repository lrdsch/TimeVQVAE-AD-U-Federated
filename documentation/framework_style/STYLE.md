# Stile della figura-framework (MLISE) — specifica misurata

Estratta da `images/ClusteringFramework.png` del pacchetto
`TimeSeriesConferencePaper___MLISE_Conference.zip`. **Tutti i numeri qui sono misurati
dal PNG**, non stimati a occhio. Serve a rifare una figura nello stesso linguaggio visivo.

## Come è stata prodotta l'originale

| | |
|---|---|
| strumento | **Inkscape** (`Software: www.inkscape.org` nei metadati PNG) |
| export | PNG **9930×14020 @ 1200 dpi** = 8,28 × 11,68 pollici, 6,36 MB |
| inclusione | `\includegraphics[width=\columnwidth]{...}` in `IEEEtran` conference |
| conseguenza | in colonna IEEE (3,5″) la figura è scalata al **42%** ⇒ **2837 dpi effettivi** |

⚠️ È sovra-risoluta di circa 9×. A 600 dpi alla larghezza di resa il file passa da 6,36 MB a
1,74 MB senza differenza visibile in stampa; a 300 dpi sta sotto il mezzo mega.
Le altre figure dello stesso paper sono matplotlib 3.10.8 esportate anch'esse a 1200 dpi.

## 1. Palette — riempimenti puri + trasparenza

Questa è **la firma dello stile**, ed è la cosa che va copiata per prima. I riempimenti non
sono colori pastello: sono **colori primari saturi con l'opacità abbassata**. Solo lo 0,5%
dei pixel è pienamente opaco; il 63% ha alpha intermedia.

| ruolo | fill | alpha | resa su bianco | area |
|---|---|---|---|---|
| sfondo stage «Finetuning» | `#FFFF00` | **19%** | `#FFFFCD` | 18,1% |
| sfondo stage «Pretraining» | `#FF8029` | **19%** | `#FFE6D5` | 11,7% |
| pannello in evidenza (*Instance level*) | `#71CEDC` | **59%** | `#ABE2EA` | 7,1% |
| box decoder (*Intra-view*) | `#71E2D5` | **59%** | `#ABEEE6` | 0,6% |
| trapezio encoder (*Mamba*) | `#FF9A9A` | **49%** | `#FFCDCD` | 1,5% |
| box ingresso (*Augmentation*) | `#AEC6FF` | **49%** | `#D7E3FF` | 1,1% |
| box vista (*View k*) | `#90EF90` | **49%** | `#C8F7C8` | 1,0% |
| interno delle ellissi | `#F9F9FA` | 99% | `#F9F9FA` | 4,6% |
| tratti e testo | `#000000` | 100% | — | 13,2% |

**Tre soli livelli di opacità**, ed è ciò che tiene insieme la tavolozza:

- **19%** → i due grandi sfondi di stage (devono stare *sotto* tutto senza competere)
- **49%** → i box dei componenti
- **59%** → i due pannelli che il testo commenta

Accenti dentro le illustrazioni (punti, barre, frecce piccole), pieni e saturi:
`#4567A9` blu · `#D17920` arancio · `#2B7B4F` verde · `#7B3A99` viola · `#B0302F` rosso.

## 2. Geometria

| proprietà | valore misurato | regola |
|---|---|---|
| spessore tratto | **38 px @1200 dpi** su box da 410 px *e* su box da 130 px *e* sugli sfondi | **uniforme**, non proporzionale: 2,28 pt in assoluto ⇒ **≈1 pt alla dimensione di resa** |
| raggio d'angolo | 144 px su box alto 410 px | **35% dell'altezza del box** — molto generoso, quasi a pastiglia |
| colore tratto | nero pieno su ogni forma | |
| sfondi di stage | riempimento senza contorno | |

Forme in uso: rettangolo arrotondato (componenti), **trapezio rastremato verso il basso**
(encoder), **ellisse** come contenitore delle illustrazioni concettuali, rettangolo arrotondato
gigante e non contornato (stage). Le frecce sono spezzate/curve tracciate a mano con punta
triangolare **barbata** (concava sul retro), tipica dei marker Inkscape `Arrow2`.

## 3. Tipografia

Tutto il testo è in **grassetto corsivo con grazie**, di taglio molto pesante: terminali a
goccia, `k` con gamba ricurva, `&` svolazzante, lettere larghe. È il carattere che dà alla
figura il suo aspetto riconoscibile.

L'identificazione più probabile è **Bookman Demi Italic** (URW Bookman / Bookman Old Style Bold
Italic), che è il font PostScript standard con esattamente quelle caratteristiche ed è la scelta
naturale in Inkscape su Linux. ⚠️ **Non verificato**: su questa macchina sono installati solo
DejaVu e Liberation, quindi non ho potuto rendere un confronto diretto. Verificalo aprendo l'SVG
originale, o installando `fonts-urw-base35` e confrontando.

In LaTeX il corrispondente è `\usepackage{bookman}` + `\textbf{\textit{...}}`.

Le etichette matematiche (`z_1`, `u`, `\mathcal{L}_{cross}`) sono nello stesso peso, corsivo, con
il pedice più piccolo: in TikZ si ottiene con `$\boldsymbol{\mathcal{L}}_{\text{cross}}$`.

## 4. Composizione

La lettura è **dall'alto verso il basso**, con due colonne logiche affiancate:

1. banda d'ingresso (serie grezze → box di augmentation)
2. ventaglio verso N viste parallele, con `⋯` a marcare la ripetizione
3. una riga di encoder identici, uno per vista
4. le rappresentazioni `z_1 … z_N`, da cui partono le frecce verso i due stage
5. **due regioni di sfondo affiancate**, una per stage, ciascuna etichettata in basso a lettere
   grandi (*Pretraining Stage*, *Finetuning Stage*)
6. dentro ogni regione, i termini di loss come nodi terminali (`ℒ_cross`, `ℒ_rec`, …) che
   convergono su una loss aggregata (`ℒ_pre`, `ℒ_final`)

Le ellissi «Before / After» sono il trucco espositivo ricorrente: dentro un contenitore, lo
stato prima e dopo separati da una linea tratteggiata orizzontale.

## 5. Applicato: il framework di TimeVQVAE-AD

`timevqvae_framework.py` → `timevqvae_framework.pdf`. Ricompone la Fig. 1 (inferenza) e la
Fig. 4 (stage 1 / stage 2) di `2311.12550v5.pdf` in questa grammatica: tronco comune
(`x` → STFT → encoder → VQ → token `s`), due regioni di stage affiancate con le rispettive
loss, banda d'inferenza separata da una tratteggiata e biforcata sotto il prior nei due rami
del paper — rilevazione e spiegazione.

**3,50 × 4,93 pollici, 41,7 kB, vettoriale verificato** (nessun `/Subtype /Image`, 1736
operatori di path) contro i 6,36 MB del PNG originale. Entra a `\columnwidth` senza scalare.

⚠️ Il font è DejaVu Serif bold italic, non Bookman: qui non è installato. Cambia la costante
`FONT` in cima allo script quando lo compili dove Bookman c'è. Vale per entrambe le figure.

`a2_framework.py` → `a2_framework.pdf` è la variante federata (arm A2), **3,50 × 4,66 pollici,
40,9 kB, vettoriale verificato**. Stessa grammatica, con in più il ciclo client↔server dentro
ogni regione di stage.

⚠️ La figura **parte dal client e dalle sue finestre**, non dalla serie intera divisa in fette:
come i dati siano stati distribuiti fra i client è **assetto sperimentale**, non modello, e sta
in §Protocol del paper.

⚠️ **I conteggi non sono più stampati in figura** (163, 44, 207, 209, 2, 55, κ=1,000, «Prop. 1»):
un numero dentro un diagramma si legge come proprietà del metodo, e quei numeri hanno ciascuno
una provenienza e un campione che vanno detti nel testo. Restano qui sotto come base della
figura, e vanno riportati in §Method o §Protocol, non nella didascalia:

| gruppo di tensori | n | max \|Δ\| fra i 5 client di `ucr_187` | |
|---|---|---|---|
| encoder (pesi) | 163 | `0.000e+00` | condiviso |
| encoder (BN running stats) | 44 | `0.000e+00` | condiviso, **poolato** |
| quantizer (codebook + EMA) | 6 | `0.000e+00` | condiviso |
| decoder_2d | 209 | `2.03e+05` | **locale** |
| refinement | 2 | `6.56e-01` | **locale** |

163 + 44 = 207, cioè i «207 shared tensors» del banner di run. Il decoder è escluso **per
costruzione**: `_encoder_shared_keys` filtra su `k.startswith("encoder.")`
(`pipeline/federated.py:797`). Lo stage 2 condivide 55 chiavi del corpo del prior e tiene locali
`channel_embedding` e `output_bias` (`pipeline/federated.py:2052`).

## 5-bis. Le due figure che entrano nel paper

| file in `documentation/` | prodotto da | dove sta in `paper.tex` |
|---|---|---|
| `fig_framework.pdf` | **copia** di `framework_style/a2_framework.pdf` | `\ref{fig:framework}`, §Federation Design, una colonna |
| `fig_overview.pdf` | `scripts/fig_overview.py` | `\ref{fig:overview}`, §Federated split, `figure*` |

⚠️ `fig_framework.pdf` è una copia, non un link: se rigeneri `a2_framework.py` **ricopialo**
(`cp framework_style/a2_framework.pdf fig_framework.pdf`), altrimenti il paper compila la
versione vecchia senza dirlo.

`fig_overview.py` non esegue nessun modello: legge `data/raw/ucr_split_w2p`, i `scores.npz`
già su disco di `artifacts/runs/zn_a2/…` e il dump counterfactual
`evidence/cf_quality/arrays/A2_ucr_011.npz`. Tre fasce sullo stesso asse dei tempi — lo split
in 5 fette, i 5 profili di punteggio, lo zoom sull'anomalia col counterfactual del client 1 —
in **coordinate della serie originale**, che sono quelle in cui l'archivio annota l'anomalia
(il test parte a 10 000 su `ucr_011`, l'evento è 11 800–12 100).

Il counterfactual torna nelle unità del segnale con l'affine della finestra stessa
(`mean + std * (x + d)`): la z-norm per finestra annulla esattamente lo scaler per-entità,
verificato a `max|Δ| = 0,013` contro il grezzo. Le regioni riscritte sono **due finestre**,
perché l'evento è più lungo di `W = 182` e ogni finestra tiene almeno il 10 % di contesto
normale — quindi non tassellano l'evento, e la didascalia lo dice.

## 6. Da qui

- `framework_template.svg` — apribile in Inkscape, contiene una primitiva già stilata per ogni
  forma (box, trapezio, ellisse, sfondo di stage, freccia) con i colori e le alpha qui sopra.
  Si duplica e si compone.
- `framework_style.tex` — le stesse convenzioni come stili TikZ, se preferisci una figura
  vettoriale dentro il documento invece di un PNG da 6 MB. ⚠️ Non compilato: su questa macchina
  non c'è `pdflatex`.
