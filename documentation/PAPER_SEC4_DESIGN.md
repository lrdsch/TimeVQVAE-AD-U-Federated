# Progetto della §IV — *Federation design space*

> Documento di progettazione, non testo del paper. Scritto 2026-08-11.
> Contesto: §I e §II sono scritte, §III (Background) è in `paper.tex` dal 2026-08-11.
> Questa sezione è la cerniera fra §III (il detector) e §VI (i risultati).

---

## 1. Che lavoro deve fare la sezione

Tre compiti, in ordine di importanza:

1. **Rendere ogni arm di §VI una cella di uno spazio**, non un metodo con un nome proprio.
   Se il lettore arriva in §VI e legge «A2 batte cb_only», non ha imparato niente. Se legge
   «spostarsi da *codebook federato / prior locale* a *codebook federato / body del prior
   condiviso* cambia il segno», ha imparato una regola.
2. **Separare la SUPERFICIE dal PRIMITIVO di aggregazione.** È il punto non ovvio del lavoro:
   la stessa superficie (il codebook) aggregata in due modi diversi dà risultati che differiscono
   di un ordine di grandezza. Se le due cose restano confuse in un'unica lista di metodi, il
   risultato di §VI sembra rumore fra sigle.
3. **Fissare cosa è federation-legal**, così che in §VI si possano usare riferimenti superiori
   (tokenizer poolato, centralized) senza che il lettore li scambi per arm proponibili.

Corollario: §IV **non contiene numeri**. Nessun risultato, nessun Δ, nessuna anticipazione di
quale cella vince. Le uniche frasi valutative ammesse sono strutturali («questa combinazione è
rifiutata dal sistema perché mal definita»).

---

## 2. Vincoli — cosa la sezione NON può dire

Ritrattazioni già chiuse nel ledger, da non far rientrare dalla finestra:

- ❌ **Non chiamare `cb_only`/`cb_ema` «Federated Analytics».** Non è difendibile (demarcazione
  Elkordy et al.). La formulazione ammessa: *il tokenizer è federato da un primitivo esattamente
  additivo e non-gradiente, invece che per media dei pesi*.
- ❌ **Non rivendicare novità matematica sul merge per statistiche sufficienti.** È la M-step di
  Lloyd su statistiche pooled = Dhillon & Modha 1999, ed è k-FED `\cite{dennis2021kfed}` nella sua
  forma one-shot. Va citato come *primitivo adottato*, non proposto.
- ❌ **Non presentare `--fed-enc-scope neck/partial` come parte dello spazio esplorato.** È
  pre-registrato e **mai girato**. O esce dalla tabella, o entra con una riga esplicita «declared,
  not run» in §VIII (limitazioni). Raccomandazione: fuori dalla tabella, una riga in limitazioni.
- ❌ **Non dire «l'aggregazione pesata per conteggi diluisce la competenza sul pattern raro»** —
  falsificata (`union_recluster` peggiora, `fedavg_cb_only` migliora).
- ⚠️ **`ctfp` (tokenizer poolato) è illegale**: usa il pool dei dati. Può comparire solo come
  *oracolo/skyline* etichettato come tale, mai come arm.

---

## 3. Struttura proposta

Budget totale: **~1,25 colonne di testo + 1 tabella**. Cinque sottosezioni brevi.

### IV-A — Cosa può attraversare la rete (le superfici)  [~15 righe]

Elenca le superfici del detector di §III e, per ciascuna, se e come può essere condivisa:

| superficie | opzioni | flag reale |
|---|---|---|
| encoder $E$ | locale · federato (intero) | `--arms federated_enc_*` |
| statistiche BN | affini condivisi / stat locali · tutto locale (FedBN) · stat pooled per legge della varianza totale | `--fed-enc-bn buffers_local\|fedbn\|shared` |
| codebook $\mathcal C$ | locale · merge per statistiche sufficienti · union+recluster · media dei pesi | `--fed-enc-cb local\|suffstat\|union_recluster\|fedavg` |
| body del prior | locale · condiviso | `--fed-enc-prior local\|partial\|shared` |
| teste del prior (`channel_embedding`, `output_bias`) | locali in `partial` | `LOCAL_PRIOR_PREFIXES` |
| decoder | **sempre locale** | — |

Il decoder chiude il paragrafo con l'argomento già piazzato in §III: non viene valutato a
inferenza, quindi tenerlo locale non costa detection. È l'unica scelta dello spazio che si può
motivare *a priori* — dirlo qui fa risparmiare una domanda al reviewer.

### IV-B — Come si aggrega (i primitivi), separato dal *cosa*  [~20 righe, il cuore]

Tre primitivi, definiti in astratto e poi mappati sulle superfici:

1. **Media dei pesi** (FedAvg pesato per numerosità dei dati; FedProx = stesso con termine
   prossimale). Applicabile a encoder, prior, e — cosa insolita — anche al *dizionario*, che si
   muove localmente (EMA + scadenza dei codici morti) e viene mediato.
2. **Merge per statistiche sufficienti**: i client inviano conteggi e somme per codeword, il
   server esegue la M-step. Esattamente additivo, non-gradiente, un solo passaggio per round.
   `\cite{dennis2021kfed}`.
3. **Union + re-clustering**: centroidi per client → unione → farthest-point + Lloyd pesato.
   Variante che preserva i motivi dei client di minoranza invece di mediarli.

Frase chiave della sottosezione (la tesi strutturale del paper): *superficie e primitivo sono assi
indipendenti, e la letteratura li confonde perché per una rete neurale ordinaria esiste un solo
primitivo sensato. Un dizionario discreto ne ammette almeno tre, e la scelta non è un dettaglio
implementativo.*

### IV-C — Il coordinamento fra le due fasi  [~15 righe]

Qui vive il contributo 1 di §I, e va detto in forma **strutturale**, non empirica:

- Un token è un indice; mediare i prior presuppone che l'indice significhi lo stesso ovunque
  (già argomentato in §III). Il sistema **rifiuta** la combinazione *prior federato + token id
  client-specifici*: è un guard nel codice, non una scelta sperimentale. Dirlo qui trasforma una
  precondizione in un fatto verificabile.
- Il ritmo del server in stage 2: $\tau$ passi locali per round invece di epoche locali. Definire
  $\tau$ qui (è un asse dello spazio), non nei risultati.
- Il prior `partial`: body condiviso, `channel_embedding` e `output_bias` locali. Va detto **cosa
  sono** quei due tensori — un reviewer non può indovinare che sono la testa di lettura per canale
  e il bias sul vocabolario.

### IV-D — Le configurazioni nominate (tabella)  [tabella + ~8 righe]

Una sola tabella, righe = configurazioni usate in §VI, colonne = coordinate. Schema in §4 qui sotto.
Il testo che la accompagna dice soltanto: (a) quali righe sono baseline/riferimenti e quali sono
arm, (b) che ogni riga è un punto dello spazio di IV-A/IV-B, (c) che `centralized` e il tokenizer
poolato non sono arm ma soffitti.

### IV-E — L'alternativa senza aggregazione  [~8 righe]

Definizione della fusione degli score: nessun parametro attraversa la rete, i client calcolano lo
score localmente e si media dopo z-normalizzazione. Sta in §IV perché è **un punto dello spazio**
(il punto in cui l'aggregazione avviene nello spazio delle funzioni invece che dei pesi), non un
risultato. Il risultato sta in §VI-4.

---

## 4. Schema della tabella (già compilato coi flag veri)

Colonne: `Encoder | BN | Codebook | Prior | Ritmo s2`. Il flag esatto va in una nota, non in cella.

| # | riga nel paper | Encoder | BN | Codebook | Prior | s2 |
|---|---|---|---|---|---|---|
| 0 | `local` (baseline) | locale | locale | locale | locale | — |
| 1 | `centralized` (soffitto, dati poolati) | — | — | — | — | — |
| 2 | codebook-only, merge suff-stat | locale | locale | suffstat | locale | — |
| 3 | codebook-only, media dei pesi | locale | locale | fedavg | locale | — |
| 4 | encoder-only | fedavg | **shared** (non `buffers_local`) | locale | locale | — |
| 5 | **A1** — tokenizer allineato, prior locale | fedavg | shared | suffstat | locale | — |
| 6 | **A2** — + body del prior condiviso | fedavg | shared | suffstat | partial | epoche |
| 7 | **cbfa** — A2 con dizionario mobile | fedavg | shared | fedavg | partial | epoche |
| 8 | **A2 + τ** | fedavg | shared | suffstat | partial | τ passi/round |
| 9 | fusione degli score (nessuna aggregazione di pesi) | locale | locale | locale | locale | — |
| — | *tokenizer poolato* (**illegale**, solo oracolo) | | | | | |

> **Correzione 2026-08-11 (verifica sul codice).** La riga 4 diceva `buffers_local`: sbagliato.
> L'unica cella encoder-only a disco (`zn_170_cblocal`, **una sola serie**, ucr_170) girò con
> `--fed-enc-bn shared`. `buffers_local` è solo il default della CLI, usato dal tag `zn_enc` —
> che però **non** è encoder-only, perché federa anche il codebook. La sezione scritta usa
> `pooled`.

Note operative sulla tabella:

- **Rinominare A1/A2.** «A1»/«A2» sono nomi interni e non dicono nulla. Proposta: *aligned
  tokenizer, local prior* e *aligned tokenizer, shared prior body*. Le sigle possono restare fra
  parentesi per il riferimento incrociato con §VI.
- **FedProx e FedProto**: sono girati, ma occupano 2 righe per rispondere a una domanda («e se
  l'aggregazione dell'encoder fosse più robusta?») che non è la domanda del paper. Raccomandazione:
  **fuori dalla tabella**, una frase in IV-B che dice che sono stati provati come varianti del
  primitivo 1 e riportati in §VI/appendice.
- La colonna `BN` ha valore solo dove l'encoder è federato: nelle righe 0-3 va un `—`, non
  `locale`, altrimenti sembra una scelta dove non c'è nulla da scegliere.

---

## 5. Figura (opzionale, decidere per ultimo)

Uno schema delle due fasi con box tratteggiati su ciò che attraversa la rete, e tre icone diverse
per i tre primitivi di IV-B. Vale una figura **solo se** riesce a mostrare che la stessa superficie
può avere primitivi diversi — altrimenti duplica la tabella e va tagliata per spazio.
Costo stimato: mezza colonna.

---

## 6. Decisioni aperte (servono prima di scrivere)

1. **Nomi delle righe.** A1/A2/cbfa oppure nomi descrittivi. → Raccomando descrittivi, sigla fra
   parentesi.
2. **FedProx/FedProto dentro o fuori la tabella.** → Raccomando fuori (una frase in IV-B).
3. **Dove definire la fusione degli score.** IV-E (spazio) oppure §VI come contributo separato.
   → Raccomando IV-E per la definizione, §VI per l'effetto: tiene §IV completa come mappa.

---

## 7. Checklist di verifica prima di fissare il testo

- [ ] Confermare, tag per tag in `artifacts/runs/`, che ogni riga della tabella corrisponde a celle
      realmente a disco con quei flag (il `cohort_fingerprint` **non** copre `fed_enc_prior`,
      `fed_enc_bn`, `window_normalization`: la garanzia è il `--tag`, non il fingerprint).
- [ ] Verificare che la riga 8 (τ) sia descritta col suo confonditore noto: `--fed-s2-val fixed`
      viaggia insieme a τ, e il contrasto pulito è τ − ctrl. In §IV basta **definire** i due assi
      separatamente; in §VI va riportato il contrasto pulito.
- [ ] Controllare che la descrizione del regime BN `buffers_local` non venga chiamata FedBN (lo
      diceva un vecchio help text del codice, ed è sbagliato: `buffers_local` condivide gli affini).
- [ ] Rileggere il guard di `federated_eval.py` che rifiuta prior federato + token id
      client-specifici, e citarne il comportamento in IV-C con precisione.
