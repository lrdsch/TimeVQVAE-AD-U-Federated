# §Results — piano deciso (2026-08-11)

Tutti i numeri qui sotto sono ricalcolati dagli out-json a disco, AUPRC media sui 5 client,
unità di analisi = la serie, n=10 sulla coorte `ucr2p_10`, seed 0.

---

## Decisione 0 — la configurazione di punta resta **A2**

τ=64 esce dai contributi. Motivo, misurato: `ctrl` (= A2 + `--fed-s2-val fixed`) copre tutte
e 10 le serie, e il contrasto τ **pulito** `tau64 − ctrl` è **4 vittorie / 6 sconfitte,
mediana −0,0009**. Il `tau64 − A2` che sembrava forte (7W/3L, somma +0,268) è in gran parte
l'oracolo di validazione, non τ.

Verificato che `ctrl − A2` è un contrasto a **flag singolo**: gli altri due flag del tag
(`--fed-s2-agg-penalty`, `--fed-s2-snapshot-every 4`) non toccano il training — il primo
logga `val_post − val_pre` (`federated_eval.py:1412-1414`), il secondo salva checkpoint.

| contrasto | W | L | mediana | somma |
|---|---:|---:|---:|---:|
| `ctrl − A2` (solo val fissa) | 5 | 4 | +0,0028 | +0,186 |
| `tau64 − ctrl` (solo τ) | 4 | 6 | −0,0009 | +0,081 |
| `tau64 − A2` (entrambi) | 7 | 3 | +0,0019 | +0,268 |

**Conseguenza editoriale: Introduction e Table `tab:space` NON vanno riscritte.** Dicono già
A2. Va tolta solo la coda della frase su τ in `paper.tex:104-107`.

τ sopravvive come **una sottosezione breve di ablazione**, con il risultato onesto e non
banale: l'effetto del ritmo di sincronizzazione è **asimmetrico**. Peggiorarlo costa molto
(`zn_a2s2_lep`, stesso budget di passi ma sync ogni ~310 passi invece di 100-190: −0,198 su
011, −0,245 su 043), migliorarlo oltre il default compra poco (τ=64: nullo a n=10). Il
default a 10 epoche locali è già sul tratto piatto della curva.

---

## Decisione 1 — baseline: **composito convergito**, non `zn_main/local`

`zn_main/local` ha training che toccano il tetto di 10 000 step su **6 serie su 10**
(222: 6/12 client, 011: 4/10, 229: 4/10, 001: 2/10, 014: 2/10, 170: 1/10). Il protocollo già
scritto (`paper.tex`, §Training) dice che le celle troncate sono **escluse e non riportate**:
usarle contraddirebbe la sezione precedente.

Baseline = `zn_es` dove esiste (001, 011, 014, 170, 222, 229), `zn_main` altrove (043, 082,
083, 086). Va dichiarato in nota, con il conteggio dei troncamenti.

`zn_ot` **non entra**: gira con `KEEP_LAST_WEIGHTS=1`, che viola «i parametri del round
migliore vengono ripristinati». Va citato solo in nota come l'unica baseline appaiata su
hardware, spiegando perché non è utilizzabile.

---

## Decisione 2 — i tre claim del corpo, con le formulazioni esatte

### C1 · `centralized > local` — la motivazione
Regge su AUPRC. **Ma** sulla metrica ufficiale dell'archivio (top-1@64) è 4 vittorie / 4
pareggi / 0 sconfitte, e `centralized − max(local)` è **0,0000 su 8 serie su 8**.

> Formulazione ammessa: «pooling improves the *typical* client, not the best one: on no
> series does the pooled model locate an anomaly that no client locates.»
> **Vietato**: «pooling unlocks detection».

Questa frase è già corretta in `paper.tex:84-86`. Va solo sostenuta con la colonna
`max(local)` dentro la tabella principale, non solo nel testo.

### C2 · `A2 > local` — il risultato
**9 vittorie / 1 sconfitta, p = 0,011** contro la baseline convergita. Ma l'effetto è
**concentrato**: 043 +0,456, 011 +0,389, 014 +0,334, e tutte le altre sei ≤ +0,035. A un
margine di 0,03 diventa 4W/1L/5 pareggi.

> Formulazione: «A2 improves over converged local training on 9 of 10 series; the improvement
> is concentrated — on three series it exceeds +0.33 AUPRC, on five it is below 0.035, and on
> one (`ucr_170`) the federated model loses 0.138.»

Il segno unico è il risultato; la concentrazione va detta nella stessa frase, non nascosta.

### C3 · L'interazione — **il contributo vero**
Lo stesso cambio di flag (prior locale → corpo del prior condiviso) misurato due volte:

| backbone | contrasto | W/L | mediana |
|---|---|---|---|
| tokenizer **allineato** | `A2 − A1` | **9/1** | **+0,069** |
| tokenizer **non allineato** | `federated − cb_only` | 4/6 | −0,001 |
| differenza di differenze | | **8/2** | **+0,136**, p = 0,055 |

> Formulazione: «sharing the prior body is directionally beneficial only once encoder
> averaging has aligned the tokenizers: the identical modification is a null on an unaligned
> backbone (median −0.001, 4/10) and a consistent gain on an aligned one (median +0.069,
> 9/10); the interaction contrast is positive on 8 of 10 series.»
> **Vietato**: «il prior condiviso è la leva» (è un'interazione, non un effetto principale).
> **Vietato**: riportare l'interazione confrontando «p significativo vs p non significativo» —
> il test corretto è la DiD, e va riportata quella.

La cella «prior condiviso senza tokenizer allineato» **non è misurabile**: il codice la rifiuta
al lancio. Va detto — è una proprietà del problema, non una lacuna dell'esperimento.

---

## Decisione 3 — struttura della sezione

Ordine **sviluppo → conferma → meccanismo**, perché è l'unico in cui il risultato scomodo (la
metrica ufficiale non separa) sta al centro e non in coda: una sezione letta a metà non deve
produrre una conclusione più ottimistica del vero.

1. **V.A Reference points** — C1. Tabella I.
2. **V.B The federated configuration** — C2. Tabella I.
3. **V.C What makes it work: an interaction** — C3. Tabella II (matched ablations).
4. **V.D Stage-2 rhythm** — l'ablazione τ + l'asimmetria. Due paragrafi, nessuna tabella.
5. **V.E Confirmation on a random sample** — c50, pre-registrata.
6. **V.F Failure case and an alternative** — `ucr_170` + fusione function-space.
7. **V.G Limitations**.

## Decisione 4 — tabelle e figure

- **Tabella I** (corpo): 10 serie × {local, max(local), centralized, A2}, AUPRC + top-1@64.
  È la tabella che risponde all'Introduzione. Riempie il `[INSERT HERE...]` di `paper.tex:108`
  con **9/1/0** (W/L/T a margine zero) o **4/1/5** a margine 0,03 — da scegliere insieme alla
  soglia, non dopo.
- **Tabella II** (corpo): le ablazioni appaiate — A1, A2, cb_only, federated, cbfa — con la DiD.
- **Figura 1** (corpo): `ucr_170`, profili di score per-timestep dei 5 client locali contro
  quelli di A2, che mostra il falso positivo che diventa unanime. È l'unica figura che dice
  qualcosa che nessuna tabella dice.
- **Supplementare**: sensibilità media↔mediana (sposta `A2−A1` da p=0,011 a p=0,109), curva
  del margine, tabella τ completa, sweep `ks`.

## Decisione 5 — contingenza su c50

- **arriva a 50**: V.E è la conferma pre-registrata, e diventa il risultato principale.
- **si ferma a ~30**: V.E resta valida — l'ordine di esecuzione era randomizzato, quindi il
  prefisso completato è un campione casuale per costruzione. Va riportato n effettivo e freeze.
- **non arriva**: V.E cade, la sezione resta sullo sviluppo, e il limite va dichiarato: le 10
  serie sono una selezione mirata e nessuna stima si estende all'archivio.

## Decisione 6 — limiti da scrivere

Seed singolo (nessuna barra d'errore da repliche complete); n effettivo ~6 perché 082 è a zero
per tutti, 083 a soffitto, 229 a pavimento, 086 shard-dipendente; set di sviluppo purposivo
(criterio sul periodo, non sulla difficoltà); i 5 client sono shard di un segnale, non repliche.
