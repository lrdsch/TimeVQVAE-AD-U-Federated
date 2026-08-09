# Pre-registrazione — sweep di `score_window_size_rates` sulle 10 serie

**Scritta il 2026-08-09, PRIMA di produrre qualunque numero fuori da `ucr_170`.**
Chiusa alle 02:2x, prima che lo scout del codice restituisse il piano operativo.
Se una scelta qui sotto viene cambiata dopo aver visto i risultati, va scritto **qui**, con
data e ragione — altrimenti il risultato non è pre-registrato e va riportato come esplorativo.

---

## 1. Da dove viene l'ipotesi, e perché va pre-registrata

Su `ucr_170`, cambiando **solo** `score_window_size_rates` e senza riallenare nulla, tutti e
8 gli arm passano da «non rileva» a «rileva»: `zn_a2` 0,086 → 0,316, `tau16` 0,291 → 0,686,
`ctrl` 0,082 → 0,378. Il meccanismo misurato è questo:

- il **falso positivo** rende il prior **affilato e sbagliato** (entropia 1,55 contro 1,83 di
  fondo; «sicuro e sbagliato» 40,7% contro 11,3%; NLL > 8 nel 36,0% contro 0,1%);
- l'**anomalia** rende il prior **incerto** (entropia 2,08). Una predittiva quasi uniforme dà
  NLL ≈ log K = 4,16: la sua sorpresa è **limitata per costruzione**, quella del FP no.

Quindi **allargare la maschera premia il FP più dell'anomalia**. Rapporto picco FP/anomalia su
`zn_a2`: 1,04 a ks=5 · 1,64 a ks=13 · **1,89 a ks=21** — monotono nel raggio.

⚠️ Questa diagnosi è **post-hoc su una serie di cui conoscevamo la risposta**, e `ks=1` **non
è un rate del paper**. Da sola non vale nulla: serve una previsione fatta prima.

## 2. Cosa prevede il meccanismo — e cosa NON prevede

Il meccanismo **non** prevede un guadagno uniforme. Prevede un effetto **eterogeneo con un
segno strutturato**:

- **aiuta** dove il falso positivo dominante è una ricostruzione sicura-e-sbagliata di un
  tratto normale ma insolito;
- **non aiuta, e può danneggiare**, dove l'anomalia stessa è una deviazione **lunga e
  strutturale**: una maschera stretta la rende localmente prevedibile dal contesto adiacente,
  che è ancora dentro l'anomalia.

Registrare questo **ora** è ciò che impedisce all'ipotesi di essere infalsificabile: senza
questo paragrafo, qualunque esito la «conferma».

## 3. La griglia

| asse | valori | nota |
|---|---|---|
| kernel di maschera | **paper (0,1/0,3/0,5)** · ks=1 · ks=3 · ks=5 | il primo è il controllo |
| serie | tutte e 10 della coorte `ucr2p_10` | `ucr_170` inclusa ma **esclusa dall'analisi primaria** |
| arm | `zn_a2` (primario) · `zn_main/centralized` · `zn_main/local` | vedi §4 |
| seed | 0 | unico disponibile; nessun ± |

⚠️ **I rate del paper sono RELATIVI, non assoluti.** `ks = rate × W_lat`, e `W_lat` dipende
dalla serie perché `W = 2 × periodo`. Quindi la configurazione del paper applica maschere
assolute **diverse** a serie diverse: su una serie a periodo lungo il buco è molto più largo.
Poiché la variabile del meccanismo è la **larghezza assoluta del buco**, la griglia è definita
in **ks assoluto**, e `W_lat` per serie va registrato insieme al risultato. Se il codice
accettasse solo rate, si sceglie per ogni serie il rate che dà il ks assoluto voluto, e si
riporta lo scarto di arrotondamento.

**Ordine di esecuzione**: prima il controllo (rate del paper) su tutte le celle, per il test di
riproduzione di §6. Solo dopo i ks nuovi.

## 4. Il contrasto primario, e perché è quello

Non è «ks=1 migliora l'AUPRC». È:

> **Δ_A2 − Δ_centralized**, dove Δ_arm = AUPRC(ks=1) − AUPRC(paper), mediana sulle 9 serie
> diverse da `ucr_170`.

La ragione: se restringere la maschera aiuta `centralized` **quanto** aiuta `A2`, allora è
semplicemente una configurazione di scoring migliore **per tutti** e non spiega **niente** del
divario federato. Solo se aiuta `A2` di più, una parte di ciò che attribuiamo alla federazione
è in realtà configurazione di punteggio. `local` entra come terzo termine per distinguere
«aiuta i modelli a poco addestramento» da «aiuta i modelli federati».

Metrica primaria **AUPRC**, media sui 5 client come nel resto della campagna.
Unità di analisi il **cluster** (n = 9). VUS-PR e `paper_top1@64` come secondarie —
la VUS-PR **non è aggregabile fra serie UCR** e serve solo per la concordanza di segno.

## 5. Il criterio di lettura, deciso adesso

Soglia di rumore **AUPRC = 0,075** (2 SD in regime dal contrasto nullo `ctrl − A2`, 9 serie,
post-z-norm, W = 2P). È un **limite superiore** del rumore di run, quindi conservativa.

| esito | condizione | conseguenza |
|---|---|---|
| **GENERALIZZA** | mediana Δ_A2 > +0,075 **e** ≥ 6/9 serie positive | la configurazione di scoring è un confondente reale: la sezione «failure case» del paper va riscritta, e ogni divario va ri-misurato a maschera fissa |
| **SPECIFICO DI 170** | ≤ 3/9 serie positive **e** mediana entro ±0,075 | resta una diagnosi per-serie. Nessuna riga di tabella, nessuna riscrittura |
| **DANNOSO** | mediana Δ_A2 < −0,075 | i rate del paper sono vicini all'ottimo; il risultato su 170 è un artefatto di quella serie e va detto |
| **NON FEDERATO** | Δ_A2 − Δ_centralized entro ±0,075, qualunque sia il segno di Δ_A2 | è una proprietà di TimeVQVAE-AD, **non** della federazione: non tocca nessun claim del paper |

L'ultima riga ha **precedenza** sulle prime tre: si legge per prima.

## 6. Controlli di validità obbligatori, PRIMA di leggere qualunque Δ

1. **Riproduzione.** Ri-punteggiare a rate del paper deve riprodurre i numeri già a disco a
   ~6 decimali. Cella di riferimento: `zn_a2 / ucr_011 / federated_enc_fedavg`, AUPRC 0,820.
   Se non riproduce, lo sweep si ferma: si sta misurando un'altra cosa.
2. **La cache non deve mentire.** Nei ckpt c'è `detect_score_cache.npz`. Se
   `score_window_size_rates` **non** entra nella sua chiave, ri-punteggiare restituirebbe in
   silenzio gli score vecchi e l'intero sweep sarebbe un no-op travestito da risultato.
   Prova richiesta: due ks diversi **devono** produrre vettori di score diversi sulla stessa
   cella. Se sono identici bit-a-bit, la cache non è stata neutralizzata.
3. **Tutto il resto fisso.** Tolleranza 64, `W = 2 × periodo`, z-norm per finestra,
   aggregazione fra client invariata, stesso codice di metrica. Ogni parametro che cambia
   insieme a ks rende il confronto non appaiato.

## 7. Cosa NON si potrà dire, qualunque sia l'esito

- Non si potrà mettere `ks=1` in una tabella del paper come configurazione: **non è un rate
  del paper**, e sceglierlo dopo aver visto i risultati sarebbe selezione sull'esito.
- Non si potrà rivendicare l'**inversione della classifica** fra arm dopo aver rimosso i falsi
  positivi: già misurato, dipende dal metodo.
- Non si potranno usare percentili di «tipicità» calcolati sullo shard: passano da 22,6 a 85,4
  solo cambiando il set di riferimento.
- Tutto è **seed 0**, e su `ucr_split` l'unità è il cluster: 5 client non sono 5 replicati.
  Nessun risultato di questo sweep è multi-seed, e va detto ogni volta che lo si cita.
- Tre serie su nove non discriminano (`ucr_229` pavimento, `ucr_083` soffitto, `ucr_086`
  shard-dipendente): l'n effettivo è ~6, non 9. La mediana su 9 va riportata **insieme** alla
  mediana sulle discriminanti.

## 8. Costo dichiarato

Ri-punteggiamento di checkpoint esistenti: **nessun training**. Se il costo misurato dovesse
risultare paragonabile a un training, lo sweep si ferma e si rinegozia — la ragione per cui è
stato messo in cima alla coda è proprio che è quasi gratis.
Vincolo di macchina: g2, **solo GPU1**; GPU0 è di ssanchez e non si tocca. Il collo di
bottiglia di g2 è la CPU (16 core), quindi lo sweep va in coda o a bassa concorrenza.
