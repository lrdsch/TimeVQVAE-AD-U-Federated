# ucr_split — previsioni pre-registrate a run incompleto

**Congelate il 2026-07-27T10:49:49+00:00**, commit `2538987` (branch `speedup-port`, working tree
sporco: 428 file non committati — le previsioni numeriche vivono in
`documentation/ucrsplit_prereg.json`, che è tracciato).

Scopo: scrivere *prima* che i dati arrivino cosa mi aspetto dai 58 cluster ancora da chiudere,
così che a run finito il confronto sia una verifica e non una ricostruzione a posteriori.
Verifica con:

```bash
/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10 scripts/ucrsplit_predictions_check.py
```

---

## 1. Stato al congelamento

| | |
|---|---|
| run | `scripts/run_ucrsplit.sh`, avviato 2026-07-25 21:51:36, orchestratore PID 2038499 |
| job | 147/492 json prodotti, 14 in volo, **0 fallimenti**, **0 run TRUNCATED** (round mediano 35, max 108 su un tetto di 300 → la patience scatta davvero) |
| cluster con tutti e 6 gli arm | **24 / 82** |
| metrica primaria | `vus_pr` |
| unità di analisi | **il cluster** (media entro cluster dei delta appaiati per-client) |
| protocollo | `--protocol converged --s1-rounds 300 --fed-patience-rounds 6 --local-epochs 10 --batch 64`, seed 0 |

### L'unità è il cluster, non il client

I 5 client di un cluster **condividono lo stesso test set** (in `ucr_109` tutti e 5 hanno
`test_length` 44795; è solo il *train* a essere diviso in 10/10/20/20/30%). Le loro 5
valutazioni sono quindi 5 letture della stessa serie: ICC 0.82, **n effettivo ≈ 27** per 115
client. Analizzare a n=115 è pseudo-replicazione e gonfia i p-value di circa un fattore √5.

Concretamente, all'unità onesta (n=24) i contrasti federati che a n=115 sembravano
significativi non lo sono: `fedavg` p 0.027 → 0.48, `fedprox` 0.012 → 0.50,
`commoninit` 0.028 → 0.17. Sopravvive solo `centralized`, p < 0.001.

Stessa trappola già documentata nell'audit del 2026-07-15 (lì: n=31 e non 124).

---

## 2. Il difetto del disegno da cui nasce tutto

Lo scheduler è LPT (job ordinati per costo decrescente) con la prima ondata seminata sui **3
cluster più economici**. Il costo *è* la lunghezza della serie. Risultato: i 24 cluster chiusi
sono **bimodali**.

```
train totale dei 24 fatti:
  2430  2430  2477  |<--- buco --->|  18000 ... 50510   (21 cluster)
                     tutti i 58 rimasti
                     stanno QUI: [2520, 17100]
```

**Zero** dei 24 cluster chiusi cade nel range dei 58 pendenti. I 58 rimasti sono 5.45× più
corti dei 24 fatti (train mediano 4 500 vs 24 525, Mann-Whitney p = 1.0e-7): non sono lo stesso
popolo, e la differenza è sulla variabile che governa il meccanismo in studio.

### La pendenza regge su 3 serie

| campione | slope Δ(central−local) ~ log₁₀(train) | p |
|---|---|---|
| tutti i 24 | **−0.176 / decade** | **0.004** |
| solo il clump lungo (n=21) | −0.117 / decade | 0.377 |

I 3 cluster corti pesano il **44% della leva** della retta (leverage 0.29 ciascuno, soglia
2p/n = 0.17). Dentro il clump lungo la pendenza non è distinguibile da zero. Il contrasto grezzo
però è forte: **Δ(central−local) = +0.258 sui 3 corti vs +0.062 sui 21 lunghi**, quattro volte.

Direzione coerente col meccanismo atteso (col 10% di 2 450 campioni il client locale è affamato,
col 10% di 30 000 no) — ma sono 3 serie, scelte per essere economiche, non a caso.

**È questo il motivo per non fermare il run**: la tesi vive nel regime a serie corta, e in quel
regime oggi n = 3.

La lunghezza spiega r² = 0.32 del gap. Il **68% resta rumore per-serie**: le previsioni sotto
sono medie, non valgono per il singolo cluster.

---

## 3. Le previsioni

Δ = arm − `local`, media dei delta appaiati per-cluster, metrica `vus_pr`. `pred sui 58` =
retta fittata sui 24 e valutata sui log-lunghezze veri dei 58 pendenti. `pred n=82` = media su
tutti gli 82.

| arm | osservato n=24 | pred sui 58 | **pred n=82** | IC95 |
|---|---|---|---|---|
| `centralized` | +0.086 | +0.185 | **+0.156** | [+0.107, +0.205] |
| `fedproto` | −0.008 | +0.054 | **+0.036** | [−0.047, +0.119] |
| `commoninit` | −0.005 | +0.041 | +0.028 | [−0.061, +0.116] |
| `fedprox` | −0.044 | +0.032 | +0.010 | [−0.074, +0.094] |
| `fedavg` | −0.051 | −0.003 | −0.017 | [−0.099, +0.066] |

Livelli assoluti (media, non delta) previsti sui 58 corti: `local` 0.186 → **~0.30**,
`centralized` 0.272 → **~0.48**. Le mediane basse osservate ora (0.036 / 0.093) sono un
artefatto delle serie lunghe e difficili.

MDE (80% potenza, α .05 bilaterale) — **è specifico per arm**, la sd dei delta cambia:

| contrasto | MDE a n=24 | MDE a n=82 (se la sd regge) |
|---|---|---|
| `centralized` − `local` | 0.069 | 0.037 |
| `fedproto` − `local` | 0.105 | 0.056 |

Sui cluster corti la sd probabilmente cresce, quindi l'MDE reale a n=82 sarà più vicino a
0.05 che a 0.04. Solo a quel punto un effetto da +0.036 diventa visibile: **ora è invisibile
per costruzione**, e per questo «`fedproto` pareggia `local`» oggi non è dicibile — p = 0.35 è
un test di superiorità fallito, non un'equivalenza, e l'IC95 [−0.086, +0.062] contiene una
perdita pari all'intero premio dell'accentramento.

---

## 4. Cosa NON cambierà (già in cassaforte)

1. **L'ordinamento degli arm.** Stabile da k=9: `centralized` ≫ `local` ≳ `fedproto` ≈
   `commoninit` > `fedprox` > `fedavg`.
2. **Nessun arm federato recupera l'accentramento.** Anche nella previsione più generosa il
   migliore (`fedproto`, +0.036) copre ~23% di un premio che sale a +0.156.
3. **Il premio dell'accentramento è reale su dati IID.** `ucr_split` divide il train della
   *stessa* serie: `centralized − federated` è perdita di aggregazione pura, l'eterogeneità
   non è disponibile come scusa.

## 5. Cosa può cambiare segno — l'esito interessante

Tutti e quattro gli arm federati sono previsti passare da **sotto** `local` a **sopra**.
Meccanismo: quando il client è affamato di dati anche un'aggregazione mediocre aggiunge
informazione; quando ha già 10 000 campioni suoi la sincronizzazione è solo interferenza.

Se accade, la storia passa da *«la federazione non serve»* (negativo piatto) a *«la federazione
aiuta dove il client è affamato e danneggia dove non lo è, e in nessun regime recupera più di un
quarto del premio dell'accentramento»* — un negativo con meccanismo e crossover.

### Regole di decisione, fissate ora

| esito a n=82 | lettura |
|---|---|
| Δ(central−local) ∈ [+0.107, +0.205] | previsione confermata; il gap dipende dalla lunghezza; riportare la regressione come risultato meccanicistico |
| Δ(central−local) resta ≈ +0.086 | **il modello lunghezza→gap è falsificato**: i 3 corti erano un caso. Il gap è una costante e il crossover non esiste. Da riportare come tale |
| Δ(central−local) > +0.25 | i 3 corti sottostimavano; controllare che non sia un artefatto di `local` che collassa su shard minuscole (verificare i livelli assoluti, non solo i delta) |
| ≥1 arm federato con Δ > 0 e IC95 che esclude lo zero | **crossover confermato** → il paper diventa negativo-con-meccanismo |
| tutti gli arm federati con IC95 che contiene lo zero | nessun crossover dimostrabile a n=82; serve seed multipli, non più cluster |

Nota su cosa questo run **non** potrà mai dire: un'equivalenza (`fedproto` = `local`) richiede un
TOST con margine deciso in anticipo, e con MDE ~0.05 il margine dovrebbe essere più grande del
premio stesso dell'accentramento. Non è ottenibile qui: servono seed multipli.

## 6. Restrizione d'ambito se il run venisse interrotto

> Su 24 cluster UCR **a serie lunga** (train 18k–50k, più 3 a ~2.4k), divisi in modo IID, il
> premio dell'accentramento è +0.086 [+0.039, +0.135] VUS-PR e nessun arm federato lo recupera.
> Il gap cresce al ridursi della lunghezza (−0.18/decade, p=0.004, stimata però su 3 serie
> corte), quindi questa è una **stima per difetto**.

L'ambito va dichiarato, non nascosto: senza i 58 pendenti il claim poggia su 3 serie nel regime
che conta.
