# Istruzioni per il repo

## ⛔ `artifacts_archive/` — non leggere

Contiene le run **precedenti al 2026-08-04**, tutte prodotte **senza
`window_normalization=zscore`**. Sono un altro modello: su `ucr_001`, a parità di tutto il
resto, l'AUPRC di `centralized` passa da 0,646 a 0,946.

**Nessun numero di lì è confrontabile con i numeri nuovi e nessuno può entrare nel paper.**
Se per ricostruire la storia devi comunque aprirlo, etichetta ogni riga
`window_normalization = none` e non mescolarla mai con righe nuove in una media, un test
appaiato o una figura. Dettagli in [artifacts_archive/README.md](artifacts_archive/README.md).

I numeri sono già estratti in `evidence/records_pre_znorm_20260804*.{csv,json}`: la cartella
da 26 GB si può cancellare senza perdere una metrica.

Le run nuove vanno in `artifacts/runs/`.

## Regola di configurazione: z-norm sempre

Dal 2026-08-04 **ogni run usa `dataset.window_normalization = zscore`**. Non è un'opzione
da valutare: è la configurazione del paper originale (§5.2 e §A.1 di
`documentation/2311.12550v5.pdf`, enunciata nella stessa frase della regola `W = 2×periodo`).

Non si ablaziona. Va **pinnata nella coorte**, non passata come `--extra`, così è
impossibile lanciare senza per distrazione.

## Trappola: il `cohort_fingerprint` è incompleto

Oggi copre solo `datasets` e `seeds`. **Non** copre `window_normalization` né `amp_dtype`.
Quindi due celle con lo stesso fingerprint possono essere modelli diversi, e gli script che
aggregano per fingerprint le dichiarano confrontabili. Finché non è corretto, la protezione
è il `--tag` distinto.

## ⛔ GPU: quali si possono usare

### g4 — **GPU0 e GPU5 sono RISERVATE**

**Non si usano se non te lo dice espressamente l'utente.** Sono le due schede che cede ad
altri. Non è una questione di carico: **anche completamente vuote, non sono nostre da
prendere.**

Usabili: **GPU1, GPU2, GPU3** (3090) e **GPU4** (Ada).

Il guardiano è `scripts/_g4_gpu.sh`, e va chiamato prima di ogni lancio su g4:

```bash
bash scripts/_g4_gpu.sh "$LAUNCH_GPUS" || exit 4   # esce 4 se la lista tocca 0 o 5
bash scripts/_g4_gpu.sh --libere                   # -> "1 2 3 4"
```

Deroga solo su indicazione diretta, e visibile nel comando:
`G4_ALLOW_RESERVED=1 LAUNCH_GPUS="0 5" …`

Restano valide le due regole dure preesistenti su g4: **mai una scheda con processi di altri
utenti**, **mai MPS** (è una macchina di dipartimento con 20+ utenti).

⚠️ `scripts/launch.sh` non ha ancora il controllo dentro — al 2026-08-06 girava in 7 istanze
e bash rilegge gli script a offset di byte. La patch è documentata in testa a `_g4_gpu.sh` e
va applicata **a dispatcher fermi**.

### g2 — **mai entrambe le schede se lavora qualcun altro**

```
altri su 0 GPU  ->  entrambe
altri su 1 GPU  ->  l'altra, e una sola
altri su 2 GPU  ->  UNA sola, la meno carica, senza impedirgli il lavoro
```

Non si applica a mano: `export LAUNCH_GPUS="$(bash scripts/_g2_gpu.sh)"`.

⚠️ Su g2 c'è **ssanchez** (e `javih`). Due trappole nell'output di `nvidia-smi`: il conteggio
dei processi per GPU **non dice di chi sono** (serve `ps -o user=`), e il
`nvidia-cuda-mps-server` compare nella lista **senza essere un job**. Leggerli male fa
sembrare libere schede che sono di altri — è già successo.

⚠️ Il vincolo vero su g2 è la **CPU**, non la GPU: 16 core, e ssanchez da solo ne usa il 491%
contro il nostro 129%. Restare leggeri anche in **slot**, non solo in schede.

## Ambiente

Python: `/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10` — non esiste `python` sul
PATH.
