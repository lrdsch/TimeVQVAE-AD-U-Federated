#!/usr/bin/env bash
# Onda 1 su g4 — wrapper senza argomenti, apposta.
#
# 🔴 PERCHE' ESISTE QUESTO FILE. Al primo tentativo (17:35) avevo passato gli argomenti a
# `wave_runner.sh` dentro la stringa di `run_on_g4.sh`, che li inoltra come
# `ssh ... "screen -dmS nome bash -lc '<CMD>'"`. Il mio `'2 3'` conteneva apici singoli DENTRO
# una stringa gia' fra apici singoli: la lista schede si e' spezzata in due parole, gli argomenti
# si sono spostati di posto (SLOTS=3, LOGDIR="4") e il dispatcher si e' duplicato cinque volte,
# lanciando 60 processi sulla sola GPU 2 — piu' copie della stessa cella sulle stesse cartelle.
# Ucciso dopo due minuti, nessun danno su disco (erano tutte ancora in stadio 1).
# La regola che ne segue: **niente argomenti dentro il comando remoto**. Si scrive un wrapper.
#
# SCHEDE. GPU 2 e 3, le due RTX 3090 libere. Non le Ada: la 0 e la 5 hanno VLLM di `pablo`, e
# la 4 e' si' libera ma mescolare Ada e 3090 metterebbe il confondente hardware DENTRO un
# contrasto fra arm (cudnn.benchmark sceglie i kernel a tempo). Due schede identiche, come
# vuole l'intestazione di `run_on_g4.sh`. Il tetto di due schede su g4 e' una richiesta esplicita.
#
# SLOT. 4 per scheda = 8 paralleli. g4 ha 48 core e stava a load ~8: 8 job aggiungono ~20,
# quindi ~28, sotto il tetto pratico di ~35. La memoria non e' il vincolo (1,5 GB per job su 24).
set -u
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated

bash scripts/wave_runner.sh logs/probe/wave1.jobs "2 3" 4 logs/probe
