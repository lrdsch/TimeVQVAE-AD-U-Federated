# Cose da fare a mano, adesso (2026-08-02)

Vivevano nel prompt di un cron che **non è mai scattato** (scatta solo a sessione ferma: zero
esecuzioni in dieci ore). Ora il battito arriva da `scripts/status_tick.sh` sotto Monitor, che
emette una riga ogni 15 min — ma quella riga è solo un allarme, non esegue niente. Queste
restano azioni a mano, ed è per questo che sono scritte qui invece che in un commento.

## 🔴 Due uccisioni programmate

1. **`chain_g4` va ucciso appena `ucr_043` è completa (30/30).** In lista dopo ucr_043 gli
   restano ucr_086, ucr_170 e ucr_083 — tutte e tre coperte da flussi dedicati (`g086_g4`,
   `gpu5_g4`, `gpu0_g4`). Se dispatchasse, due processi partirebbero sulla **stessa cella**:
   `launch.sh` salta solo ciò che ha già un `report.json` **al momento del dispatch**, quindi
   una cella in corso non lo protegge. `kill -9 -<PGID>`, poi verificare 0 orfani.
2. **Lo screen `abl` su g2 va ucciso appena `ucr222_proto_count` è 2/2**, perché poi passerebbe
   ai tag `ucr229_*` che sta già facendo `abl229`.

Se il dispatch doppio è già avvenuto: controllare i ckpt di quelle celle e rifarle da zero.

## Ambito allargato il 2026-08-02

`local` e `centralized` a **K=128** (tag `*_cb128base`), 36 celle: 2 arm x 2 build x 9 serie.
Senza di loro il blocco K=128 aveva i due arm federati ma **né baseline né tetto**, quindi
«la federazione continua a non battere `local` anche a K=128?» era senza risposta — e
confrontare con `local@64` avrebbe messo K dentro il contrasto. `--codebook-size` scrive
`cfg.quantizer.codebook_size` prima di ogni ramo di arm (`federated_eval.py:1232`): K è un
iperparametro del modello, non una manopola di federazione.

Totale: **270 → 306 celle**. Il floor NON si rifà: non ha codebook (media mobile, AR, PCA),
quindi è già K-invariante e àncora entrambi i blocchi.

## Alla fine di tutto

1. `scripts/backfill_protocol_topk.py` un'ultima volta — i job in volo durante la modifica di
   `detect.py` scrivono solo le chiavi `@64`.
2. Audit appaiato a 9 serie con le soglie corrette: top-1 ±0,23 · top-3 ±0,40 · VUS-PR ±0,143 ·
   AUROC ±0,029, **unità = SERIE, non cella**.
3. Tabella finale.

## Trappole già pagate, da non ripagare

- `pgrep -f <pattern>` conta anche i worker del dataloader (400 invece di 12) **e uccide sé
  stesso** se il pattern è nella propria riga di comando: usare `[f]ederated_ev` e filtrare sul
  ppid.
- Un tag a **0 celle può essere morto, non in coda**: verificare che esista `cohorts/<nome>.json`
  e che lo screen sia vivo.
- `nvidia-smi` al 99% non vuol dire saturo; `ps -eo pcpu` è una media di vita, non istantanea.
- Su g2 non superare load ~16 su 16 core; su g4 ~35 su 48.
