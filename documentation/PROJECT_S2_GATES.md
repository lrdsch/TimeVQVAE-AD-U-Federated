# PROGETTO — Oracolo val, ctrl strumentato, fedtok-centralprior, hardening server-opt, FedProx-prior

**2026-08-07.** Progetto esecutivo dei gate e del codice deciso in
`A2_STAGE2_UPGRADE_DESIGN.md` §7. Quattro fasi: le prime due sono SOLO codice (zero GPU),
poi due lanci diagnostici, poi FedProx. Regole ferree invariate: un tag per variante
(l'out-json è cieco ai knob), GPU0/5 di g4 riservate, `_g4_gpu.sh` prima di ogni lancio.

---

## FASE 0 — Fix dell'oracolo val (prerequisito di tutto)

**Bug**: `_val_loss_prior` (federated.py:482-498) ri-campiona le maschere MaskGIT dal RNG
globale a ogni eval, senza seed, sotto autocast fp16 — σ ≈ 0,006-0,018 nats contro
`min_delta=1e-4`. Stesso problema nella val delle baseline (federated_eval.py:472).

**Fix — flag `--fed-s2-val {legacy,fixed}`, default `legacy`** (nessun run esistente cambia):
- `fixed` = (a) maschere deterministiche: `torch.random.fork_rng(devices=[d])` +
  `torch.manual_seed(blake2b(seed, client, batch_idx))` attorno alla forward di val — il
  fork NON tocca lo stream RNG del training (traiettorie byte-identiche); il seed è FISSO
  fra i round (stesse maschere a ogni eval ⇒ val comparabile round-su-round);
  (b) val in **fp32** (specchio della scelta stage-1, federated.py:466-474).
- Stesso hook per la val stage-2 delle baseline (`_converged_loop` val_fn) — non si
  rilancia nulla di vecchio: serve solo ai run nuovi.
- Dopo il fix: misurare la σ residua su 20 eval ripetute dello stesso corpo (attesa ~0);
  se σ < 1e-4 `min_delta` resta 1e-4, altrimenti `min_delta ≈ 2σ`.

**Accettazione**: (i) flag assente ⇒ nessun percorso toccato (diff dietro `if`); (ii) con
`fixed`, 20 eval dello stesso corpo ⇒ val identiche bit-per-bit; (iii) smoke 3-round: lo
stream di training è identico con flag on/off (confronto `mean_loss` per round).

## FASE 1 — Hardening server-opt + plumbing (solo codice, zero GPU)

Tutto ciò che la verifica ha dimostrato rotto o pericoloso, indipendentemente dalle varianti:

1. **Guardia mutua esclusione**: `server_opt=fedadam && server_momentum>0` ⇒ SystemExit al
   dispatch. **Fix del banner bugiardo**: allocare `_srv_x/_srv_v` e stampare il banner
   FedAvgM SOLO se il ramo girerà (oggi: alloca+stampa anche quando fedadam vince,
   federated.py:2088-2091 vs 2132/2139).
2. **FedAdam**: esporre `--server-b1/--server-b2/--server-eps` (default = gli attuali
   hardcoded 0.9/0.99/1e-3 ⇒ zero cambi di comportamento) e documentare nel help il cap
   del passo (~server_lr, picco transitorio ~2·server_lr al round ~13, niente
   bias-correction). Griglia sensata: slr {0.01, 0.03} — mai 1.0.
3. **FedAvgM**: documentare nel codice le **convenzioni di segno opposte** (FedAdam:
   Δ=agg−x ascent; FedAvgM: δ=x−agg descent — federated.py:2134 vs 2141) perché nessuno le
   "corregga"; aggiungere `--server-momentum-skip-round0` (v non accumula al round 0 — la
   media round-0 è misurata DISTRUTTIVA: val(agg₀)=4,59>ln64=4,16) e un refuse/warn per
   β>0.5 senza override esplicito (a β=0.9 il passo efficace è 10×, heavy-ball senza
   damping).
4. **τ + teste**: banner di contabilità quando `tau_steps>0` con prior partial (le teste
   prendono τ passi Adam/round: a τ piccoli la personalizzazione si accoppia al conteggio
   round; `--local-epochs` ignorato va DETTO nel banner). Knob nuovi `--fed-s2-val-every K`
   e `--fed-s2-patience` (oggi una sola patience comanda entrambi gli stage —
   federated_eval.py:723/757); patience in EVAL: `ceil(1800/(τ·K))`.
5. **Plumbing anti-collisione** (dal design doc, tutto verificato): forwarding nel ramo
   enc-trio SOLO via flag dedicati `--s2-*` con default nulli (MAI inoltrare
   `args.server_momentum`: default argparse 0.9!); bit terminali in `_arm_tag`
   (`tau{N}`, `srv-fedadam_slr{..}`, `sm{β}`, `pmu{μ}[_dec]`, emessi solo se non-default);
   entry in `FLAG_OWNER_ARMS` (flag senza arm proprietario ⇒ SystemExit); `fed_echo` →
   `knobs.json` esteso; guardie: knob stage-2 con `--fed-enc-prior local` ⇒ refuse.
6. **Chiusura del debito g4**: a dispatcher fermi (ora lo sono: campagna chiusa) applicare
   la patch 3-righe `_g4_gpu.sh` in `launch.sh` documentata in testa a `scripts/_g4_gpu.sh`.

**Accettazione**: rerun A2 senza flag nuovi ⇒ dir, RUN.json e comportamento identici;
le combo-trappola sollevano SystemExit; smoke fedadam+momentum rifiutato.

## FASE 2 — I due lanci diagnostici (decidono se le varianti si comprano)

### 2a. `zn_a2s2_ctrl` — il controllo strumentato
Resume dello stage 1 di A2 (`--resume-from <arm dir> --s1-rounds 0`, i 10 `_fed_resume.pt`
sono su disco), stage 2 FedAvg puro (ricetta A2), **oracolo `fixed`**, serie probe
{011, 014, 043, 170}, tag `zn_a2s2_ctrl`. Strumentazione (nel run, non post-hoc):
- **Detect lungo la traiettoria**: `--fed-s2-snapshot-every N` salva corpo+teste (~3,3 MB
  fp32; ~5 snapshot/cella) → script post-hoc lancia detect su ogni snapshot ⇒ curva
  val↔AUPRC round-per-round: È il dato che decide la wave τ.
- **best-val vs last-round**: salvare entrambi gli stati a fine run e fare detect su
  entrambi (su 170 la selezione su val è indiziata: es 0,060 vs kept-last 0,517).
- **`aggregation_penalty` per round**: val dei 5 corpi client di fine round PRIMA della
  media vs val della media (≈ raddoppia il costo val, che è l'1,5-2% del round) → in
  `fed_history.json`. Decide FedProx con un numero.
- **Margine sintetico**: NLL su val corrotta (spike + permutazione a blocchi) vs pulita,
  agli snapshot (in val non esistono anomalie etichettate).

**Test di validità**: `ctrl − A2` deve essere un null entro il rumore appaiato. ⚠️ Il ctrl
cambia due cose insieme (resume + oracolo): se il null fallisce, disambiguare con un
mini-ctrl a oracolo `legacy` su UNA serie prima di incolpare il resume.

### 2b. `zn_fedtokcp` — il tetto di ogni upgrade stage-2
Gli encoder di A2 finiscono IDENTICI fra client (`enc_max_cross_client_delta=0.0`) e il
codebook è condiviso ⇒ la tokenizzazione è indipendente dal client. Quindi: resume stage 1,
**UN prior allenato centralmente sul flusso di token pooled dei 5 client** con la ricetta
delle baseline (`_converged_loop`, warmup+cosine, batch 64), poi detect sui test dei
client. È un **DIAGNOSTICO federation-illegal** (pool di dati): etichettarlo così, mai arm
del paper. Costo ≈ 40% di una cella A2 (solo s2+detect) × 4 serie.

**Regola di decisione (pre-registrata)**:
- `fedtokcp ≤ A2` entro rumore su 011 e 043 ⇒ **la direzione stage-2 è morta**: nessuna
  variante si lancia (nemmeno prox oltre l'esplorativo), i soldi vanno allo stage 1.
- `fedtokcp` recupera una frazione sostanziale del gap verso centralized ⇒ il headroom
  esiste: si procede con τ=64/16 (+FedAdam slr 0,03 come controllo) e prox.

## FASE 3 — FedProx-sul-prior (implementazione certa, lancio gated)

Port come da design §3 (~120-150 righe, 2 file), **decoupled primaria**:
- Snapshot `w^t` in cima al round loop (post-broadcast) per-client fp32 on-device; loss
  form calcolata FUORI da `_amp()` in fp32; μ=0 ⇒ percorso byte-identico (addizione
  condizionale, non `+0.0`); `mean_loss` resta la task loss NON penalizzata.
- Telemetria: `prox_pull_frac` (decoupled) / `g_ratio` post-`unscale_` (solo loss-form,
  solo round tardivi); campi in `fed_history`.
- Griglia al lancio: **decoupled μ ∈ {1, 3}** (μ=10 = 94,6% di pull al client mediano di
  011 = un τ-piccolo travestito — solo come terzo tag etichettato "soft-tau"); loss-form
  implementata ma non svolta in prima battuta.
- **Lancio dopo la Fase 2**: se `aggregation_penalty ≈ 0` nel ctrl (atteso: le fed_history
  di A2 mostrano risalite ≤+0,0057), i tag prox sono esplorativi dichiarati — si lanciano
  comunque (decisione utente) ma con aspettative oneste: P(vittoria oltre rumore) ~10-15%.

**Accettazione**: smoke 3-round μ=0 ⇒ `fed_history` bit-identica al pre-patch; μ=3
decoupled ⇒ `prox_pull_frac` nel range analitico 1−(1−lr·μ)^s; guardia μ>0 con prior
local ⇒ refuse.

## Stato (2026-08-07, sera)

✅ **FASE 0 e FASE 1 SCRITTE E TESTATE** (non ancora committate):
- `--fed-s2-val {legacy,fixed}` in `_val_loss_prior` (fork_rng + seed blake2b per client,
  fp32) E nel loop converged delle baseline (`_S2_VAL_MODE` + `_val` in
  `_build_train_stage2_converged`); default `legacy` = percorso intatto.
- `federated_stage2`: kwargs `server_b1/b2/eps`, `server_momentum_skip_round0`,
  `val_mode`, `val_every`; guardia fail-fast fedadam+momentum; banner FedAvgM onesto
  (warn β>0.5); patience conta le EVAL; banner τ dichiara `--local-epochs` ignorato.
- Dispatch: flag `--s2-server-momentum` (default 0!), `--s2-momentum-skip-round0`,
  `--server-b1/b2/eps`, `--fed-s2-val`, `--fed-s2-val-every`, `--fed-s2-patience`;
  guardie (prior local + knob s2 ⇒ refuse; fedadam+momentum ⇒ refuse); `fed_echo` esteso;
  `_arm_tag` con bit terminali `tau/srv/sm/pmu`; 11 entry nuove in `FLAG_OWNER_ARMS`.
- `launch.sh`: guardiano `_g4_gpu.sh` dentro (rc=4 su GPU 0/5), applicato a dispatcher fermi.
- Test passati: py_compile, `--help`, `_arm_tag` byte-identico sui nomi esistenti,
  owner-guard (`--tau-steps` + `cb_only` ⇒ SystemExit), guardia fedadam+momentum,
  seed val deterministico, `bash -n launch.sh`, `_g4_gpu.sh` rc=4.

⏳ Da scrivere: strumentazione ctrl (snapshot + aggregation_penalty), driver `zn_fedtokcp`,
port FedProx-prior (Fase 3), righe `zn_report`.

## Aggiornamento (2026-08-08)

✅ **FASE 2 ESEGUITA E VERDE** — tetto fedtokcp (AUPRC): 011 +0,123 · 043 +0,219 (sopra
`centralized`) · 170 +0,114 (tokenizer colpevole confermato) · 014 +0,001 (già al tetto).
Null ctrl−A2 regge. `agg_penalty` NEGATIVA sui round tardivi ⇒ prox resta riga di
completezza dichiarata. Wave τ (tau64/tau16/t64adam × probe) + ctrl esteso a n=10 in volo;
`tauext` (18 celle) in coda a scia.

✅ **FASE 3 SCRITTA** (2026-08-08): port FedProx-sul-prior come da design §3 —
`prior_prox_mu/form` in `federated_stage2`, meccanica in ENTRAMBI i loop
(`_local_train_prior` epoch e `_local_train_prior_steps` τ) via helper condivisi
(`_prox_pairs_prior`/`_prox_step_prior`/`_prox_finish_prior`); anchor `w^t` snapshottato in
cima al round loop (post-broadcast, solo PARAMETRI del body, mai le teste); `mean_loss`
resta la task loss; μ=0 ⇒ percorso identico (pairs vuoti). CLI `--fed-prior-prox-mu`,
`--fed-prior-prox-form {decoupled,loss}` (default decoupled); guardie: prior local ⇒
refuse (esteso `_s2_knobs_set`), fast-path converged non ingoia più μ>0, set anchorato
vuoto ⇒ ValueError; `fed_echo` pmu/pform; bit `pmu{μ}[_dec]` già in `_arm_tag`;
`FLAG_OWNER_ARMS` +2. Tag `zn_a2s2_pmu1`/`zn_a2s2_pmu3` (decoupled, 10 serie) registrati
nel manifesto; driver `scripts/zn_s2_prox.sh` (smoke | queue): la coda parte DA SOLA
quando tau/tauext/ctrl sono finiti (celle=0 E nessun `zn_s2_wave.sh` vivo — anti-collisione
con la scia di tauext). Micro-test: `_prox_term` esatto, contrazione decoupled esatta,
pull μ=1 ≈13,1%/round e μ=3 ≈34%/round a lr=1e-3, filtro buffer OK, tag pmu OK.

## Ordine, costi, report

| # | cosa | costo | dipende da |
|---|---|---|---|
| 0 | fix oracolo (`--fed-s2-val fixed`) | codice | — |
| 1 | hardening + plumbing + patch launch.sh | codice | — |
| 2a | `zn_a2s2_ctrl` 4 serie strumentate | ~3-4 job-h | 0, 1 |
| 2b | `zn_fedtokcp` 4 serie | ~3-4 job-h | 1 |
| 3 | port FedProx | codice | 1 |
| 3b | tag prox μ∈{1,3} su probe | ~5 job-h | 2a (gate), 3 |

- Manifest: nuove entry in `cohorts/zn_ownership.json` per `zn_a2s2_ctrl`, `zn_fedtokcp`,
  `zn_a2s2_px*`; `zn_owner.py --check` verde prima di ogni lancio.
- `zn_report.py`: righe ARMS per i nuovi tag; COPPIE: `ctrl − A2` (null obbligatorio),
  `fedtokcp − A2` (il gate), `fedtokcp − centralized` (quanto del gap è stage-2),
  `px* − ctrl`. Metrica primaria AUPRC appaiata; VUS-PR mai aggregata.
- GPU: g4 schede 1-4 via `_g4_gpu.sh`, g2 via `_g2_gpu.sh`, mai MPS su g4.
- Seed 0: tutti questi sono diagnostici; il multi-seed resta il prerequisito del paper.
