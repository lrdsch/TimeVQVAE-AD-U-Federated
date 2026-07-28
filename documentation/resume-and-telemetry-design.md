# Crash-safe resumable training — design

Applies to all four repos (`TimeVQVAE-AD-M`, `-Fall`, `-Federated`, `-Real`). All `file:line` citations are against `/home/leonardo/PhD/TimeVQVAE-AD-M-Real` unless noted, and were re-read against source while writing. Per-repo deltas are called out where they matter (`min_epochs` field absent in `-M`; otherwise stage/utils/run files are logically identical).

This document supersedes the draft. It resolves every blocker and major from the adversarial panel; minors are noted inline with a `[minor]` tag.

---

## 1. Feasibility verdict

**Can resume be made "perfettamente equivalente" (every element of the resumed trajectory identical to the no-crash run from the resumed epoch onward, losing only the in-flight epoch)?**

**Yes — but only under a *strict determinism tier* (Tier S), and Tier S must be the *coupled default whenever resume is enabled*, on BOTH the original and the resumed launch.** The draft's "pragmatic" Tier P (the default `cudnn.benchmark=True`, no deterministic algorithms) does **not** satisfy the contract, and the reason is structural, not cosmetic:

- The VQ-VAE quantizer assignment is `cdist → argmin` (vector_quantizer.py:179-180). A 1-ULP difference in `cdist` (from a different cuDNN/cuBLAS kernel chosen by the autotuner across a process restart, or from a nondeterministic atomic-accumulation kernel) can flip an `argmin` near a tie.
- A flipped assignment changes `ema_cluster_size` (vector_quantizer.py:139), which can cross the dead-code threshold `ema_cluster_size < threshold_dead_code` (vector_quantizer.py:152).
- Crossing that threshold triggers `_expire_dead_codes` → `_sample_vectors` → `torch.randperm`/`torch.randint` on the data device **with no `generator=`** (vector_quantizer.py:69-75, 157). That is a *conditional draw from the global device RNG* that the golden run either did or did not make.
- One extra/missing draw **permanently desynchronizes** the device RNG stream that *also* drives stage2 prior masking (`torch.randperm`, prior.py:285/498/515/720/735), dropout (prior.py:537), and — indirectly — the DataLoader base-seed.

So GPU floating-point nondeterminism is **not bounded noise**; it is a fork in the RNG stream. The contract "any other divergence is not acceptable" therefore forces determinism at the kernel level.

**Determinism tiers (formal):**

| Tier | Settings | Guarantee | Cost |
|---|---|---|---|
| **S-CPU** | device=CPU, deterministic algos on | Bit-identical, provably | training on CPU (impractical for real runs, used in CI) |
| **S-GPU** (required default under `RESUME`) | `cudnn.benchmark=False`, `cudnn.deterministic=True`, `torch.use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 off (`matmul_precision="highest"`), **same GPU+driver+torch+cuDNN build** | Bit-identical **except** for the residual nondeterministic ops listed below; in practice trajectory-identical and, with those ops on CPU, fully bit-identical | ~5–25% slowdown; TF32-off adds ~20–40% on Ampere+ matmuls |
| **P** (opt-in relaxation, NOT equivalence-preserving) | current `cudnn.benchmark=True` | Statistically similar only; **violates the contract on GPU** | zero |

**Residual non-bit-exactness even in S-GPU (must be addressed, not hidden):** `warn_only=True` is required today because `torch.istft` (transforms.py:63) and `F.interpolate` backward in the decoder have no deterministic CUDA kernel — under `warn_only=True` they silently run nondeterministic kernels on the training hot path, feeding the same `argmin`-fork mechanism. **True bit-exactness on GPU therefore requires** running STFT/iSTFT and decoder upsampling on CPU (or replacing them with deterministic ops) and flipping to `warn_only=False` so any remaining nondeterministic op hard-fails at startup. If the researcher accepts "trajectory-identical with FP-noise only in the iSTFT/interpolate path" rather than literal bit-exactness, `warn_only=True` is acceptable and should be documented as such. **Recommendation:** ship S-GPU with `warn_only=True` as the practical default, and provide a `STRICT_BITEXACT=1` switch that forces iSTFT/interpolate to CPU + `warn_only=False` for the validation runs that prove equivalence.

**Bottom line:** the *state-restore* mechanics (weights, AdamW moments, LambdaLR position, the RNG streams, VQ EMA buffers, early-stop anchors) are necessary and, as corrected below, sufficient *given* determinism. The honest contract is: **bit-identical resume under S-GPU with the iSTFT/interpolate caveat; literal bit-exactness under S-CPU or `STRICT_BITEXACT`.** Tier P is offered only as an explicitly-labelled relaxation.

---

## 2. What must be checkpointed (complete state table)

`state_dict` already carries weights **and** VQ EMA buffers (`ema_cluster_size`, `ema_embed_sum`, `initialized`) because they are `register_buffer`s (vector_quantizer.py:105-107). Everything else below is new.

| Element | Stage1 | Stage2 | Currently saved? | How to capture | How to restore | Cite |
|---|---|---|---|---|---|---|
| Model weights | yes | yes (prior + frozen stage1) | yes (`state_dict`) | `model.state_dict()` | `model.load_state_dict(..., strict=True)` | stage1.py:210; stage2.py:410 |
| VQ EMA buffers `ema_cluster_size`, `ema_embed_sum`, `initialized` | **live, mutated each step** | frozen (never mutated) | yes (inside `state_dict`) | in `state_dict` | in `load_state_dict`; restored `initialized=True` suppresses re-`_kmeans_init` | vq:105-107,114 |
| Optimizer (AdamW `exp_avg`,`exp_avg_sq`, per-param `step`) | `model.parameters()` | `model.prior.parameters()` only | **NO** | `opt.state_dict()` | `opt.load_state_dict()` | stage1.py:393; stage2.py:533-537 |
| Scheduler (`LambdaLR.last_epoch`, `_step_count`, `_last_lr`) | yes | yes | **NO** | `sched.state_dict()` | rebuild `sched` (lambda is NOT pickled), then `sched.load_state_dict()` | stage1.py:413; stage2.py:558 |
| `step` (global step == `sched.last_epoch`) | yes | yes | yes but never read | `step` | `step = ck["step"]` | stage1.py:456,500; stage2.py:598,628 |
| `epoch` | yes | yes | yes but never read | save `epoch + 1` (next epoch to run, §4) | `epoch = ck["epoch"]` | stage1.py:457,610; stage2.py:599,681 |
| `best_val`, `best_step` (early-stop anchors) | yes | yes | **NO** | capture **after** the best-update block (§4 ordering fix) | `best_val, best_step = ck[...]` | stage1.py:447-448,580-581; stage2.py:589-590,671-672 |
| Python RNG | via torch only for shuffle; **load-bearing for nothing in stage1 loop** | **load-bearing** (`random.random()` in prior masking) | NO | `random.getstate()` | `random.setstate()` | prior.py:496,513,525,718,733,745 |
| NumPy RNG | not load-bearing in loop (`np.mean` only) but cheap | same | NO | `np.random.get_state()` | `np.random.set_state()` | utils.py:109 |
| Torch CPU RNG | **load-bearing** (DataLoader shuffle base-seed) | same | NO | `torch.get_rng_state()` | `torch.set_rng_state()` | data.py:1077,1170 |
| Torch CUDA RNG (active device) | **load-bearing** (`_sample_vectors` dead-code expiry every step) | **load-bearing** (`torch.randperm`/`torch.rand` masking, dropout) | NO | `torch.cuda.get_rng_state()` for the **active** device (see §2.1) | `torch.cuda.set_rng_state(state, device)` | vq:72-74,157; prior.py:281,285,498,515,537 |
| DataLoader shuffle generator (NEW, mandatory) | yes | yes | NO | explicit `torch.Generator`, `gen.get_state()` | restore `gen.set_state()` **or** reseed `gen.manual_seed(seed+epoch)` per epoch (§5) | data.py:1072,1305 |
| `csv_rows` (metrics history, verbatim) | yes | yes | NO | store the list of dicts verbatim in payload | `csv_rows = ck["csv_rows"]` (no CSV reparse — §4.4) | stage1.py:428,564; stage2.py:571,657 |
| `loader_fingerprint` (asserted, not restored) | yes | yes (+`use_cache`) | NO | dict, see §2.2 | assert-equal on resume | — |
| `format_version`, `stage`, `cfg_dict`, `sha256` | yes | yes | partial (`cfg_dict`) | literals + integrity hash | validated on load | stage1.py:211 |

### 2.1 CUDA RNG: use the active device, not `_all`

The draft used `get_rng_state_all()`/`set_rng_state_all()`, which is keyed to device *ordering/count*. If `CUDA_VISIBLE_DEVICES` or device count changes between crash and resume (cluster reboot, different GPU assignment by the launcher), `set_rng_state_all` mis-maps or errors. **Save and restore only the single training device's RNG** via `torch.cuda.get_rng_state()` / `torch.cuda.set_rng_state(state, device)` keyed to the actual `device` used by the loop (stage1.py:333). Combined with the device fingerprint (§2.2), this removes the topology hazard.

### 2.2 `loader_fingerprint` / `training_fingerprint` — asserted on resume (recomputed, never restored)

These are pure functions of `cfg` + dataset shape; restoring them would mask a real config/data drift. **Assert equality** so any drift hard-fails with a clear message instead of silently diverging. The draft's set was too narrow.

```python
fingerprint = {
    # data/schedule shape
    "batches_per_epoch": batches_per_epoch,        # len(train_loader)  stage1:345 / stage2:493
    "max_steps": max_steps,                        # recomputed         stage1:397-404 / stage2:541-549
    "warmup_steps": warmup_steps,                  # stage1:405 / stage2:550
    "batch_size": batch_size,                      # stage1:cfg.dataset.batch_size_stage1 / stage2 _stage2
    "use_cache": use_cache,                        # stage2 ONLY        stage2:485/491
    # early-stop decision inputs (panel: blocker — none of these were asserted)
    "min_epochs": min_epochs,                      # EFFECTIVE getattr result  stage1:398 / stage2:542
    "patience_steps": patience_steps,              # stage1:451 / stage2:593
    "early_stopping": cfg.training.early_stopping, # config.py:234
    "early_stopping_min_delta": cfg.training.early_stopping_min_delta,  # config.py:236
    "check_val_every_n_epoch": cfg.training.check_val_every_n_epoch,    # config.py:193
    # LR identity (LambdaLR restores the multiplier; base_lr comes from cfg at opt build)
    "lr": cfg.training.lr,                          # config.py:187
    "weight_decay": cfg.training.weight_decay,      # stage2 only        config.py:188
    "seed": cfg.seed,                               # config.py:309
    # determinism + environment identity (panel: major — was a test-time assumption)
    "deterministic": deterministic,
    "matmul_precision": "highest" if deterministic else "high",
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "device_name": (torch.cuda.get_device_name(device) if device.type=="cuda" else "cpu"),
    "device_count": (torch.cuda.device_count() if torch.cuda.is_available() else 0),
}
```

On a version/device mismatch: **hard error under `deterministic=True`** (equivalence cannot be honored), loud warning under Tier P. `min_epochs` feeds *both* the `max_steps` floor and the early-stop gate `(epoch+1) >= min_epochs` (stage1.py:583; stage2.py:674); asserting `max_steps`+`min_epochs` closes both. **`-M` repo note:** `min_epochs` field is absent → `getattr(..., 0)` yields `0`; the *effective* `0` is what gets fingerprinted, so `-M`-to-`-M` resume is consistent and the assert still protects against accidental cfg drift.

### 2.3 What is intentionally NOT in the payload (recomputed deterministically, then asserted)

- `max_steps`, `warmup_steps`, `lr_lambda` closure — pure functions of `cfg`+`batches_per_epoch`; rebuilt, then `sched.load_state_dict` restores `last_epoch` (the lambda is not picklable and is not saved by `LambdaLR.state_dict`). After load, `lr = base_lr * lambda(last_epoch)`; `base_lr` is captured at `opt` construction from `cfg.training.lr`, which is why `lr` is in the fingerprint.
- Snapshot windows (`_select_diverse_windows`, stage1.py:241-282) — deterministic, diagnostic-only.
- Token-cache values — deterministic given the frozen stage1 best.ckpt; but the **cache file itself must be made atomic + checksummed** (§3.5).
- There is **no dead-code "age counter"** to save `[minor]` — `_expire_dead_codes` is stateless per step, keyed only on `ema_cluster_size` (vector_quantizer.py:152), which is restored via `state_dict`. The panel's draft inventory expectation of such a counter was wrong; nothing to add.

---

## 3. Atomic, crash-proof checkpoint write

### 3.1 `atomic_torch_save` (utils.py, new) — body checksum + durable rename

The panel showed that a structurally-valid zip whose tensor data blocks were lost (possible on a crash where metadata journaled but data did not, or on `nobarrier` mounts) loads **without raising** and passes a key-presence check → silent resume from zeros. **A content checksum is mandatory.** Serialize to an in-memory buffer, hash the body, embed the digest, write the buffer.

```python
import os, io, hashlib, uuid
import torch
from pathlib import Path

def atomic_torch_save(payload: dict, path: Path) -> None:
    """Crash-safe, integrity-checked torch.save: hash body -> temp -> fsync(fd)
    -> os.replace -> fsync(parent dir). A power loss never destroys the prior
    committed `path`, and a torn body is detectable on load via sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    buf = io.BytesIO()
    torch.save(payload, buf)                 # payload WITHOUT the digest
    body = buf.getvalue()
    digest = hashlib.sha256(body).hexdigest()

    # Re-serialize with the digest embedded so load can self-verify.
    buf2 = io.BytesIO()
    torch.save({"__body__": body, "__sha256__": digest}, buf2)
    blob = buf2.getvalue()

    # Unique temp name (panel major: fixed temp name races a stray relaunch).
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "wb") as fh:
        fh.write(blob)
        fh.flush()
        os.fsync(fh.fileno())                # data on platter, not just page cache
    os.replace(tmp, path)                    # atomic rename, same dir/fs (ext4 verified)
    _fsync_dir(path.parent)                  # durable directory entry — MANDATORY (see below)

def _fsync_dir(d: Path) -> None:
    fd = os.open(str(d), os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
```

`os.replace` is atomic only when src and dst share a filesystem; the same-dir temp construction guarantees that. **The post-replace `_fsync_dir` is mandatory, not best-effort** (panel major): on ext4 ordered mode the tmp-fd fsync makes the data durable and the post-replace dir-fsync makes the renamed dirent durable; swallowing its error silently can publish a `last.ckpt` whose blocks never reached disk. Do **not** wrap it in a bare `except OSError: pass`. `[minor]` Assert at startup that `ckpt_dir` is local (not NFS/overlay): `os.stat(ckpt_dir).st_dev == os.stat(tmp_dir).st_dev` is guaranteed by same-dir, but additionally refuse atomic-resume claims on network FS — document "local ext4/xfs only" (the verified mount is ext4 at `/dev/mapper/vg0-lv--0`).

### 3.2 Load with corruption rejection

```python
def _load_one(p: Path) -> dict | None:
    try:
        wrapper = torch.load(str(p), map_location="cpu", weights_only=False)
        body = wrapper["__body__"]; want = wrapper["__sha256__"]
    except Exception:
        return None                          # truncated/torn zip, missing keys
    if hashlib.sha256(body).hexdigest() != want:
        return None                          # structurally valid but torn data blocks
    ck = torch.load(io.BytesIO(body), map_location="cpu", weights_only=False)
    if not _ckpt_complete(ck):
        return None                          # old 4-key format or wrong version
    return ck

def _ckpt_complete(ck) -> bool:
    required = {"format_version","stage","state_dict","optimizer","scheduler",
                "step","epoch","best_val","best_step","rng",
                "loader_fingerprint","csv_rows"}
    return isinstance(ck, dict) and ck.get("format_version") == 2 and required <= ck.keys()
```

`[minor]` After `model.load_state_dict(strict=True)` (which already validates keys/shapes) add a fast `torch.isfinite` smoke probe on a sampled restored tensor; treat failure as corruption → fall back.

### 3.3 Ping-pong double buffer with an atomic pointer (replaces draft's copy-then-overwrite)

The draft's "demote current good to `prev`, then overwrite `last`" can leave **both** slots invalid (panel blocker): during the new write the only good copy lives in a possibly-unfsynced `prev`. **Never copy a live good file.** Use two real slots and a tiny atomic pointer; the pointer flip is the single linearization point.

```python
def save_resumable(ckpt_dir: Path, payload: dict) -> None:
    cur = _read_pointer(ckpt_dir)            # "0" | "1" | None
    nxt = "1" if cur == "0" else "0"         # write to the slot NOT in use
    slot = ckpt_dir / f"ckpt_{nxt}.pt"
    atomic_torch_save(payload, slot)         # fully committed before pointer moves
    _write_pointer(ckpt_dir, nxt)            # atomic: pointer.tmp -> fsync -> replace -> dir fsync

def _write_pointer(ckpt_dir: Path, val: str) -> None:
    p = ckpt_dir / "CURRENT"
    tmp = p.with_name(f"CURRENT.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "w") as fh:
        fh.write(val); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, p); _fsync_dir(ckpt_dir)

def load_resumable(ckpt_dir: Path) -> dict | None:
    order = []
    cur = _read_pointer(ckpt_dir)
    if cur in ("0","1"):
        order = [cur, ("1" if cur=="0" else "0")]   # prefer pointed slot, then the other
    else:
        order = ["0","1"]
    for s in order:
        ck = _load_one(ckpt_dir / f"ckpt_{s}.pt")
        if ck is not None:
            return ck
    return None
```

**Crash-point analysis:** before the pointer flip the old slot is authoritative and untouched; after the flip the new slot is fully fsynced. At every instant ≥1 complete, checksum-valid checkpoint exists. A torn `CURRENT` (caught by `_read_pointer` try/except → `None`) just falls back to trying both slots.

**`best.ckpt`** is also written via `atomic_torch_save` (with checksum) but is **single-slot** (not a resume anchor). To remove the foot-gun the panel flagged, write `best.ckpt` as the **legacy 4-key payload** (`state_dict`/`cfg_dict`/`step`/`epoch`) so it can never be mistaken for a resume anchor, and select it downstream via a load+validate, not bare `.exists()` (§4.6, §6).

### 3.4 Cadence

**Epoch-boundary only** — matches the accepted "lose the in-flight epoch" contract; no mid-epoch sampler/RNG serialization is needed. One `save_resumable` per epoch at the corrected save site (§4.3).

### 3.5 metrics.csv and token_cache.pt are inside the crash window — make them atomic too

- **metrics.csv** is rewritten in `"w"` (truncate) mode every epoch (stage1.py:440; stage2.py:583). A crash mid-rewrite destroys all history the instant `open("w")` succeeds (panel blocker). **Render to a temp file, fsync, `os.replace`, dir-fsync** — reuse a tiny `atomic_write_text`. Because we now restore `csv_rows` verbatim from the checkpoint (§4.4), metrics.csv is a *reconciled-to-checkpoint* artifact and never the source of truth.
- **token_cache.pt** is a single non-atomic `torch.save` at the end of stage1 (data.py:1254) and is gated downstream purely by `cache.exists()` + a `config_hash` (stage2.py:478, data.py:1289) with **no integrity check** (panel blocker). A torn cache silently flips `use_cache` → changes `batches_per_epoch` → changes `max_steps`/`warmup_steps`/LR-horizon for stage2 — a wholesale trajectory change, not "lose one epoch". Fix:
  1. Write the cache via `atomic_torch_save` (checksum embedded).
  2. In `PrecomputedTokenDataModule.setup` (data.py:1285) verify the checksum; on failure treat as "no cache" and regenerate, never train on a torn cache.
  3. Tie the stage1 `.train_complete` sentinel (§6) to *after* the cache is durably committed, so resume regenerates a missing/torn cache before stage2.

### 3.6 Strict cross-file commit order

Each individual file is atomic, but they must agree. **Commit order per epoch:** (1) atomically commit `metrics.csv`; (2) atomically commit the checkpoint **last** via `save_resumable`. A present, pointer-referenced checkpoint then *implies* `metrics.csv` was already durable. The checkpoint is the single commit record; on resume, `csv_rows` is restored from it verbatim and `metrics.csv` is rewritten from that (so a stale/ahead `metrics.csv` is simply overwritten and reconciled). `[minor]` Diagnostic artifacts (snapshot PNGs, `vram.csv`) stay non-atomic but their (re)initialization on resume is wrapped in try/except so a torn diagnostic file can never abort the resume path.

---

## 4. Resume logic in `main()`

### 4.1 Opt-in flag + coupled determinism default

Add to `TrainingConfig` (config.py:182), default **off** so current behavior is byte-identical:

```python
resume: bool = False
deterministic: bool = False
```

In `main()` (both stages), near the top after `cfg` is loaded:

```python
resume = cfg.training.resume or os.environ.get("RESUME") == "1"
# Equivalence requires strict determinism. Couple it to resume by default.
deterministic = (cfg.training.deterministic
                 or os.environ.get("DETERMINISTIC") == "1"
                 or resume)                       # resume ON ⇒ Tier S unless explicitly relaxed
if os.environ.get("ALLOW_NONDETERMINISTIC_RESUME") == "1":
    deterministic = cfg.training.deterministic or os.environ.get("DETERMINISTIC") == "1"
```

This makes Tier S the default whenever resume is on (panel: Tier P violates the contract). A researcher who knowingly accepts Tier P sets `ALLOW_NONDETERMINISTIC_RESUME=1`.

**Required imports** (panel minor — the stage files import neither): add `import os` and `import random` to both `pipeline/stage1.py` and `pipeline/stage2.py`.

### 4.2 Determinism block (replaces the unconditional `benchmark=True`)

Replace stage1.py:330-331 / stage2.py:443-444:

```python
if deterministic:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # MUST precede CUDA init
    torch.set_float32_matmul_precision("highest")                 # TF32 OFF (panel: TF32 widens argmin ties)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    bitexact = os.environ.get("STRICT_BITEXACT") == "1"
    torch.use_deterministic_algorithms(True, warn_only=not bitexact)
    # bitexact path additionally forces iSTFT/decoder-interpolate to CPU (transforms.py:63,
    # decoder upsampling) so warn_only=False does not hard-fail; see §1 residual-ops note.
else:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True                          # unchanged Tier-P behavior
```

### 4.3 Save-site ordering fix (best-anchor off-by-one — panel BLOCKER)

Verified: `save_*_checkpoint(last_ckpt_path, ...)` runs at stage1.py:576 / stage2.py:667 **before** `best_val`/`best_step` are updated at stage1.py:580-581 / stage2.py:671-672. If we capture the anchors at the old save site, `last.ckpt` for epoch N carries the epoch-(N−1) anchors, so on resume the patience countdown `step - best_step` is stale → **early stopping fires at a different step than the no-crash run**.

**Fix: move the `save_resumable` call to AFTER the best-update block**, and capture `best_val`/`best_step` *after* they reflect epoch N's val result. New structure (stage1 shown; stage2 identical with its line numbers):

```python
# ─ Checkpoint (best update FIRST, then the resumable snapshot) ─
if ran_val:
    current_val = row.get("val/loss", float("inf"))
    if current_val < best_val - cfg.training.early_stopping_min_delta:
        best_val = float(current_val)
        best_step = step
        save_legacy_best(best_ckpt_path, model, cfg, step, epoch)   # 4-key, atomic
    elif (cfg.training.early_stopping and (epoch + 1) >= min_epochs
          and step - best_step >= patience_steps):
        print(... early stopping ...)
        stop = True

# RNG snapshot is taken HERE — the provably-last RNG event of the epoch
# (snapshots/val below are eval/no_grad and draw no training RNG; see §4.7).
payload = build_payload(
    stage="stage1", model=model, cfg=cfg, opt=opt, sched=sched,
    step=step, epoch=epoch + 1,                 # NEXT epoch to run (§4.5)
    best_val=best_val, best_step=best_step,     # now reflect epoch N
    rng=snapshot_rng(device), fingerprint=fingerprint,
    csv_rows=csv_rows,
)
atomic_write_text(csv_path, render_csv(csv_rows))   # metrics committed first (§3.6)
save_resumable(ckpt_dir, payload)                    # checkpoint committed last
```

The snapshot-reconstruction block (stage1.py:593-608) and the val loop (stage1.py:510-532) run under `model.eval()` + `no_grad`; in eval the VQ skips `_kmeans_init`/`_update_ema`/`_expire_dead_codes` (guarded by `self.training`, vector_quantizer.py:174,183) and dropout is off, so they draw **no** training RNG. Placing the snapshot after them but capturing it as the last RNG event keeps the boundary clean. (Validation test 4.5/4.6 proves this empirically.)

### 4.4 Resume restore block — placement and order

Insert **after** the scheduler is built and **after** the second `seed_everything` (stage1.py:418; stage2.py:564), and **before** the `step=0; epoch=0` init (stage1.py:456-457; stage2.py:598-599). It needs `opt`, `sched`, `csv_path`, `best_*`, `max_steps`, `warmup_steps`, `batches_per_epoch`, `device` to already exist.

```python
start_step, start_epoch = 0, 0
if resume:
    ck = load_resumable(ckpt_dir)
    if ck is None:
        print("[resume] no valid checkpoint; starting fresh.")
    else:
        fp = ck["loader_fingerprint"]
        _assert_fingerprint(fp, fingerprint)     # raises on ANY drift (data/schedule/early-stop/lr/env)
        model.load_state_dict(ck["state_dict"])  # weights + VQ EMA buffers, strict=True
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])   # last_epoch == saved step; lambda rebuilt above
        best_val, best_step = ck["best_val"], ck["best_step"]
        start_step, start_epoch = ck["step"], ck["epoch"]
        csv_rows[:] = ck["csv_rows"]             # verbatim — NO CSV reparse (panel major)
        # RNG restored LAST so it overrides the pre-loop seed_everything:
        restore_rng(ck["rng"], device)
        print(f"[resume] epoch={start_epoch} step={start_step} "
              f"best_val={best_val:.6f} best_step={best_step}")
step = start_step          # was: step = 0   (stage1.py:456 / stage2.py:598)
epoch = start_epoch        # was: epoch = 0  (stage1.py:457 / stage2.py:599)
```

`restore_rng` must be the **structurally last RNG-touching statement** before the loop. `model.load_state_dict`, `opt.load_state_dict`, `sched.load_state_dict` are RNG-free; placing `restore_rng` at the end of the block guarantees the loop enters with the saved RNG.

### 4.5 Epoch-boundary / off-by-one rule

`last.ckpt` (now `ckpt_{0,1}.pt`) is written at the epoch boundary, *after* the epoch's batches advanced `step`, *before* `epoch += 1` (stage1.py:610; stage2.py:681). **Save `epoch + 1`** so `start_epoch` is the first epoch *not yet run*. The `while ... epoch < stage{1,2}_max_epochs` bound (stage1.py:463; stage2.py:605) then skips the completed epochs naturally. The RNG snapshot at that point reflects the state *after* the completed epoch's shuffle/mask/dead-code draws and *before* the next epoch's loader iterator is created → the resumed loop's first `for batch in train_loader` draws the identical permutation (given §5).

### 4.6 metrics.csv: verbatim restore (panel major — drop the CSV reparse)

The draft's `_reload_metrics_csv` re-typed cells via `_num()` and dropped empty cells, reconstructing each row's key-set differently from the in-memory original; the next `_write_metrics_csv` union/header could then differ from the golden file (float round-trip, NaN, trailing zeros, key-presence drift). **Store `csv_rows` verbatim in the payload and restore `csv_rows[:] = ck["csv_rows"]`.** The rows are already plain `dict[str, float|int]`. This guarantees byte-identical subsequent rewrites with no round-trip. Drop `_reload_metrics_csv` entirely.

### 4.7 `seed_everything`-clobber fix

The pre-loop `seed_everything(cfg.seed)` (stage1.py:418; stage2.py:564) must still run on a fresh start (it wipes RNG consumed by materialization/window-selection/token-weight gathering). On resume we **let it run, then immediately overwrite** the RNG in the restore block (`restore_rng` is the last statement). This is correct because `random.setstate`/`np.random.set_state`/`torch.set_rng_state`/`torch.cuda.set_rng_state` fully replace the seeded state, and it touches fewer lines than conditional skipping. (Equivalently: `if not resumed: seed_everything(cfg.seed)`.)

---

## 5. Determinism: strict vs pragmatic + the DataLoader generator (mandatory)

§1 establishes Tier S as the required equivalence tier. Two additional, **mandatory** hardening items for the actual run configuration (`num_workers=4` default, config.py:97; `persistent_workers=True`, data.py:1080/1313):

### 5.1 Explicit, checkpointed shuffle generator (panel BLOCKER — was "optional hardening")

With `shuffle=True` and **no** `generator=` (data.py:1077/1170, 1311), `RandomSampler` builds a fresh generator seeded from the **global torch CPU RNG at iterator-creation time**. The per-epoch permutation is then a function of incidental global-RNG consumption between the restore point and the first batch — fragile. **Make shuffle a pure, restorable function of saved state:**

- Add `generator: torch.Generator | None = None` to `_build_loader` (data.py:1072) and `_loader` (data.py:1305), passed straight to `DataLoader`. Default `None` preserves current behavior when resume is off.
- Build **one** `torch.Generator` per training run; either (a) checkpoint `gen.get_state()` and restore it, **or** (b) reseed deterministically per epoch at the top of the loop: `train_gen.manual_seed(cfg.seed + epoch)`. Option (b) is simpler and makes the permutation for epoch N identical regardless of crash history; it is the recommended form.

### 5.2 Worker RNG audit + `worker_init_fn` (panel BLOCKER → resolved by audit)

Each persistent worker is seeded once at iterator creation from `base_seed + worker_id`; that seed differs between a fresh run (workers spawned at epoch 0) and a resumed run (workers spawned at epoch N), so **any worker-side randomness would diverge** and is never serialized. **Audit result (verified):** `SlidingWindowDataset.__getitem__` (data.py:~1010-1053) and `_TokenDataset.__getitem__` (data.py:1266) perform only deterministic numpy slicing / tensor indexing — **no `random`/`np.random`/torch RNG**. Therefore worker base-seed divergence is irrelevant *today*, and equivalence holds despite it. To keep this guarantee robust against future `__getitem__` changes:

- Set a `worker_init_fn` that seeds each worker deterministically from `(cfg.seed, epoch, worker_id)` (not from `torch.initial_seed()`), and
- Add a one-time startup assertion/comment that `__getitem__` is RNG-free.
- Validation runs the resume tests with `num_workers=4` (the production default), not `0`, since that is exactly where this would silently break.

### 5.3 Cost summary

| Setting | Throughput cost | Needed for |
|---|---|---|
| `cudnn.benchmark=False` + `deterministic=True` | ~5–25% | Tier S baseline |
| TF32 off (`matmul_precision="highest"`) | +20–40% on Ampere+ matmuls | killing TF32 argmin-tie flips |
| iSTFT/interpolate on CPU + `warn_only=False` (`STRICT_BITEXACT`) | extra; varies | literal bit-exactness (validation only) |
| explicit per-loader generator | ~0 | mandatory; restorable shuffle |

---

## 6. Launcher (run.py) interaction & idempotent re-launch

`run.py` runs each `(dataset, entity)` pipeline as a chain of `subprocess.run([...], check=False)` with `env = os.environ.copy()` plus `DATASET_NAME`/`DATASET_ENTITY`/`CUDA_VISIBLE_DEVICES` (run.py:241-264). Per-script results append to `timings.csv` (run.py:271-273). There is **no skip-if-exists** guard — re-running retrains from scratch and overwrites checkpoints.

**Re-launch after reboot (no launcher change strictly required):** re-run the *same* command with `RESUME=1` exported. `env.copy()` propagates it to every stage1/stage2 subprocess, which each continue from their pointer-referenced checkpoint. Because `deterministic` is coupled to `resume` (§4.1), the resumed subprocess also runs Tier S.

**Recommended launcher hardening (small, optional):**
- **Stage-level idempotency via a sentinel.** At the end of `main()` write an atomic `.train_complete` sentinel in `ckpt_dir` **after** the token cache (stage1) is durably committed. Before launching stage1/stage2 for an entity, skip the subprocess if the sentinel exists **and** (stage1) the token cache passes its checksum. With `RESUME=1` + sentinel, a finished stage is neither retrained nor resumed; a crash *inside* the cache write leaves no sentinel → stage1 regenerates the deterministic cache from the frozen `best.ckpt` before stage2.
- **best.ckpt selection** must use load+validate, not bare `.exists()` (stage1.py:621; stage2.py:692; `best_checkpoint`, utils.py:46-58): a 0-byte/torn `best.ckpt` satisfies `.exists()` and would feed garbage frozen weights into stage2. Route selection through `_load_one` (checksum) and fall back to `last`/the pointer slot on failure.
- **timings.csv** `[minor]`: rows accumulate across re-runs; either add a `(dataset,entity,script,start_iso)` key or tolerate duplicates (nothing reads it back).
- **Concurrency guard** `[minor]`: take an exclusive `flock` in `ckpt_dir` at `main()` start so a second pipeline for the same entity refuses to run rather than racing the checkpoint files; clean stale `*.tmp` on startup.

With `RESUME` unset the launcher path is byte-identical to today.

---

## 7. Federated repo: extra state

The `-Federated` repo currently contains **no federated training code**: `run.py`, `pipeline/stage1.py`, `pipeline/stage2.py`, `utils.py`, and `config.py` are logically identical to `-Real` (it even carries the same `stage{1,2}_min_epochs=15`, config.py:209-210). The "federated" content is a spec only (no `flwr`/FedAvg/client/round/server code). **Today:** apply the exact same patch — **zero extra state**, it only needs the single-process stage1/stage2 payload of §2.

**Forward-looking note (when FL lands per the spec):** a crash-safe *federated* resume additionally needs, per round `r`: the round index; the **silo-selection RNG** state (which silos participate); for any silo whose local training was in-flight, its full per-silo `{model, optimizer, scheduler, RNG, step, epoch}` — i.e. recursively the §2 payload; and the **server aggregation accumulator** with the **set of silos already folded into the current round's average** (to avoid double-counting on resume). For Stage1 the server-side accumulator is the per-code EMA sufficient statistics `{(N_j, M_j)}` accumulated across silos *before* the codebook overwrite `e_j ← M_j/N_j`; for Stage2 it is the running FedAvg weight accumulator of the shared prior. The §3 atomic-write + ping-pong + checksum machinery is reused for the server checkpoint. None of this exists yet; the per-stage payload is forward-compatible (a silo-local checkpoint *is* the §2 payload).

---

## 8. Surgical diff plan (smallest footprint; default OFF = byte-identical to today)

With `RESUME` unset and `deterministic=False`, the only behavioral change is checkpoint writes going through `os.replace` (atomic) instead of in-place — strictly safer, identical contents. To make it *literally* byte-identical when off, gate the rich payload: write the legacy 4-key dict when `not resume`. **Recommended:** always write the rich payload (cost negligible; enables resuming a run that didn't *start* with `RESUME=1`), and keep `best.ckpt` legacy 4-key.

### `utils.py` (add functions; touch nothing existing). Add `import os, io, hashlib, uuid`; `torch` already imported lazily.
```
+ atomic_torch_save(payload, path)              # §3.1  body sha256 + temp+fsync+replace+dir-fsync
+ atomic_write_text(path, text)                 # §3.5  same primitive for metrics.csv
+ _fsync_dir(d)                                 # §3.1
+ save_resumable(ckpt_dir, payload)             # §3.3  ping-pong + CURRENT pointer
+ load_resumable(ckpt_dir) -> dict | None       # §3.3 / §3.2  pointer-preferring, checksum-verified
+ _load_one(p) / _ckpt_complete(ck)             # §3.2
+ _read_pointer / _write_pointer                # §3.3
+ snapshot_rng(device) / restore_rng(rng, device)  # §2 / §2.1  active-device CUDA RNG
+ build_payload(...) / _assert_fingerprint(have, want)  # §2.2 / §4.4
```

### `config.py` (`TrainingConfig`, after `stage2_patience_steps`, ~line 238)
```
+ resume: bool = False
+ deterministic: bool = False
```

### `pipeline/stage1.py`
```
+ import os, random                              # currently absent
~ :330-331  determinism block (Tier S/P gate)    # §4.2
~ :393      keep AdamW(model.parameters())       # capture opt.state_dict in payload
+ :~417     add explicit train_gen + pass to loader (or reseed per epoch)  # §5.1
+ near top of main(): resume / deterministic flags  # §4.1
+ :~452 (after :418, before :456) resume restore block  # §4.4
~ :456-457  step = start_step; epoch = start_epoch
~ :576-588  REORDER: best-update first, then RNG snapshot, then atomic metrics + save_resumable(epoch+1)  # §4.3 (BLOCKER fix)
~ :582      best.ckpt via atomic legacy 4-key write   # §3.3
+ end of main(): atomic token cache + .train_complete sentinel after cache durable  # §3.5/§6
~ :621      best selection via load+validate not .exists()  # §6
```

### `pipeline/stage2.py` — mirror exactly
```
+ import os, random
~ :443-444  determinism block
~ :533-537  keep AdamW(model.prior.parameters())
+ :~557     explicit train_gen + pass to token/raw loader  # §5.1
+ resume / deterministic flags; fingerprint adds use_cache + weight_decay
+ :~594 (after :564, before :598) resume restore block
~ :598-599  step / epoch from start_*
~ :667-679  REORDER: best-update first, then RNG snapshot, then atomic metrics + save_resumable(epoch+1)  # §4.3
~ :673      best.ckpt atomic legacy 4-key
~ :692      best selection via load+validate
```

### `data.py`
```
~ :1072 _build_loader(..., generator=None) -> DataLoader(..., generator=generator, worker_init_fn=_wif)  # §5
~ :1170-1172 pass train generator into the train loader build
~ :1305 _loader(..., generator=None) -> same
~ :1254 encode_and_cache_tokens: atomic_torch_save(payload, cache_path)  # §3.5
~ :1285 PrecomputedTokenDataModule.setup: verify cache checksum; regenerate/fallback on failure  # §3.5
+ _wif(worker_id): deterministic per-worker seed from (cfg.seed, epoch, worker_id)  # §5.2
```

### `run.py` (optional, §6)
```
+ skip stage1/stage2 subprocess when .train_complete present (and stage1 cache valid)
+ (minor) timings.csv dedup; flock guard
```

**Footprint:** ~12 new `utils.py` functions, 2 config fields, ~30 changed/added lines per stage file, ~6 lines in `data.py`. Default OFF = byte-identical run (verified by Test 6).

---

## 9. Validation plan proving equivalence

Run on the **same GPU + driver + torch + cuDNN build** (enforced by the §2.2 fingerprint, not assumed). Use a tiny fast config (small entity, low `max_steps`, `check_val_every_n_epoch=1`). Run **`num_workers=4`** (production default) to exercise the worker path. Use `STRICT_BITEXACT=1` for bit-exact assertions; Tier S (`warn_only=True`) for trajectory-identical assertions; Tier P only for the explicit relaxation test.

**Test 1 — Golden trajectory.** Run stage1 to completion with `RESUME=1 STRICT_BITEXACT=1`. At **every** epoch boundary dump a fingerprint: SHA of each weight tensor; AdamW `exp_avg`/`exp_avg_sq`/per-param `step` SHAs; `sched.last_epoch` and `lr`; SHAs of `random.getstate()`, `torch.get_rng_state()`, active-device `torch.cuda.get_rng_state()`; the **batch-index permutation** of the epoch; `train/loss`, `val/loss`, `best_val`, `best_step`. Save `metrics.csv` and final `best.ckpt`.

**Test 2 — `kill -9` at an epoch boundary.** Instrument a `KILL_AFTER_EPOCH=N` hook that `os._exit(137)` immediately after `save_resumable` for epoch N commits (simulates power loss with a committed checkpoint).

**Test 3 — `kill -9` inside the write path.** Debug hooks that `os._exit` (a) after the tmp fsync but before `os.replace`, (b) after `os.replace` but before dir-fsync, (c) after the new slot commits but before the `CURRENT` pointer flips. Assert: `load_resumable` always returns a complete, checksum-valid checkpoint (the pointed slot in (c) is still the *old* good one); a leftover `*.tmp` or torn `CURRENT` is ignored. Also crash inside the metrics.csv and token_cache.pt atomic writes → assert the previous committed file survives and the cache checksum rejects a torn cache.

**Test 4 — Resume equivalence (core proof).** Re-launch Test 2 with `RESUME=1 STRICT_BITEXACT=1`. At **every** boundary ≥ N assert vs golden:
1. **Weights** — per-tensor SHA identical.
2. **VQ EMA** — `ema_cluster_size`, `ema_embed_sum`, `initialized` identical; `initialized==True` (no spurious `_kmeans_init`).
3. **Optimizer moments** — `exp_avg`/`exp_avg_sq`/per-param `step` identical.
4. **Scheduler/LR** — `sched.last_epoch == step`; `lr` column identical.
5. **RNG** — python/torch-cpu/active-cuda RNG SHAs identical at the boundary.
6. **DataLoader order** — the resumed epoch-N batch permutation equals golden's (single most load-bearing check given `shuffle=True`).
7. **Loss/val curves** — `metrics.csv` rows N..end byte-identical; **no duplicate or missing epoch rows** (proves verbatim `csv_rows` restore).
8. **Early stop** — resumed run stops at the **same** step (proves the §4.3 best-anchor ordering fix; this would FAIL against the un-fixed save site).
9. **Final `best.ckpt`** — byte-identical.

**Test 5 — Stage2.** Repeat 1–4 for stage2; additionally assert **Python-RNG continuity** (prior `random.random()` masking) and run twice — once with the token cache present, once deleted — confirming the `use_cache` fingerprint assert fires when the cache is removed between crash and resume, and that a re-encoded cache is checksum-valid.

**Test 6 — Guard rails & default-OFF.**
- Shrink/delete the dataset between crash and resume → fingerprint asserts raise a clear error (no silent divergence).
- Resume against an old 4-key or torn checkpoint → `_load_one` rejects it (checksum/format), falls back to the other slot or starts fresh with a printed warning.
- `RESUME` unset → diff `metrics.csv` SHAs and final-checkpoint SHAs against a pre-patch `git stash` baseline; must be identical (proves opt-in default is non-disruptive).
- Version/device fingerprint mismatch (simulate by editing the stored `torch_version`) → hard error under `deterministic`.

**Test 7 — Cross-repo.** Apply to all four; run Test 4 (stage1) + Test 5 (stage2) per repo. For `-M` assert the `getattr(..., "stage{1,2}_min_epochs", 0)` path yields effective `0`, the fingerprint stores `0`, and resume is consistent.

**Test 8 — Tier-P honesty.** Run Test 4 with `ALLOW_NONDETERMINISTIC_RESUME=1` on GPU and **show** it diverges (some boundary fails bit-exact), documenting that Tier P does not satisfy the contract.

**Pass criterion:** `STRICT_BITEXACT` → bit-identical weights/opt/RNG/loss from the resumed epoch onward, identical early-stop step, complete uncorrupted `metrics.csv`. Tier S (`warn_only=True`) → identical *trajectory* (same shuffle permutations, same early-stop step, `allclose(atol=1e-6)` numerics) with FP divergence confined to the iSTFT/interpolate path.

---

## 10. Risks & residual non-equivalence (honest)

1. **iSTFT / decoder-interpolate nondeterminism on CUDA.** Under Tier S with `warn_only=True`, `torch.istft` (transforms.py:63) and `F.interpolate` backward run nondeterministic kernels on the training hot path. Via the `cdist→argmin→ema→dead-code-RNG` mechanism (§1) this can, in principle, fork the RNG stream. **Mitigation:** `STRICT_BITEXACT=1` forces these to CPU + `warn_only=False`. **Residual:** with the practical `warn_only=True` default, equivalence is "trajectory-identical with bounded FP noise in that path", not literal bit-exactness. This is the single irreducible GPU caveat; it is fully removed only by paying the CPU-iSTFT cost.

2. **Cross-environment resume.** A different torch/CUDA/cuDNN/GPU between crash and resume breaks bit-exactness even with perfect state restore (kernel selection, AdamW state schema across torch minors). **Mitigation:** the §2.2 fingerprint hard-errors on mismatch under `deterministic`. **Residual:** the researcher must resume on the same image/GPU; this is enforced, not assumed.

3. **Worker RNG fragility.** Today `__getitem__` is RNG-free (audited), so worker base-seed divergence is irrelevant. **Residual:** if a future change adds augmentation randomness in `__getitem__`/`collate`, the `worker_init_fn` (§5.2) covers it only if the new code reads the worker seed rather than the global stream; a code review gate is required when touching the dataset.

4. **Mid-epoch loss accepted by contract.** The in-flight epoch's partial progress is discarded by design. This is a *deliberate* deviation from a hypothetical mid-step resume, consistent with the stated acceptance.

5. **Filesystem assumptions.** `os.replace` atomicity and dir-fsync durability are correct for the verified ext4 mount. On NFS/overlay (clusters) `os.replace` may be non-atomic or `EXDEV`. **Mitigation/Residual:** `[minor]` startup asserts local FS and documents "local ext4/xfs only"; on network FS the checksum-verified ping-pong still avoids loading a torn primary but cannot guarantee the one-epoch-loss bound.

6. **Diagnostics not restored.** Profiler windows (TVQ_PROFILE), `vram.csv`, and snapshot PNGs are diagnostic-only and intentionally not checkpointed; with `TVQ_PROFILE=1` the sampled trace differs post-resume (the sampled window targets low step indices but `step` resumes high). **Residual:** none for the training trajectory; recommend disabling `TVQ_PROFILE` on resumed runs. `[minor]`

7. **`best.ckpt` propagation under Tier P only.** If a researcher knowingly runs Tier P, FP noise near the `early_stopping_min_delta` threshold can flip a best-val comparison and hand a *different* frozen stage1 to stage2 — a pipeline-level divergence invisible to stage1's "correct" last-checkpoint resume. Under Tier S this cannot happen (val/loss is bit-reproducible). **Residual:** Tier P is explicitly out of the equivalence contract (§1, Test 8).

---

**Key source anchors for the implementer:** payload/save `stage1.py:207-214, 576-588` / `stage2.py:407-414, 667-679` (best-anchor ordering BLOCKER at the 576/580 and 667/671 gap); loop inits `stage1.py:447-457, 500, 610` / `stage2.py:589-599, 628, 681`; reseed `stage1.py:418` / `stage2.py:564`; `benchmark`/`matmul` `stage1.py:330-331` / `stage2.py:443-444`; `seed_everything` `utils.py:107-116`; VQ EMA buffers `vector_quantizer.py:105-107,114`; dead-code RNG `vector_quantizer.py:69-75,152,157`; prior masking RNG `model/prior.py:281-286,491-526,713-745,769-771`; iterative-decode RNG (NOT in training loop) `prior.py:167,183`; loaders `data.py:1072-1081, 1170-1172, 1305-1314`; token cache `data.py:1203-1256, 1285-1301`; `min_epochs` `config.py:226-227` (absent in `-M`); early-stop fields `config.py:234-238`; launcher subprocess/env `run.py:241-273`.
---

## 11. Integrazione con la telemetria GPU (`log_gpu_peak` / `logs/vram.csv`)

Questa sezione fonde il design del resume con la telemetria VRAM, perché i due interventi vanno applicati **insieme, nello stesso commit, su tutti e quattro i repo**, così restano coerenti.

**Stato attuale (sessione corrente):** la telemetria GPU (`utils.log_gpu_peak` → `logs/vram.csv`) e i profili batch per-dataset (`_BATCH_PROFILES` in `config.py`) esistono **solo in `-Real`**. `-M`, `-Fall`, `-Federated` non hanno né l'una né gli altri.

### 11.1 Cosa portare su `-M`, `-Fall`, `-Federated` insieme alla patch resume

| Componente | Portare ovunque? | Note |
|---|---|---|
| `utils.log_gpu_peak()` | **Sì**, identico | Diagnostico puro, non tocca il training. Banale, basso rischio. |
| 3 call-site (`stage1` fine, `stage2` fine, `detect` dopo forward) | **Sì**, identici | Stessi punti di -Real. |
| `_BATCH_PROFILES` (mappa smap/smd/msl) | **No, non i valori** | Tarati per Quadro RTX 8000 + quei dataset. Portare lo **scaffold** (`DATASET_BATCH_PROFILE`) ma con mappa per-repo (vuota di default su -Fall/-Federated se i dataset/GPU differiscono). |

Lo scaffold telemetria è ortogonale al resume: nessuno dei suoi elementi è stato load-bearing (vedi §3.6, §10.6 — `vram.csv` resta non-atomico e **non** ripristinato dal checkpoint). Quindi non introduce nuovi vincoli di equivalenza.

### 11.2 Upgrade: picco VRAM **per epoca** (utile col resume)

Oggi `log_gpu_peak` è chiamato **una volta** a fine training. Con run lunghe e resumabili conviene loggare il picco **per epoca**:

- All'inizio di ogni epoca: `torch.cuda.reset_peak_memory_stats(device)`.
- A fine epoca, accanto al save del checkpoint (§4.3): `log_gpu_peak(f"{stage}", batch_size, cfg)` con una colonna `epoch` aggiunta a `vram.csv`.

Questo dà la **curva VRAM per epoca** — esattamente il dato che serve per ritarare `batch_size_stage1/stage2/eval` in futuro, come da richiesta originale. **Resta diagnostico:** non entra nel payload come stato da ripristinare. Opzionale: aggiungere al payload un campo **non-assertito** `diag_vram_peak_gib` (solo per ispezione del checkpoint, mai confrontato in `_assert_fingerprint`).

### 11.3 Interazione con i tier di determinismo

Il Tier S (TF32 off, `cudnn.deterministic=True`) cambia kernel e quindi **leggermente il consumo/throughput**: i numeri `vram.csv` sotto Tier S **non** sono confrontabili 1:1 con quelli Tier P. **Aggiungere una colonna `deterministic` (e `matmul_precision`) a `vram.csv`** così il tuning del batch tiene conto del tier sotto cui è stato misurato. (La riga `vram.csv` di -Real attuale è pre-resume: dopo la patch va versionata l'intestazione.)

### 11.4 Ordine di applicazione consigliato (per repo)

1. **Telemetria prima** (innocua, non cambia la traiettoria di training): `log_gpu_peak` + 3 chiamate + colonne `epoch`/`deterministic`.
2. **Resume poi** (§8): è la patch che tocca la loop e introduce i tier di determinismo.

Entrambe con default OFF/diagnostico ⇒ con `RESUME` spento e `deterministic=False` il comportamento resta byte-identico a oggi (telemetria a parte, che scrive solo un CSV).

### 11.5 Nota pratica sul run smd attualmente in corso (NON resumabile)

⚠️ Il training `smd:pooled` in esecuzione adesso è stato lanciato **prima** di questa patch: gira in **Tier P** (`cudnn.benchmark=True`) e i suoi checkpoint sono il **vecchio formato a 4 chiavi** (solo `state_dict/cfg_dict/step/epoch`), senza optimizer/scheduler/RNG. Di conseguenza:

- **Non può essere reso resumabile retroattivamente** in modo equivalente. Se la macchina si spegne ora, l'epoch 0 (le 6h già fatte) è perso e si riparte da zero — esattamente lo scenario che questo design previene **per i run futuri**.
- Questo design si applica ai **prossimi** lanci. Per coprire subito smd/smap/msl conviene: far finire (o fermare) il run attuale, applicare la patch, e **rilanciare con `RESUME=1`** (che attiva anche il Tier S). Da quel momento ogni crash costa al massimo l'epoca in corso.

---

## 12. Riepilogo operativo (cosa implementare, in ordine)

1. **`utils.py`** (×4 repo): aggiungi `atomic_torch_save`, `atomic_write_text`, `_fsync_dir`, `save_resumable`, `load_resumable`, `_load_one`, `_ckpt_complete`, `_read_pointer`/`_write_pointer`, `snapshot_rng`/`restore_rng`, `build_payload`, `_assert_fingerprint` (§8) **+** porta `log_gpu_peak` dove manca (§11.1).
2. **`config.py`** (×4): `resume: bool=False`, `deterministic: bool=False` (§4.1); scaffold `_BATCH_PROFILES` per-repo (§11.1).
3. **`pipeline/stage1.py` / `stage2.py`** (×4): blocco determinismo (§4.2), blocco restore (§4.4), fix ordering best-anchor + save_resumable a `epoch+1` (§4.3 — il **BLOCKER**), `import os, random`, `reset_peak_memory_stats` per epoca + `log_gpu_peak` per epoca (§11.2).
4. **`data.py`** (×4): `generator=` esplicito nei loader + `worker_init_fn` (§5), token cache atomico + checksum (§3.5).
5. **`run.py`** (opzionale, ×4): sentinel `.train_complete` + selezione `best.ckpt` via load+validate (§6).
6. **Validazione** (§9): Test 1–8 su un'entità piccola, `num_workers=4`, `STRICT_BITEXACT=1` per le asserzioni bit-exact. Il criterio chiave è **Test 4.8** (early-stop allo stesso step) che prova il fix del BLOCKER.

**Footprint per repo:** ~12 funzioni nuove in `utils.py`, 2 campi config, ~30 righe per file-stage, ~6 righe in `data.py`. Default OFF = run byte-identico a oggi.

---

## 13. Implementation status (applied to all four repos)

Implemented and syntax-checked (`/usr/bin/python3.10 -m py_compile`, **no GPU executed**) in
`-M`, `-Fall`, `-Federated`, `-Real`. Default OFF (`RESUME` unset, `deterministic=False`) ⇒ behavior
is unchanged: no extra checkpoints are written and the determinism tier is today's `cudnn.benchmark=True`.

**Done:**
- `utils.py`: `atomic_torch_save`, `atomic_write_text`, `_fsync_dir`, ping-pong `save_resumable`/`load_resumable`
  (+`_load_one`/`_ckpt_complete`/`_read_pointer`/`_write_pointer`), `snapshot_rng`/`restore_rng`,
  `build_payload`, `build_fingerprint`, `_assert_fingerprint`, `apply_determinism`. Plus `log_gpu_peak`
  ported to the three repos that lacked it.
- `config.py`: `resume: bool = False`, `deterministic: bool = False` in `TrainingConfig`.
- `pipeline/stage1.py` & `pipeline/stage2.py`: determinism tier block (replaces unconditional
  `benchmark=True`); atomic `_write_metrics_csv`; full-state restore block (weights+EMA+optimizer+scheduler+
  RNG+early-stop anchors+csv) guarded by `_assert_fingerprint`; `step/epoch` from the restored values;
  the **§4.3 BLOCKER fix** — `save_resumable(... epoch+1 ...)` written AFTER the `best_val/best_step`
  update — gated behind `if resume:`; end-of-stage `log_gpu_peak`.
- `pipeline/detect.py`: eval-pass `log_gpu_peak` (3 repos that lacked it).

**Deferred (documented; NOT required for epoch-boundary equivalence):**
- **`data.py` explicit shuffle generator** (§5.1). Not implemented because `restore_rng` already restores
  the torch-CPU RNG that drives the default `RandomSampler`, and `restore_rng` is the last statement before
  the loop, so the resumed epoch draws the identical permutation. The explicit generator is robustness
  hardening against *future* incidental global-RNG use between restore and the first batch; add it if that
  invariant ever changes.
- **Atomic + checksummed `token_cache.pt`** (§3.5). Wrapping it in `atomic_torch_save` changes the file
  format and would require updating every token-cache loader; left as a follow-up. The cache is regenerable
  from the frozen stage-1 `best.ckpt`, so a torn cache is recoverable by deletion + rerun.
- **Per-epoch VRAM logging** (§11.2). The existing end-of-stage `log_gpu_peak` already gives the whole-run
  peak needed for batch tuning; per-epoch logging + a `vram.csv` schema bump is a small future enhancement.
- **`best.ckpt` / `run.py` sentinel hardening** (§6). `best.ckpt` is still the legacy 4-key write and is
  selected via `best_checkpoint()`; the launcher `.train_complete` sentinel was not added.

**How to use it (the real pooled runs):** start the run **with** `RESUME=1` so it is Tier-S from epoch 0
and writes resume checkpoints:
```bash
RESUME=1 python run.py --datasets smd:pooled,smap:pooled,msl:pooled --mode research -w 1 -g 0
```
After a crash/power loss, **re-run the identical command** (with `RESUME=1`): each stage discovers its
pointer-referenced checkpoint, asserts the fingerprint (same cfg/env), and continues from the last committed
epoch boundary — losing only the in-flight epoch. A run started WITHOUT `RESUME=1` is Tier-P and writes no
resume checkpoint, so it cannot be resumed equivalently (by design); the currently-running smd job is in
this category. Use `STRICT_BITEXACT=1` for the validation runs (§9) that prove bit-exact equivalence.
