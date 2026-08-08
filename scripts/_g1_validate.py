"""Validazione di g1 come nodo di calcolo. Sette controlli, dal piu' economico al piu' caro.

Non basta che `import torch` funzioni. Le domande vere sono: torch 2.11+cu128 emette kernel
per **Volta sm_70**, i tensor core fp16 funzionano lì, e la nostra pipeline gira sul repo
montato via sshfs a velocita' utile. Un venv trapiantato puo' superare i primi tre controlli
e fallire il quarto senza dare errore.
"""
import os
import sys
import time

REPO = "/home/leonardo/mnt/tvq-u"
OUT = {}


def check(name):
    def deco(fn):
        print(f"\n--- {name} ---", flush=True)
        try:
            r = fn()
            OUT[name] = r
            print(f"  OK  {r}", flush=True)
        except Exception as e:                       # noqa: BLE001
            OUT[name] = f"FALLITO: {type(e).__name__}: {e}"
            print(f"  !! FALLITO  {type(e).__name__}: {e}", flush=True)
        return fn
    return deco


@check("1. torch importa")
def _():
    import torch
    return f"torch {torch.__version__}  cuda-runtime {torch.version.cuda}"


@check("2. CUDA vede le schede")
def _():
    import torch
    assert torch.cuda.is_available(), "cuda non disponibile"
    n = torch.cuda.device_count()
    d = [f"{torch.cuda.get_device_name(i)} sm_{''.join(map(str, torch.cuda.get_device_capability(i)))}"
         for i in range(n)]
    return f"{n} schede: {', '.join(d)}"


@check("3. sm_70 e' fra le architetture compilate")
def _():
    import torch
    arch = torch.cuda.get_arch_list()
    cap = torch.cuda.get_device_capability(0)
    want = f"sm_{cap[0]}{cap[1]}"
    assert want in arch, f"{want} NON in {arch} -- girerebbe in JIT o non girerebbe affatto"
    return f"{want} presente ({len(arch)} arch compilate)"


@check("4. matmul fp32 e fp16 danno il risultato giusto")
def _():
    import torch
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(512, 512, device="cuda", generator=g)
    b = torch.randn(512, 512, device="cuda", generator=g)
    ref = (a.cpu() @ b.cpu())
    err32 = (a @ b).cpu().sub(ref).abs().max().item()
    err16 = (a.half() @ b.half()).float().cpu().sub(ref).abs().max().item()
    assert err32 < 1e-2, f"fp32 sbagliato: {err32}"
    assert err16 < 2.0, f"fp16 sbagliato: {err16}"
    return f"err fp32 {err32:.2e}  ·  err fp16 {err16:.2e}"


@check("5. tensor core fp16: quanto vanno")
def _():
    import torch
    res = {}
    for dt, lab in ((torch.float32, "fp32"), (torch.float16, "fp16")):
        x = torch.randn(4096, 4096, device="cuda", dtype=dt)
        for _ in range(3):
            x @ x
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            x @ x
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / 10 * 1000
        res[lab] = ms
    sp = res["fp32"] / res["fp16"]
    assert sp > 1.5, f"fp16 non accelera (x{sp:.2f}): tensor core assenti o non usati"
    return (f"matmul 4096^3: fp32 {res['fp32']:.1f} ms · fp16 {res['fp16']:.1f} ms "
            f"-> fp16 e' {sp:.1f}x piu' veloce")


@check("6. il repo montato e' leggibile a velocita' utile")
def _():
    assert os.path.isdir(REPO), f"{REPO} non montato"
    t = time.perf_counter()
    n = sum(1 for _ in os.scandir(f"{REPO}/data/raw/ucr_split_w2p"))
    dt = time.perf_counter() - t
    p = f"{REPO}/data/raw/ucr_split_w2p/metadata.json"
    t = time.perf_counter()
    sz = len(open(p, "rb").read())
    dt2 = time.perf_counter() - t
    return f"scandir {n} voci in {dt*1000:.0f} ms · letto metadata.json ({sz} B) in {dt2*1000:.0f} ms"


@check("7. il codice del repo importa (dipendenze complete)")
def _():
    sys.path.insert(0, REPO)
    os.environ.setdefault("FEDVQ_AMP", "fp16")
    import importlib
    mods = ["config", "pipeline.federated", "pipeline.federated_eval", "model.prior_upstream"]
    ok = []
    for m in mods:
        importlib.import_module(m)
        ok.append(m)
    from pipeline.federated import _amp_dtype
    return f"{len(ok)} moduli importati · _amp_dtype() -> {_amp_dtype()}"


print("\n" + "=" * 64)
bad = [k for k, v in OUT.items() if isinstance(v, str) and v.startswith("FALLITO")]
print("ESITO:", "TUTTO OK" if not bad else f"{len(bad)} CONTROLLI FALLITI: {bad}")
print("=" * 64)
sys.exit(1 if bad else 0)
