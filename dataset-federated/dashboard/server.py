"""
Localhost dashboard for the federated *univariate* datasets, served from the shared
pipeline layout at <repo>/data/raw/<dataset>/ (train|val|test|test_label/*.npy +
clusters.json + metadata.json).

MULTI-DATASET: one server process serves EVERY dataset under data/raw that has a
`train/` dir + `clusters.json`, switchable in-page from the header dropdown — no
need to relaunch with a different DASH_DATASET. Discovered on this node:
  - wsd_fed             (real,      31 KPI clients in 4 clusters c0..c3)
  - toy_fed_uni[...]    (synthetic, clients uni_00.. in M*-named clusters)

Every JSON endpoint takes `?ds=<dataset>` (default = DASH_DATASET or wsd_fed).
Client ids are the string entity ids (kpi_015, uni_00); clusters carry both a stable
index (colour / %6) and their real name (display). Provenance is dataset-aware:
WSD windows come from manifest.csv; synthetic events come from metadata.json.

Stdlib-only HTTP server + JSON API + static frontend.

Run:  python dashboard/server.py                    # serves ALL datasets, http://localhost:8765
Env:  DASH_DATASET (default wsd_fed — only the *initial* selection now) · DASH_PORT (default 8765)
"""
import os, re, json, http.server, socketserver, urllib.parse, datetime
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)                        # dataset-federated/
REPO = os.path.dirname(BASE)                        # repo root (…/TimeVQVAE-AD-U-Federated)
RAW_ROOT = os.path.join(REPO, "data", "raw")        # shared pipeline layout for every dataset
WSD_RAW = os.path.join(BASE, "data", "WSD", "real-world")   # wsd_fed only: original CSVs (timestamps)
STATIC = os.path.realpath(os.path.join(HERE, "static"))
PORT = int(os.environ.get("DASH_PORT", "8765"))
DEFAULT_DATASET = os.environ.get("DASH_DATASET", "wsd_fed")


def discover_datasets():
    """Every dir under data/raw that looks like a pipeline dataset (train/ + clusters.json).
    wsd_fed first (sensible default), then the rest alphabetically."""
    out = []
    if os.path.isdir(RAW_ROOT):
        for name in os.listdir(RAW_ROOT):
            d = os.path.join(RAW_ROOT, name)
            if os.path.isdir(os.path.join(d, "train")) and os.path.exists(os.path.join(d, "clusters.json")):
                out.append(name)
    out.sort(key=lambda n: (n != "wsd_fed", n))
    return out


DATASETS = discover_datasets()
if not DATASETS:
    raise SystemExit(f"no datasets found under {RAW_ROOT} (need train/ + clusters.json)")


class Ctx:
    """All per-dataset state (the old module globals), built lazily and cached per name."""
    def __init__(self, name):
        self.name = name
        self.SRC = os.path.join(RAW_ROOT, name)
        self.CLUSTERS = json.load(open(os.path.join(self.SRC, "clusters.json"), encoding="utf-8"))
        self.CLUSTER_NAMES = list(self.CLUSTERS)                 # preserve file order
        self.CIDX = {n: i for i, n in enumerate(self.CLUSTER_NAMES)}
        self.ENTITIES = [e for n in self.CLUSTER_NAMES for e in self.CLUSTERS[n]]
        self.ENT_CLUSTER = {e: n for n in self.CLUSTER_NAMES for e in self.CLUSTERS[n]}

        _mp = os.path.join(self.SRC, "metadata.json")
        self.META = json.load(open(_mp, encoding="utf-8")) if os.path.exists(_mp) else {}
        _el = self.META.get("entities", self.META.get("series", []))
        self.META_BY_EID = {r.get("entity_id", r.get("id")): r for r in _el} if isinstance(_el, list) else {}

        _man = os.path.join(self.SRC, "manifest.csv")           # WSD provenance (optional)
        self.MAN = pd.read_csv(_man) if os.path.exists(_man) else None
        if self.MAN is not None and "series_id" in self.MAN.columns and "id" not in self.MAN.columns:
            self.MAN = self.MAN.rename(columns={"series_id": "id"})

        self.IS_WSD = name == "wsd_fed"
        self.NOUN = "KPI" if self.IS_WSD else "series"
        self.desc = self.META.get("family_description") or self.META.get("description") or name

        _ov = os.path.join(HERE, "overview.json")               # 210-KPI population context is WSD-only
        self.HAS_OVERVIEW = self.IS_WSD and os.path.exists(_ov)
        self.OVERVIEW = json.load(open(_ov, encoding="utf-8")) if self.HAS_OVERVIEW else {"meta": {}, "series": []}

        self._arr_cache, self._ts_cache = {}, {}


_CTX = {}


def get_ctx(name):
    """Cached Ctx for a KNOWN dataset name, or None (guards against arbitrary paths)."""
    if name not in DATASETS:
        return None
    if name not in _CTX:
        _CTX[name] = Ctx(name)
    return _CTX[name]


def _numid(eid):
    m = re.search(r"(\d+)$", str(eid))
    return int(m.group(1)) if m else None


def load_client(ctx, eid):
    """Lazy per-client arrays from the pipeline layout; squeeze (N,1)->(N,), take ch0 if C>1."""
    if eid not in ctx._arr_cache:
        d = {}
        for part in ("train", "val", "test", "test_label"):
            p = os.path.join(ctx.SRC, part, f"{eid}.npy")
            if os.path.exists(p):
                a = np.load(p)
                if a.ndim > 1:
                    a = a.reshape(a.shape[0], -1)[:, 0]     # first channel (univariate here)
            else:
                a = np.zeros((0,), dtype=np.float32)
            d[part] = a
        ctx._arr_cache[eid] = d
    return ctx._arr_cache[eid]


def anomaly_spans(mask):
    mask = np.asarray(mask).astype(bool); out = []; i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]: j += 1
            out.append([int(i), int(j)]); i = j + 1
        else: i += 1
    return out


def _man_row(ctx, eid):
    if ctx.MAN is None: return None
    num = _numid(eid)
    if num is None or "id" not in ctx.MAN.columns: return None
    row = ctx.MAN[ctx.MAN.id == num]
    return None if row.empty else row.iloc[0]


def raw_timestamps(ctx, num):
    if num not in ctx._ts_cache:
        try:
            ctx._ts_cache[num] = pd.read_csv(os.path.join(WSD_RAW, f"{num}.csv"),
                                             usecols=["timestamp"])["timestamp"].values
        except Exception:
            ctx._ts_cache[num] = None
    return ctx._ts_cache[num]


def iso(unix):
    try: return datetime.datetime.utcfromtimestamp(int(unix)).strftime("%Y-%m-%d %H:%M")
    except Exception: return None


def _provenance(ctx, eid, n):
    """Dataset-aware provenance block, tagged by `kind` for the frontend to render."""
    r = _man_row(ctx, eid)
    if r is not None and "win_start" in r and "win_end" in r:          # WSD real windows
        ws, we = int(r.win_start), int(r.win_end)
        on = int(r.orig_n) if "orig_n" in r else n
        num = _numid(eid); ts = raw_timestamps(ctx, num) if num is not None else None
        ts_start = iso(ts[ws]) if ts is not None and len(ts) > ws else None
        ts_end = iso(ts[we - 1]) if ts is not None and we > 0 and len(ts) >= we else None
        return {"kind": "wsd", "orig_n": on, "win_start": ws, "win_end": we,
                "ts_start": ts_start, "ts_end": ts_end,
                "source": f"data/WSD/real-world/{num}.csv" if num is not None else None}
    m = ctx.META_BY_EID.get(eid)
    if m is not None:                                                  # synthetic generator metadata
        evs = [{"start": int(e.get("start", 0)), "stop": int(e.get("stop", 0)),
                "variant": e.get("variant", ""), "description": e.get("description", "")}
               for e in m.get("events", [])]
        return {"kind": "synthetic", "waveform": m.get("waveform"), "period": m.get("period"),
                "base_period": m.get("base_period"), "jitter": m.get("jitter"),
                "variants": m.get("variants", []), "events": evs}
    return {"kind": "none"}


def meta_payload(ctx):
    per = {name: len(ctx.CLUSTERS[name]) for name in ctx.CLUSTER_NAMES}
    return {"label": f"{ctx.name} · {ctx.desc}", "dataset": ctx.name,
            "n_clients": len(ctx.ENTITIES), "n_clusters": len(ctx.CLUSTER_NAMES),
            "clients_per_cluster": per, "entity_noun": ctx.NOUN,
            "has_overview": ctx.HAS_OVERVIEW, "seed": int(ctx.META.get("seed", 0)),
            "source": ctx.desc}


def _client_summary(ctx, eid):
    a = load_client(ctx, eid)
    tr, va, te, tl = a["train"], a["val"], a["test"], a["test_label"]
    test_anom = int(np.asarray(tl).astype(bool).sum())
    test_len = int(len(te))
    return {"id": eid, "numid": _numid(eid),
            "cluster": ctx.CIDX[ctx.ENT_CLUSTER[eid]], "cluster_name": ctx.ENT_CLUSTER[eid],
            "n": int(len(tr) + len(va) + len(te)),
            "train_len": int(len(tr)), "val_len": int(len(va)), "test_len": test_len,
            "test_anom": test_anom,
            "anom_pct": round(100.0 * test_anom / test_len, 2) if test_len else 0.0}


def manifest_payload(ctx):
    tree = []
    for name in ctx.CLUSTER_NAMES:
        clients = [_client_summary(ctx, e) for e in ctx.CLUSTERS[name]]
        clients.sort(key=lambda c: (c["numid"] if c["numid"] is not None else 0, c["id"]))
        tree.append({"cluster": ctx.CIDX[name], "cluster_name": name,
                     "n_clients": len(clients), "clients": clients})
    return {"clusters": tree}


def series_payload(ctx, eid):
    if eid not in ctx.ENT_CLUSTER: return None
    a = load_client(ctx, eid)
    tr, va, te, tl = a["train"], a["val"], a["test"], a["test_label"]
    val_start = int(len(tr)); split = int(len(tr) + len(va)); n = int(split + len(te))
    v = np.concatenate([tr, va, te]).astype(float) if n else np.zeros((0,))
    value = [None if not np.isfinite(x) else round(float(x), 4) for x in v]
    sp = [[s + split, e + split] for s, e in anomaly_spans(np.asarray(tl) == 1)]
    test_anom = int(np.asarray(tl).astype(bool).sum())
    return {"id": eid, "numid": _numid(eid),
            "cluster": ctx.CIDX[ctx.ENT_CLUSTER[eid]], "cluster_name": ctx.ENT_CLUSTER[eid],
            "n": n, "split": split, "split_frac": round(split / n, 4) if n else 0.5,
            "val_start": val_start, "val_len": int(len(va)),
            "train_len": int(len(tr)), "test_len": int(len(te)),
            "test_anom": test_anom, "train_anom": 0,
            "nan_train": int(np.isnan(tr).sum()), "nan_test": int(np.isnan(te).sum()),
            "nan_total": int(np.isnan(v).sum()),
            "prov": _provenance(ctx, eid, n),
            "value": value, "anomaly_spans": sp}


def datasets_payload():
    items = []
    for name in DATASETS:
        c = get_ctx(name)
        items.append({"name": name, "n_clients": len(c.ENTITIES),
                      "n_clusters": len(c.CLUSTER_NAMES), "has_overview": c.HAS_OVERVIEW,
                      "noun": c.NOUN})
    default = DEFAULT_DATASET if DEFAULT_DATASET in DATASETS else DATASETS[0]
    return {"datasets": items, "default": default}


CTYPES = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8", ".json": "application/json"}


class H(http.server.BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)): body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str): body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(body)

    def _static(self, relpath):
        relpath = relpath.lstrip("/")
        full = os.path.realpath(os.path.join(STATIC, relpath))
        if (full != STATIC and not full.startswith(STATIC + os.sep)) or not os.path.isfile(full):
            return self._send(404, {"error": "not found"})
        ext = os.path.splitext(full)[1].lower()
        with open(full, "rb") as f:
            self._send(200, f.read(), CTYPES.get(ext, "application/octet-stream"))

    def _ctx(self, qs):
        """Resolve ?ds=<name> (default = DEFAULT_DATASET) to a Ctx, or None if unknown."""
        name = qs.get("ds", [DEFAULT_DATASET])[0]
        return get_ctx(name)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"): return self._static("index.html")
            if path.startswith("/static/"): return self._static(path[len("/static/"):])
            if path == "/api/datasets": return self._send(200, datasets_payload())

            if path in ("/api/meta", "/api/manifest", "/api/overview") or path.startswith("/api/series/"):
                ctx = self._ctx(qs)
                if ctx is None:
                    return self._send(404, {"error": f"unknown dataset {qs.get('ds', ['?'])[0]!r}"})
                if path == "/api/meta": return self._send(200, meta_payload(ctx))
                if path == "/api/manifest": return self._send(200, manifest_payload(ctx))
                if path == "/api/overview": return self._send(200, ctx.OVERVIEW)
                eid = urllib.parse.unquote(path[len("/api/series/"):])
                s = series_payload(ctx, eid)
                return self._send(200, s) if s else self._send(404, {"error": "unknown series"})

            return self._send(404, {"error": "not found"})
        except Exception as e:
            return self._send(500, {"error": str(e)})

    def log_message(self, *a): pass


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True; allow_reuse_address = True


if __name__ == "__main__":
    print(f"dashboard  ->  http://localhost:{PORT}")
    print(f"  {len(DATASETS)} datasets (switch in-page): {', '.join(DATASETS)}")
    print(f"  default selection: {DEFAULT_DATASET if DEFAULT_DATASET in DATASETS else DATASETS[0]}")
    Server(("127.0.0.1", PORT), H).serve_forever()
