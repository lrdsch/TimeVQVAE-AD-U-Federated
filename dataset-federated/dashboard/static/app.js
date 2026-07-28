"use strict";
// cluster palette (brightened for dark bg) — ties to the earlier cluster plots
const CLC = ["#4f9fe0","#ff9f45","#4fb06a","#a884d8","#c08a6a","#3fc9d6"];
const GRID = "rgba(139,152,167,0.14)", MUT = "#8b98a7";

// shared render state read by the uPlot overlay plugin
const S = { n:0, split:0, valStart:0, spans:[], cluster:0,
            showTrain:true, showTest:true, showAnom:true };
let u = null, cur = null, viewmode = "full", fullLen = false;
let NOUN = "series";                               // entity noun from /api/meta (KPI | series)
let DS = null;                                     // currently-selected dataset (server has many)

const $ = s => document.querySelector(s);
// every JSON endpoint is dataset-scoped: ?ds=<current dataset>
const dsq = (extra) => "?ds=" + encodeURIComponent(DS) + (extra ? "&" + extra : "");

// ---------- uPlot overlay: train/test shading, anomaly bands, split line ----------
// x-coords are centered on sample points: uPlot plots point i AT valToPos(i),
// so region/band boundaries use i-0.5 .. i+0.5 (right-inclusive spans keep width).
function fillRange(u, xa, xb, style, minW){
  const L=u.bbox.left, W=u.bbox.width, T=u.bbox.top, H=u.bbox.height;
  let a=u.valToPos(xa,"x",true), b=u.valToPos(xb,"x",true);
  if(b<=L || a>=L+W) return;                       // fully off-screen: never draw
  a=Math.max(a,L); b=Math.min(b,L+W);
  if(minW && b-a<minW){ const m=(a+b)/2; a=Math.max(m-minW/2,L); b=Math.min(m+minW/2,L+W); }
  if(b<=a) return;
  u.ctx.fillStyle=style; u.ctx.fillRect(a,T,b-a,H);
}
function drawSplit(u){
  const x=u.valToPos(S.split-0.5,"x",true), L=u.bbox.left, W=u.bbox.width, T=u.bbox.top, H=u.bbox.height;
  if(x<L-1||x>L+W+1) return;
  const c=u.ctx; c.save(); c.strokeStyle="#e8b64c"; c.lineWidth=1.5; c.setLineDash([5,4]);
  c.beginPath(); c.moveTo(x,T); c.lineTo(x,T+H); c.stroke(); c.restore();
}
function overlayPlugin(){
  return { hooks:{
    drawClear:[u=>{
      if(S.showTrain){
        fillRange(u,-0.5,S.valStart-0.5,"rgba(70,160,90,0.13)");           // train
        fillRange(u,S.valStart-0.5,S.split-0.5,"rgba(232,182,76,0.16)");   // val
      }
      if(S.showTest)  fillRange(u,S.split-0.5,S.n-0.5,"rgba(120,140,175,0.10)");
    }],
    draw:[u=>{
      if(S.showAnom){                              // merge floored rects so alpha doesn't compound
        const L=u.bbox.left, W=u.bbox.width, T=u.bbox.top, H=u.bbox.height, minW=2, iv=[];
        for(const [s,e] of S.spans){
          let a=u.valToPos(s-0.5,"x",true), b=u.valToPos(e+0.5,"x",true);
          if(b<=L || a>=L+W) continue;
          a=Math.max(a,L); b=Math.min(b,L+W);
          if(b-a<minW){ const m=(a+b)/2; a=Math.max(m-minW/2,L); b=Math.min(m+minW/2,L+W); }
          if(b>a) iv.push([a,b]);
        }
        iv.sort((p,q)=>p[0]-q[0]);
        u.ctx.fillStyle="rgba(224,96,91,0.36)";
        let ca=null, cb=null;
        for(const [a,b] of iv){
          if(ca===null){ ca=a; cb=b; }
          else if(a<=cb){ cb=Math.max(cb,b); }
          else { u.ctx.fillRect(ca,T,cb-ca,H); ca=a; cb=b; }
        }
        if(ca!==null) u.ctx.fillRect(ca,T,cb-ca,H);
      }
      drawSplit(u);
    }],
  }};
}
function wheelZoomPlugin(){
  return { hooks:{ ready:[u=>{
    u.over.addEventListener("wheel", e=>{
      e.preventDefault();
      const r=u.over.getBoundingClientRect();
      const {min,max}=u.scales.x, cx=(e.clientX-r.left)/r.width, xv=min+cx*(max-min);
      const f=e.deltaY<0?0.82:1.22;
      const lo=u.data[0][0], hi=u.data[0][u.data[0].length-1];
      let nmin=Math.max(lo, xv-(xv-min)*f), nmax=Math.min(hi, xv+(max-xv)*f);
      if(nmax-nmin>1) u.setScale("x",{min:nmin,max:nmax});
    }, {passive:false});
  }]}};
}

function chartSize(){
  const cw = (($("#chartwrap").clientWidth) || 1000) - 18;      // stable container (visible) width
  const h = Math.max(320, Math.min(620, Math.round(window.innerHeight*0.56)));
  // full-length: print the whole series long (~4 samples/px), scroll horizontally
  const w = (fullLen && cur) ? Math.min(12000, Math.max(cw, Math.round(cur.n/4))) : cw;
  return { width:w, height:h };
}

function render(series){
  cur = series;
  S.n = series.n; S.split = series.split; S.valStart = series.val_start ?? series.split;
  S.spans = series.anomaly_spans; S.cluster = series.cluster;
  const xs = Array.from({length:series.n}, (_,i)=>i);
  const ys = series.value;
  const {width,height} = chartSize();
  if(u){ u.destroy(); u=null; }
  const opts = {
    width, height,
    scales:{ x:{ time:false }, y:{} },
    cursor:{ drag:{x:true,y:false}, y:false, points:{show:false} },
    legend:{ live:true },
    series:[
      { label:"idx" },
      { label:"value", stroke:CLC[series.cluster%6], width:1, points:{show:false}, spanGaps:false },
    ],
    axes:[
      { stroke:MUT, grid:{stroke:GRID}, ticks:{stroke:GRID}, font:"11px ui-monospace, monospace" },
      { stroke:MUT, grid:{stroke:GRID}, ticks:{stroke:GRID}, font:"11px ui-monospace, monospace",
        size:58 },
    ],
    plugins:[ overlayPlugin(), wheelZoomPlugin() ],
    hooks:{ setScale:[(uu,key)=>{ if(key==="x") syncSeg(uu); }] },
  };
  u = new uPlot(opts, [xs, ys], $("#chart"));
  applyView();
}

function applyView(){
  if(!u || !cur) return;
  if(viewmode==="train")      u.setScale("x",{min:0, max:cur.split});
  else if(viewmode==="test")  u.setScale("x",{min:cur.split, max:cur.n-1});
  else                        u.setScale("x",{min:0, max:cur.n-1});
}
// keep the full/train/test control in sync with the ACTUAL x-scale after wheel/drag/dblclick zoom
function syncSeg(uu){
  if(!cur) return;
  const {min,max}=uu.scales.x; let v="";
  if(min===0 && max===cur.split)            v="train";
  else if(min===cur.split && max===cur.n-1) v="test";
  else if(min===0 && max===cur.n-1)         v="full";
  viewmode = v || "custom";
  setSeg(v);                                    // "" => no preset button lit (custom zoom)
}

// ---------- info panel ----------
function infoCard(k,v){ return `<div class="card"><div class="k">${k}</div><div class="val">${v}</div></div>`; }
// dataset-aware provenance: WSD real windows vs synthetic generator metadata vs none
function provCard(s){
  const p = s.prov || {kind:"none"};
  if(p.kind==="wsd"){
    return `<div class="card prov"><div class="k">provenance</div>
      Real WSD KPI — original <code>${p.source||"series"}</code>${p.orig_n?` = ${p.orig_n.toLocaleString()} pts`:""}.
      This client is the longest <b>NaN-free window</b> whose first half is anomaly-free,
      <code>original[${p.win_start}:${p.win_end}]</code> (${s.n.toLocaleString()} pts).
      <code>train = window[:${s.val_start}]</code> and <code>val = window[${s.val_start}:${s.split}]</code>
      are both clean (anomaly- &amp; NaN-free) · <code>test = window[${s.split}:]</code> holds
      ${s.test_anom} anomalies. <b>val</b> is for model selection — never touch test.</div>`;
  }
  if(p.kind==="synthetic"){
    const off = s.split||0;   // metadata events are test-relative; shift to the plotted series index
    const ev = (p.events||[]).map(e=>`<code>${off+e.start}–${off+e.stop}</code> ${e.variant}`).join(" · ") || "none";
    const vr = (p.variants&&p.variants.length)?`, variants ${p.variants.join(", ")}`:"";
    return `<div class="card prov"><div class="k">provenance</div>
      Synthetic univariate client — waveform <b>${p.waveform||"?"}</b>, period <b>${p.period??"?"}</b>,
      jitter <b>${p.jitter||"?"}</b>${vr}.
      <code>train = [:${s.val_start}]</code> · <code>val = [${s.val_start}:${s.split}]</code> (model selection) ·
      <code>test = [${s.split}:]</code> holds ${s.test_anom} anomaly points.
      Injected events (absolute index): ${ev}.</div>`;
  }
  return `<div class="card prov"><div class="k">provenance</div>
    <code>train = [:${s.val_start}]</code> · <code>val = [${s.val_start}:${s.split}]</code> ·
    <code>test = [${s.split}:]</code> — ${s.test_anom} anomaly points in test.</div>`;
}
function renderInfo(s){
  const anomPct = s.test_len ? (100*s.test_anom/s.test_len).toFixed(2) : "0.00";
  const ts = (s.prov&&s.prov.kind==="wsd")
    ? infoCard("timespan", (s.prov.ts_start||"?")+" → "+(s.prov.ts_end||"?")) : "";
  $("#info").innerHTML =
    infoCard("cluster", s.cluster_name||("c"+s.cluster)) +
    infoCard("total length", s.n.toLocaleString()) +
    infoCard("split index", s.split.toLocaleString()+" ("+(s.split_frac*100).toFixed(0)+"%)") +
    infoCard("train length", s.train_len.toLocaleString()+" · 0 anom") +
    infoCard("val length", (s.val_len||0).toLocaleString()+" · 0 anom") +
    infoCard("test length", s.test_len.toLocaleString()) +
    infoCard("test anomalies", s.test_anom+" ("+anomPct+"%)") +
    infoCard("missing (NaN)", (s.nan_total||0)+" · "+(s.nan_train||0)+" train / "+(s.nan_test||0)+" test") +
    ts + provCard(s);
}

// ---------- data loading ----------
async function loadSeries(id, el){
  document.querySelectorAll(".client.active").forEach(e=>e.classList.remove("active"));
  if(el) el.classList.add("active");
  if(location.hash !== "#"+id) history.replaceState(null,"","#"+id);   // deep-link: /#143
  let s;
  try {
    const resp = await fetch("/api/series/"+id+dsq());
    s = await resp.json();                          // may throw on invalid JSON
    if(!resp.ok || s.error) throw new Error(s.error || ("HTTP "+resp.status));
  } catch(err){
    if(u){ u.destroy(); u=null; }
    $("#toolbar").style.display="none";
    $("#chartwrap").style.display="none";
    const empty=$("#empty"); empty.style.display="";
    empty.textContent="Failed to load KPI "+id+": "+err.message;
    return;
  }
  $("#empty").style.display="none";
  $("#toolbar").style.display="flex";
  $("#chartwrap").style.display="block";
  $("#cliTitle").innerHTML = `${NOUN} ${s.id} <span class="sub">${s.cluster_name||("cluster "+s.cluster)} · ${s.n.toLocaleString()} pts</span>`;
  render(s);
  renderInfo(s);
}

// ---------- sidebar tree ----------
let TREE = null;
function buildTree(data){
  TREE = data;
  const tree = $("#tree"); tree.innerHTML="";
  data.clusters.forEach(cl=>{
    const folder=document.createElement("div"); folder.className="folder"; folder.dataset.cluster=cl.cluster;
    const head=document.createElement("div"); head.className="fhead";
    head.innerHTML=`<span class="tw">▾</span><span class="dot" style="background:${CLC[cl.cluster%6]}"></span>`+
                   `<span class="nm">${cl.cluster_name||("cluster "+cl.cluster)}</span><span class="ct">${cl.n_clients}</span>`;
    head.onclick=()=>folder.classList.toggle("collapsed");
    const box=document.createElement("div"); box.className="clients";
    cl.clients.forEach(c=>{
      const row=document.createElement("div"); row.className="client"; row.dataset.id=c.id;
      if(c.numid!=null) row.dataset.numid=c.numid;
      const hot=c.anom_pct>=3?" hot":"";
      row.innerHTML=`<span class="id">${NOUN} ${c.id}</span><span class="badge${hot}">${c.anom_pct}% · ${c.test_anom}a</span>`;
      row.onclick=()=>loadSeries(c.id,row);
      box.appendChild(row);
    });
    folder.appendChild(head); folder.appendChild(box); tree.appendChild(folder);
  });
  const want = location.hash.slice(1);                  // open /#uni_00 or /#143 straight on that client
  if(want && want!=="overview"){
    const row = tree.querySelector(`.client[data-id="${want}"]`);
    if(row) loadSeries(want, row);
  }
}
function applySearch(q){
  q=q.trim();
  let anyVisible=false;
  document.querySelectorAll(".folder").forEach(f=>{
    let any=false;
    f.querySelectorAll(".client").forEach(c=>{
      const show = !q || c.dataset.id.includes(q);
      c.style.display=show?"":"none"; any=any||show;
    });
    f.style.display=any?"":"none";
    if(q) f.classList.remove("collapsed");
    anyVisible=anyVisible||any;
  });
  let msg=document.querySelector("#noResults");
  if(!anyVisible && q){
    if(!msg){ msg=document.createElement("div"); msg.id="noResults"; msg.className="empty";
              msg.style.padding="20px"; $("#tree").appendChild(msg); }
    msg.textContent=`no KPIs match "${q}"`; msg.style.display="";
  } else if(msg){ msg.style.display="none"; }
}

// ---------- overview (all 210 in context) ----------
let OV = null, ovFilter = "all";
function switchTab(tab){
  // overview exists for wsd only; if its tab is hidden (toy datasets), fall back to
  // detail so a stale #overview deep-link can't render an empty overview and throw.
  const ovBtn = document.querySelector('#tabs button[data-tab="overview"]');
  if(tab==="overview" && ovBtn && ovBtn.style.display==="none") tab="detail";
  document.querySelectorAll("#tabs button").forEach(b=>b.classList.toggle("on", b.dataset.tab===tab));
  $("#detailView").style.display = tab==="detail" ? "" : "none";
  $("#overviewView").style.display = tab==="overview" ? "" : "none";
  if(tab==="overview") loadOverview();
  else if(u){ const {width,height}=chartSize(); u.setSize({width,height}); } // chart was hidden -> re-fit
}
async function loadOverview(){
  if(!OV){
    try { OV = await (await fetch("/api/overview"+dsq())).json(); }
    catch(e){ $("#ovGrid").innerHTML = `<div class="empty">failed to load overview: ${e.message}</div>`; return; }
  }
  renderOverview();
}
function renderOverview(){
  const m = OV.meta;
  $("#ovSummary").innerHTML =
    `<span class="chip"><b>${m.frozen}</b>/${m.total} frozen (kept as clients)</span>` +
    Object.entries(m.per_cluster).map(([c,o])=>
      `<span class="chip"><span class="cdot" style="background:${CLC[c%6]}"></span>c${c} <b>${o.frozen}</b>/${o.total}</span>`).join("") +
    `<span class="chip">excluded: ${Object.entries(m.reasons).map(([k,v])=>v+" "+k.replace(/_/g,"-")).join(" · ")||"none"}</span>`;
  $("#ovfAll").textContent      = `all ${m.total}`;                 // counts come from the data, never hardcoded
  $("#ovfFrozen").textContent   = `frozen (${m.frozen})`;
  $("#ovfExcluded").textContent = `excluded (${m.total - m.frozen})`;
  buildGrid();
}
function buildGrid(){
  const g = $("#ovGrid"); g.innerHTML="";
  const rows = OV.series.slice().sort((a,b)=> a.cluster-b.cluster || (b.viable-a.viable) || a.id-b.id);
  const pend = [];
  for(const r of rows){
    if(ovFilter==="frozen"   && !r.viable) continue;
    if(ovFilter==="excluded" &&  r.viable) continue;
    if(ovFilter==="nan"      && r.nan_total===0) continue;
    const t = document.createElement("div");
    t.className = "tile" + (r.viable ? " frozen" : " excl");
    const cv = document.createElement("canvas"); t.appendChild(cv);
    const cap = document.createElement("div"); cap.className="cap";
    const rs = r.viable ? `c${r.cluster}` : r.reason.replace(/_/g," ");
    const rsTitle = r.viable ? "" : ` title="${r.reason.replace(/_/g," ")}${r.reason_detail?" "+r.reason_detail:""}"`;
    const nan = r.nan_total>0
      ? `<span class="nantag" title="missing points inside this client's used data">NaN ${r.nan_total}</span>` : "";
    cap.innerHTML = `<span class="cdot" style="background:${r.viable?CLC[r.cluster%6]:'#5b6672'}"></span>`+
                    `<span class="id">KPI ${r.id}</span><span class="rs"${rsTitle}>${rs}</span>${nan}`;
    t.appendChild(cap);
    if(r.viable) t.onclick = ()=>{
      switchTab("detail");
      const row = document.querySelector('.client[data-numid="'+r.id+'"]');
      if(row) row.closest(".folder")?.classList.remove("collapsed");
      loadSeries(row ? row.dataset.id : r.id, row);
    };
    g.appendChild(t); pend.push([cv, r]);
  }
  if(!pend.length) g.innerHTML = `<div class="empty">no series match this filter</div>`;
  requestAnimationFrame(()=> pend.forEach(([cv,r])=> drawSpark(cv, r)));
}
function polyline(ctx, sp, X, Y){
  ctx.beginPath(); let started=false;
  for(let i=0;i<sp.length;i++){ const v=sp[i];
    if(v===null){ started=false; continue; }
    const px=X(i), py=Y(v);
    if(!started){ ctx.moveTo(px,py); started=true; } else ctx.lineTo(px,py);
  }
  ctx.stroke();
}
function drawSpark(cv, r){
  // sparkline spans the FULL original series; the client WINDOW [win_start,win_end] is
  // highlighted in cluster colour, the rest stays grey context (shows WHERE it comes from).
  const dpr = window.devicePixelRatio||1, w = cv.clientWidth||196, h = 54;
  cv.width = Math.round(w*dpr); cv.height = Math.round(h*dpr);
  const ctx = cv.getContext("2d"); ctx.scale(dpr,dpr);
  const sp = r.spark, n = sp.length;
  let lo=Infinity, hi=-Infinity;
  for(const x of sp) if(x!==null){ if(x<lo)lo=x; if(x>hi)hi=x; }
  if(!isFinite(lo)){ lo=0; hi=1; }
  const pad=4, H=h-2*pad, rng=(hi-lo)||1;
  const X = i => (n<2?0:i/(n-1)*w), Y = v => pad + H - (v-lo)/rng*H;
  const px = oi => (r.orig_n ? oi/r.orig_n*w : 0);          // original index -> pixel
  ctx.lineWidth = 1;
  ctx.strokeStyle = r.viable ? "#586475" : "#6b7684";
  polyline(ctx, sp, X, Y);                                   // full-series context line
  if(r.viable){
    const wx0=px(r.win_start), wx1=px(r.win_end);
    ctx.save();
    ctx.beginPath(); ctx.rect(wx0,0,Math.max(1,wx1-wx0),h); ctx.clip();   // limit to the window
    ctx.fillStyle="rgba(70,160,90,0.18)"; ctx.fillRect(wx0,0,px(r.split)-wx0,h);   // train shade
    ctx.strokeStyle=CLC[r.cluster%6]; polyline(ctx, sp, X, Y);            // window line in colour
    ctx.strokeStyle="#e8b64c"; ctx.setLineDash([3,3]); ctx.beginPath();
    ctx.moveTo(px(r.split),0); ctx.lineTo(px(r.split),h); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle="rgba(224,96,91,0.9)";
    for(const [s,e] of (r.spans||[])){ const a=px(s), b=Math.max(px(e), a+1.3); ctx.fillRect(a,0,b-a,h); }
    ctx.restore();
  }
}

// ---------- dataset list + switching ----------
async function loadDatasets(){
  const sel = $("#dsSelect");
  let r;
  try { r = await (await fetch("/api/datasets")).json(); }
  catch(e){ r = {datasets:[], default:null}; }
  sel.innerHTML = "";
  r.datasets.forEach(d=>{
    const o = document.createElement("option");
    o.value = d.name;
    o.textContent = `${d.name}  ·  ${d.n_clients} clients · ${d.n_clusters} clusters`;
    sel.appendChild(o);
  });
  // initial dataset: ?ds= in the URL, else server default, else first
  const urlDs = new URLSearchParams(location.search).get("ds");
  const names = r.datasets.map(d=>d.name);
  DS = (urlDs && names.includes(urlDs)) ? urlDs : (r.default || names[0] || "wsd_fed");
  sel.value = DS;
  sel.addEventListener("change", ()=> switchDataset(sel.value));
}

function switchDataset(name){
  if(name === DS) return;
  DS = name;
  // reset all per-dataset view state
  if(u){ u.destroy(); u=null; } cur=null; OV=null; TREE=null; ovFilter="all";
  // keep the dataset in the URL (survives refresh) but drop the client hash
  history.replaceState(null, "", location.pathname + "?ds=" + encodeURIComponent(DS));
  switchTab("detail");
  $("#toolbar").style.display="none"; $("#chartwrap").style.display="none";
  $("#info").innerHTML="";
  const empty=$("#empty"); empty.style.display=""; empty.textContent="← pick a cluster folder, then a client to view its train / test / anomalies";
  $("#search").value=""; applySearch("");
  loadDataset();
}

// ---------- dataset load ----------
async function loadDataset(){
  const meta = await (await fetch("/api/meta"+dsq())).json();
  NOUN = meta.entity_noun || "series";
  document.title = (meta.dataset || meta.label) + " · dataset viewer";
  const h1 = $("#h1title"); if(h1) h1.textContent = meta.dataset || meta.label;
  const sb = $("#search"); if(sb) sb.placeholder = `filter by ${NOUN} id…`;
  const ob = document.querySelector('#tabs button[data-tab="overview"]');
  if(ob) ob.style.display = meta.has_overview ? "" : "none";     // show/hide both ways on switch
  $("#ver").textContent = "";
  $("#metaChips").innerHTML =
    `<span class="chip"><b>${meta.n_clients}</b> clients</span>`+
    `<span class="chip"><b>${meta.n_clusters}</b> clusters</span>`+
    `<span class="chip">seed <b>${meta.seed}</b></span>`;
  buildTree(await (await fetch("/api/manifest"+dsq())).json());
}

// ---------- boot ----------
async function boot(){
  $("#search").addEventListener("input", e=>applySearch(e.target.value));
  $("#reset").addEventListener("click", ()=>{ viewmode="full"; setSeg("full"); applyView(); });
  $("#viewmode").addEventListener("click", e=>{
    const b=e.target.closest("button"); if(!b) return;
    viewmode=b.dataset.v; setSeg(viewmode); applyView();
  });
  const bind=(id,key)=>$(id).addEventListener("change",e=>{ S[key]=e.target.checked; if(u) u.redraw(); });
  bind("#tgTrain","showTrain"); bind("#tgTest","showTest"); bind("#tgAnom","showAnom");
  window.addEventListener("resize", ()=>{ if(u){ const {width,height}=chartSize(); u.setSize({width,height}); } });
  $("#fulllen").addEventListener("click", ()=>{
    fullLen=!fullLen; $("#fulllen").classList.toggle("on", fullLen);
    if(u){ const {width,height}=chartSize(); u.setSize({width,height}); $("#chartwrap").scrollLeft=0; }
  });
  $("#tabs").addEventListener("click", e=>{ const b=e.target.closest("button"); if(b) switchTab(b.dataset.tab); });
  $("#ovFilter").addEventListener("click", e=>{ const b=e.target.closest("button"); if(!b) return;
    ovFilter=b.dataset.f; document.querySelectorAll("#ovFilter button").forEach(x=>x.classList.toggle("on",x===b)); buildGrid(); });
  await loadDatasets();
  await loadDataset();
  if(location.hash === "#overview") switchTab("overview");   // deep-link: /#overview or /#<kpi id>
}
function setSeg(v){ document.querySelectorAll("#viewmode button").forEach(b=>b.classList.toggle("on", b.dataset.v===v)); }
boot();
