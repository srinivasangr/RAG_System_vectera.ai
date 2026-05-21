// RAG System — ingestion console client

const $ = (id) => document.getElementById(id);

// Stage definitions (order matters for the timeline)
const STAGES = [
  ["parse", "Parse"],
  ["identify", "Identify"],
  ["chunk", "Chunk"],
  ["vision", "Vision"],
  ["propositions", "Propositions"],
  ["embed", "Embed"],
  ["store", "Store"],
];

const ICON = { pending: "○", active: "⚙", done: "✓", error: "✗", skipped: "–" };

let selectedFile = null;

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
window.addEventListener("DOMContentLoaded", () => {
  loadProfile();
  loadDocuments();
  loadDocFilter();

  // tabs
  document.querySelectorAll(".tab").forEach((b) =>
    b.addEventListener("click", () => switchView(b.dataset.view)));

  // ask
  $("ask-form").addEventListener("submit", onAsk);
  $("sel-all").addEventListener("click", () => setAllDocs(true));
  $("sel-none").addEventListener("click", () => setAllDocs(false));

  // ingest
  $("file").addEventListener("change", (e) => {
    selectedFile = e.target.files[0] || null;
    $("file-label").textContent = selectedFile ? selectedFile.name : "Choose a PDF…";
    $("filedrop").classList.toggle("has-file", !!selectedFile);
    $("ingest-btn").disabled = !selectedFile;
  });
  $("ingest-form").addEventListener("submit", onIngest);
  $("refresh-docs").addEventListener("click", () => { loadDocuments(); loadProfile(); });

  // history
  $("refresh-hist").addEventListener("click", loadHistory);

  // modal
  $("modal-close").addEventListener("click", () => $("modal").classList.add("hidden"));
  $("modal").addEventListener("click", (e) => { if (e.target.id === "modal") $("modal").classList.add("hidden"); });
});

function switchView(view) {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  $("view-ask").classList.toggle("hidden", view !== "ask");
  $("view-history").classList.toggle("hidden", view !== "history");
  $("view-ingest").classList.toggle("hidden", view !== "ingest");
  if (view === "ingest") { loadDocuments(); loadProfile(); }
  if (view === "history") loadHistory();
}

async function loadHistory() {
  const box = $("hist-list");
  box.innerHTML = "loading…";
  try {
    const rows = await (await fetch("/api/history?limit=50")).json();
    if (!Array.isArray(rows) || !rows.length) { box.innerHTML = '<div class="dim">No queries yet.</div>'; return; }
    box.innerHTML = rows.map((r) => `
      <details class="hist">
        <summary>
          <span class="hist-q">${esc(r.question || "")}</span>
          <span class="hist-meta">${esc((r.created_at || "").slice(0, 19))} · ${esc(r.intent || "")} · ${esc(r.llm_model || r.llm_provider || "")}${r.total_latency_ms ? " · " + r.total_latency_ms + "ms" : ""}</span>
        </summary>
        <div class="hist-ans">${esc(r.answer || "").replace(/\n/g, "<br>")}</div>
      </details>`).join("");
  } catch { box.innerHTML = '<div class="dim">Failed to load history.</div>'; }
}

// ---------------------------------------------------------------------------
// Document filter sidebar
// ---------------------------------------------------------------------------
async function loadDocFilter() {
  const box = $("doc-filter");
  try {
    const docs = await (await fetch("/api/documents")).json();
    if (!Array.isArray(docs) || !docs.length) { box.innerHTML = '<div class="dim">No documents.</div>'; return; }
    box.innerHTML = docs.map((d) => `
      <label class="docchk">
        <input type="checkbox" class="docbox" value="${esc(d.doc_id)}" checked />
        <span class="docchk-name">${esc(d.company || d.doc_id)}</span>
        <span class="docchk-meta">${esc(d.doc_type || "")} · ${esc(d.as_of_date || "—")}</span>
      </label>`).join("");
  } catch { box.innerHTML = '<div class="dim">Failed to load.</div>'; }
}
function setAllDocs(on) { document.querySelectorAll(".docbox").forEach((c) => (c.checked = on)); }
function selectedDocIds() {
  const all = [...document.querySelectorAll(".docbox")];
  const chk = all.filter((c) => c.checked).map((c) => c.value);
  // empty selection OR all selected => no filter (search everything)
  return (chk.length === 0 || chk.length === all.length) ? [] : chk;
}

const STAGE_LABEL = {
  routing: "Understanding your question…",
  retrieving: "Searching documents (dense + keyword + tables/charts)…",
  reranking: "Ranking the best passages…",
  expanding: "Gathering context + checking versions…",
  generating: "Writing a grounded answer…",
};

// Parse an SSE POST stream frame-by-frame.
async function streamPost(url, payload, onEvent) {
  const resp = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true }).replace(/\r\n/g, "\n"); // normalize CRLF
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
      const line = frame.split("\n").find((l) => l.startsWith("data:"));
      if (line) { try { onEvent(JSON.parse(line.slice(5).trim())); } catch {} }
    }
  }
}

// ---------------------------------------------------------------------------
// Ask
// ---------------------------------------------------------------------------
async function onAsk(e) {
  e.preventDefault();
  const q = $("q").value.trim();
  if (!q) return;
  const [provider, model] = $("ask-engine").value.split("|");

  $("ask-btn").disabled = true;
  $("answer-wrap").classList.add("hidden");
  const st = $("ask-status");
  st.classList.remove("hidden");
  st.innerHTML = '<span class="spin">⟳</span> Starting…';

  let r = null;
  try {
    await streamPost("/api/query/stream",
      { query: q, provider, model, doc_ids: selectedDocIds() },
      (ev) => {
        if (ev.event === "stage") {
          st.innerHTML = `<span class="spin">⟳</span> ${esc(STAGE_LABEL[ev.stage] || ev.stage)}`;
        } else if (ev.event === "done") {
          r = ev.result;
        } else if (ev.event === "error") {
          st.textContent = "Error: " + ev.message;
        }
      });
  } catch (err) {
    st.innerHTML = '<span class="spin">⟳</span> Working…';  // streaming hiccup → fall back below
  }
  // Fallback: if streaming didn't yield a result, use the plain endpoint.
  if (!r || !r.answer) {
    try {
      r = await (await fetch("/api/query", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, provider, model, doc_ids: selectedDocIds() }),
      })).json();
    } catch (err) {
      st.textContent = "Error: " + err; $("ask-btn").disabled = false; return;
    }
  }
  if (!r || !r.answer) { st.textContent = "Error: no answer"; $("ask-btn").disabled = false; return; }

  st.classList.add("hidden");
  $("answer-wrap").classList.remove("hidden");

  const t = r.timings || {};
  $("answer-meta").innerHTML =
    `<span class="pill">${esc(r.intent || "")}</span> ` +
    `<span class="pill">${esc(r.engine || "")}</span> ` +
    `<span class="dim">retrieve ${t.retrieve_ms || "?"}ms · rerank ${t.rerank_ms || "?"}ms · gen ${t.generate_ms || "?"}ms · ${r.sources.length} sources</span>`;

  $("answer").innerHTML = renderAnswer(r.answer, new Set(r.cited_numbers || []));

  $("conflicts").innerHTML = (r.conflicts && r.conflicts.length)
    ? `⚠ Conflicting versions surfaced: ` + r.conflicts.map((c) =>
        `<b>${esc(c.company)}</b> (${c.as_of_dates.join(", ")})`).join(" · ")
    : "";

  $("sources").innerHTML = r.sources.map((s) => `
    <div class="src ${s.cited ? "cited" : ""}">
      <div class="src-h">
        <span class="src-n">[${s.n}]</span>
        <b>${esc(s.company || "?")}</b>
        <span class="pill">${esc(s.doc_type || "")}</span>
        <span class="dim">p.${s.page_number} · as of ${esc(s.as_of_date || "—")}</span>
        ${s.conflict_group ? '<span class="warn">⚠ version</span>' : ""}
      </div>
      <div class="src-title">${esc(s.slide_title || "")}</div>
      <div class="src-file">📄 ${esc(s.filename || s.doc_id || "")}</div>
      <div class="src-snip">${esc(s.snippet || "")}</div>
      <button class="src-img-btn" data-pid="${esc(s.parent_id)}" data-label="${esc((s.filename||s.company||'')+' — p.'+s.page_number)}">🔍 view source page</button>
    </div>`).join("");
  $("sources").querySelectorAll(".src-img-btn").forEach((b) =>
    b.addEventListener("click", () => showPage(b.dataset.pid, b.dataset.label)));

  $("ask-btn").disabled = false;
}

// highlight [N] / [N,M] citation markers
function renderAnswer(text, cited) {
  const escd = esc(text);
  return escd.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (m) =>
    `<span class="cite">${m}</span>`).replace(/\n/g, "<br>");
}

function showPage(parentId, label) {
  $("modal-body").innerHTML =
    `<div class="modal-label">${esc(label)}</div>` +
    `<img class="modal-img" src="/api/page-image/${encodeURIComponent(parentId)}" alt="source page" />`;
  $("modal").classList.remove("hidden");
}

// ---------------------------------------------------------------------------
// Corpus profile + documents
// ---------------------------------------------------------------------------
async function loadProfile() {
  try {
    const p = await (await fetch("/api/corpus-profile")).json();
    if (p.error) { $("corpus-summary").textContent = "corpus: (error)"; return; }
    const dr = p.date_range && p.date_range[0] ? ` · ${p.date_range[0]} → ${p.date_range[1]}` : "";
    $("corpus-summary").textContent =
      `${p.n_documents} docs · ${p.doc_types.length} types · ${p.entities.length} entities${dr}`;
  } catch { $("corpus-summary").textContent = "corpus: unavailable"; }
}

async function loadDocuments() {
  const body = $("docs-body");
  try {
    const docs = await (await fetch("/api/documents")).json();
    if (!Array.isArray(docs) || docs.length === 0) {
      body.innerHTML = `<tr><td colspan="10" class="empty">No documents yet.</td></tr>`;
      return;
    }
    body.innerHTML = docs.map((d) => `
      <tr>
        <td>${esc(d.company || "—")}${d.ticker ? ` <span class="pill">${esc(d.ticker)}</span>` : ""}</td>
        <td><span class="pill">${esc(d.doc_type || "—")}</span></td>
        <td>${esc(d.as_of_date || "—")}</td>
        <td class="num">${d.page_count ?? "—"}</td>
        <td class="num">${d.chunks}</td>
        <td class="num">${d.parents}</td>
        <td class="num">${d.propositions}</td>
        <td class="num">${d.table_rows}</td>
        <td class="num">${d.chart_records}</td>
        <td><button class="del" title="Delete" data-id="${esc(d.doc_id)}">🗑</button></td>
      </tr>`).join("");
    body.querySelectorAll(".del").forEach((b) =>
      b.addEventListener("click", () => deleteDoc(b.dataset.id)));
  } catch {
    body.innerHTML = `<tr><td colspan="10" class="empty">Failed to load documents.</td></tr>`;
  }
}

async function deleteDoc(docId) {
  if (!confirm("Delete this document and all its chunks/propositions/etc.?")) return;
  await fetch(`/api/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });
  loadDocuments(); loadProfile();
}

// ---------------------------------------------------------------------------
// Ingestion + live progress
// ---------------------------------------------------------------------------
async function onIngest(e) {
  e.preventDefault();
  if (!selectedFile) return;

  const [provider, model] = $("engine").value.split("|");
  const fd = new FormData();
  fd.append("file", selectedFile);
  fd.append("with_vision", $("with_vision").checked);
  fd.append("with_propositions", $("with_propositions").checked);
  fd.append("llm_provider", provider);
  fd.append("llm_model", model);

  $("ingest-btn").disabled = true;
  resetStages($("with_vision").checked, $("with_propositions").checked);
  $("progress").classList.remove("hidden");
  $("progress-title").textContent = `Ingesting ${selectedFile.name}…`;
  logLine(`▶ upload ${selectedFile.name}`);

  let job;
  try {
    job = await (await fetch("/api/ingest", { method: "POST", body: fd })).json();
  } catch (err) {
    logLine(`✗ upload failed: ${err}`); $("ingest-btn").disabled = false; return;
  }
  if (job.error) { logLine(`✗ ${job.error}`); $("ingest-btn").disabled = false; return; }

  const es = new EventSource(`/api/ingest/${job.job_id}/stream`);
  es.onmessage = (m) => handleEvent(JSON.parse(m.data), es);
  es.onerror = () => { es.close(); $("ingest-btn").disabled = false; };
}

function handleEvent(ev, es) {
  const name = ev.event;
  // log everything compactly
  const detailKeys = Object.keys(ev).filter((k) => !["event", "ts"].includes(k));
  logLine(`· ${name}${detailKeys.length ? " " + JSON.stringify(pick(ev, detailKeys)) : ""}`);

  switch (name) {
    case "job_started": case "start":
      setStage("parse", "active"); break;
    case "parse_start":
      setStage("parse", "active", `${ev.total || "?"} pages`); break;
    case "parse_batch":
      setStage("parse", "active", `pages ${ev.start}–${ev.end} / ${ev.total}`); break;
    case "parse_done":
      setStage("parse", "done", `${ev.pages} pages`); setStage("identify", "active"); break;
    case "identify_done":
      setStage("identify", "done", `${ev.company || "?"} · ${ev.doc_type} · as_of ${ev.as_of}`);
      setStage("chunk", "active"); break;
    case "chunk_done":
      setStage("chunk", "done", `${ev.parents}p · ${ev.children}c · ${ev.table_rows} rows`);
      setStage("vision", "active"); break;
    case "vision_start":
      setStage("vision", "active", `0/${ev.total}`); break;
    case "vision_progress":
      setStage("vision", "active", `${ev.done}/${ev.total} · ${ev.described} figures`); break;
    case "vision_done":
      setStage("vision", "done", `${ev.records} records`); setStage("propositions", "active"); break;
    case "propositions_progress":
      setStage("propositions", "active", `${ev.done}/${ev.total} chunks`); break;
    case "propositions_done":
      setStage("propositions", "done", `${ev.total} facts`); setStage("embed", "active"); break;
    case "embed_start":
      setStage("embed", "active", `${ev.children}c + ${ev.props}p + ${ev.rows}r`); break;
    case "embed_done":
      setStage("embed", "done"); setStage("store", "active"); break;
    case "done":
      STAGES.forEach(([k]) => { if (stageState[k] !== "done" && stageState[k] !== "skipped") setStage(k, "done"); });
      $("progress-title").textContent = ev.skipped
        ? `Skipped (already ingested): ${stageFile}` : `✓ Done — stored ${ev.doc_status || ""}`;
      logLine(`✓ done ${JSON.stringify(ev.doc_id ? {doc_id: ev.doc_id, children: ev.children, props: ev.propositions, rows: ev.table_rows, charts: ev.chart_records} : ev)}`);
      es.close(); $("ingest-btn").disabled = false; loadDocuments(); loadProfile(); break;
    case "error":
      markActiveError();
      $("progress-title").textContent = `✗ Failed`;
      logLine(`✗ ERROR: ${ev.message}`);
      es.close(); $("ingest-btn").disabled = false; break;
  }
}

// ---------------------------------------------------------------------------
// Stage UI
// ---------------------------------------------------------------------------
let stageState = {};
let stageFile = "";
function resetStages(withVision, withProps) {
  stageState = {};
  const ol = $("stages"); ol.innerHTML = "";
  STAGES.forEach(([key, label]) => {
    const skip = (key === "vision" && !withVision) || (key === "propositions" && !withProps);
    stageState[key] = skip ? "skipped" : "pending";
    const li = document.createElement("li");
    li.className = `stage ${stageState[key]}`;
    li.id = `stage-${key}`;
    li.innerHTML = `<span class="ic">${ICON[stageState[key]]}</span>
                    <span class="name">${label}</span>
                    <span class="detail">${skip ? "skipped" : ""}</span>`;
    ol.appendChild(li);
  });
  $("log").textContent = "";
}

function setStage(key, status, detail) {
  if (stageState[key] === "skipped") return;       // don't reactivate skipped stages
  stageState[key] = status;
  const li = $(`stage-${key}`);
  if (!li) return;
  li.className = `stage ${status}`;
  li.querySelector(".ic").textContent = ICON[status];
  if (detail !== undefined) li.querySelector(".detail").textContent = detail;
}

function markActiveError() {
  for (const [k] of STAGES) {
    if (stageState[k] === "active") { setStage(k, "error"); return; }
  }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
function logLine(s) {
  const el = $("log");
  el.textContent += (el.textContent ? "\n" : "") + s;
  el.scrollTop = el.scrollHeight;
}
function pick(o, keys) { const r = {}; keys.forEach((k) => (r[k] = o[k])); return r; }
function esc(s) { return String(s).replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
