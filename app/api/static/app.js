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

  $("file").addEventListener("change", (e) => {
    selectedFile = e.target.files[0] || null;
    $("file-label").textContent = selectedFile ? selectedFile.name : "Choose a PDF…";
    $("filedrop").classList.toggle("has-file", !!selectedFile);
    $("ingest-btn").disabled = !selectedFile;
  });

  $("ingest-form").addEventListener("submit", onIngest);
  $("refresh-docs").addEventListener("click", () => { loadDocuments(); loadProfile(); });
});

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
