"""Streamlit chat UI for the RAG system — NotebookLM-style, materialistic theme.

  streamlit run rag_system/ui/streamlit_app.py
"""

from __future__ import annotations

import base64 as _b64
import logging
import os
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Make the package importable when launched via `streamlit run path/to/app.py`
_APP_ROOT = Path(__file__).resolve().parents[2]
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

import streamlit as st

# On Streamlit Cloud, credentials live in st.secrets (TOML). Mirror them into
# os.environ BEFORE importing anything that reads settings, so pydantic-settings
# picks them up the same way it picks up .env locally.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float, bool)):
            os.environ.setdefault(_k, str(_v))
except Exception:
    pass  # local dev: no secrets.toml; fall back to .env

from rag_system.config import settings
from rag_system.generation import query
from rag_system.ingest.pipeline import ingest_one
from rag_system.llm_providers import available_llm_providers
from rag_system.llm_providers.anthropic_provider import AnthropicProvider
from rag_system.llm_providers.gemini_provider import GeminiProvider
from rag_system.llm_providers.openai_compat import (
    CerebrasProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from rag_system.retrieval.filters import RetrievalFilters
from rag_system.storage.db import get_connection
from rag_system.storage.repository import (
    corpus_stats,
    delete_document,
    get_chunk_image_b64,
)
# NOTE: ingest_one is imported lazily (inside the upload handler) so the
# heavy Docling + PyTorch import chain doesn't fire on hosted/read-only
# deployments where uploads are disabled.

logging.basicConfig(level=logging.INFO)


# Curated, deduplicated model lists per provider.
# Order matters — the first item is the default the dropdown lands on.
PROVIDER_DEFAULT_MODELS: dict[str, list[str]] = {
    "cerebras":   ["gpt-oss-120b"],
    "gemini":     ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
    "openai":     [OpenAIProvider.DEFAULT_MODEL],
    "anthropic":  [AnthropicProvider.DEFAULT_MODEL],
    "openrouter": [OpenRouterProvider.DEFAULT_MODEL],
}


# ---------------------------------------------------------------------------
# Chat data model
# ---------------------------------------------------------------------------
@dataclass
class ChatMessage:
    role: str
    content: str
    citations: list = field(default_factory=list)
    retrieved: list = field(default_factory=list)
    latency_ms: int = 0
    retrieval_ms: int = 0
    generation_ms: int = 0
    llm_provider: str = ""
    llm_model: str = ""


@dataclass
class Chat:
    id: str
    title: str
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------
def _init_state() -> None:
    if "chats" not in st.session_state:
        st.session_state.chats = {}
        st.session_state.chat_order = []
        st.session_state.active_chat_id = None
        _new_chat()
    if "selected_doc_ids" not in st.session_state:
        st.session_state.selected_doc_ids = None
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False
    if "provider" not in st.session_state:
        avail = available_llm_providers() or ["cerebras"]
        st.session_state.provider = (
            settings.llm_provider if settings.llm_provider in avail else avail[0]
        )
    if "model" not in st.session_state:
        opts = PROVIDER_DEFAULT_MODELS.get(st.session_state.provider, [""])
        st.session_state.model = opts[0]
    if "top_k" not in st.session_state:
        st.session_state.top_k = settings.retrieval_top_k
    if "prefer_recent" not in st.session_state:
        st.session_state.prefer_recent = False
    if "_processed_uploads" not in st.session_state:
        st.session_state._processed_uploads = set()


def _new_chat() -> str:
    cid = str(uuid.uuid4())[:8]
    st.session_state.chats[cid] = Chat(id=cid, title="New chat")
    st.session_state.chat_order.insert(0, cid)
    st.session_state.active_chat_id = cid
    return cid


def _delete_chat(cid: str) -> None:
    st.session_state.chats.pop(cid, None)
    if cid in st.session_state.chat_order:
        st.session_state.chat_order.remove(cid)
    if st.session_state.active_chat_id == cid:
        st.session_state.active_chat_id = (
            st.session_state.chat_order[0] if st.session_state.chat_order else None
        )
    if not st.session_state.chat_order:
        _new_chat()


def _active_chat() -> Chat:
    return st.session_state.chats[st.session_state.active_chat_id]


# ---------------------------------------------------------------------------
# Citation rendering helpers (Perplexity / NotebookLM style)
# ---------------------------------------------------------------------------
_CHUNK_TYPE_ICON = {
    "prose": "📝",
    "table": "📊",
    "chart_description": "📈",
}


def _basename_any(p: str) -> str:
    """Return just the filename, handling Windows or POSIX paths regardless of host OS.

    Why: source_path may have been written on Windows (backslashes) and read on
    Linux (where pathlib.Path treats `\\` as part of the name).
    """
    if not p:
        return ""
    return p.replace("\\", "/").rsplit("/", 1)[-1]


def _source_card_label(c) -> str:
    """Short, two-line label rendered inside the source-card button."""
    icon = _CHUNK_TYPE_ICON.get(c.chunk_type, "📄")
    company_short = (c.company or "Doc")[:18]
    version = c.version_label or "undated"
    return f"**[{c.n}]** {icon} {company_short}\n\n{version} · p.{c.page_number}"


@st.dialog("Delete document?")
def _confirm_delete_document(doc: dict) -> None:
    """Two-click confirmation before wiping a doc from Snowflake."""
    st.warning(
        f"This will permanently delete **{doc['name']}** and all its "
        f"chunks from Snowflake. The chat history that references it "
        f"will still show, but new queries won't retrieve from it."
    )
    st.caption(
        f"📊 {doc.get('n_chunks', 0)} chunks · "
        f"📄 {doc.get('page_count', 0)} pages · "
        f"`{doc['doc_id']}`"
    )
    also_disk = st.checkbox(
        "Also delete the PDF file from disk",
        value=False,
        help=f"`{doc.get('source_path', '')}`",
    )

    btn_cancel, btn_delete = st.columns(2)
    with btn_cancel:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with btn_delete:
        if st.button("🗑 Delete", type="primary", use_container_width=True):
            try:
                counts = delete_document(doc["doc_id"])
                if also_disk and doc.get("source_path"):
                    try:
                        Path(doc["source_path"]).unlink(missing_ok=True)
                    except Exception as e:
                        st.warning(f"DB rows removed, but disk delete failed: {e}")
                # Clear checkbox state so the UI doesn't try to re-render the
                # row with a stale `True` value
                st.session_state.pop(f"doc_{doc['doc_id']}", None)
                if isinstance(st.session_state.selected_doc_ids, list):
                    st.session_state.selected_doc_ids = [
                        x for x in st.session_state.selected_doc_ids if x != doc["doc_id"]
                    ]
                st.cache_data.clear()
                st.toast(
                    f"Deleted: {counts['chunks']} chunks, "
                    f"{counts['images']} images, {counts['documents']} doc",
                    icon="🗑",
                )
                st.rerun()
            except Exception as e:
                st.error(f"Delete failed: {type(e).__name__}: {e}")


@st.dialog("Source", width="large")
def _show_citation_dialog(c) -> None:
    """Modal showing the full content of one citation."""
    icon = _CHUNK_TYPE_ICON.get(c.chunk_type, "📄")
    st.markdown(
        f"### {icon} `[{c.n}]` {c.company or 'Document'}"
        f"{' · ' + c.version_label if c.version_label else ''}"
        f" · p.{c.page_number}"
    )
    st.caption(f"chunk type: `{c.chunk_type}` · chunk_id: `{c.chunk_id}`")

    if c.chunk_type == "chart_description":
        img_b64 = get_chunk_image_b64(c.chunk_id)
        if img_b64:
            st.image(
                _b64.b64decode(img_b64),
                caption=f"Source chart from p.{c.page_number}",
                use_container_width=True,
            )

    st.markdown("##### Excerpt")
    st.markdown(c.text)

    if c.source_path:
        st.caption(f"📄 source PDF: `{_basename_any(c.source_path)}`")


def _autorename_if_needed(chat: Chat, first_msg: str) -> None:
    if chat.title == "New chat":
        t = first_msg.strip().replace("\n", " ")
        chat.title = (t[:40] + "…") if len(t) > 40 else t


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------
def _list_documents() -> list[dict]:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT d.doc_id, d.source_path, d.company, d.version_label,
                   d.page_count, d.ingested_at,
                   COUNT(c.chunk_id) AS n_chunks
            FROM documents d
            LEFT JOIN chunks c ON c.doc_id = d.doc_id
            GROUP BY d.doc_id, d.source_path, d.company,
                     d.version_label, d.page_count, d.ingested_at
            ORDER BY d.ingested_at DESC
        """)
        rows = cur.fetchall()
        cur.close()
    return [{
        "doc_id": doc_id,
        "source_path": src,
        "name": _basename_any(src) if src else doc_id,
        "company": company,
        "version_label": version,
        "page_count": pages,
        "ingested_at": ingested,
        "n_chunks": n_chunks,
    } for doc_id, src, company, version, pages, ingested, n_chunks in rows]


def _save_uploaded_pdf(uploaded_file) -> Path:
    docs_dir = settings.documents_path
    docs_dir.mkdir(parents=True, exist_ok=True)
    dest = docs_dir / uploaded_file.name
    if dest.exists():
        suffix = uuid.uuid4().hex[:6]
        dest = docs_dir / f"{dest.stem}__{suffix}{dest.suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(uploaded_file, f)
    return dest


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Materialistic CSS overhaul
# ============================================================================
st.markdown(
    """
    <style>
    /* === Global tokens === */
    :root {
        --md-bg:           #0F1115;
        --md-surface:      #1A1D24;
        --md-surface-2:    #22262F;
        --md-surface-3:    #2A2F39;
        --md-border:       #2E333E;
        --md-border-hover: #4B5161;
        --md-primary:      #7C4DFF;
        --md-primary-soft: rgba(124, 77, 255, 0.18);
        --md-primary-glow: rgba(124, 77, 255, 0.35);
        --md-text:         #E8EAED;
        --md-text-muted:   #9AA0A6;
        --md-text-soft:    #C7CAD1;
        --md-success:      #00C896;
        --md-radius:       12px;
        --md-radius-lg:    16px;
        --md-shadow-sm:    0 1px 2px rgba(0,0,0,0.4);
        --md-shadow:       0 4px 16px rgba(0,0,0,0.35);
        --md-shadow-lg:    0 8px 28px rgba(0,0,0,0.45);
    }

    /* === Page === */
    .stApp { background: var(--md-bg); }
    .block-container { padding-top: 1.5rem; padding-bottom: 8rem; max-width: 1100px; }

    /* === Sidebar === */
    section[data-testid="stSidebar"] {
        background: var(--md-surface) !important;
        border-right: 1px solid var(--md-border);
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.25rem; padding-bottom: 1rem; }

    /* === Headings === */
    h1, h2, h3, h4, h5 { color: var(--md-text); letter-spacing: -0.01em; }

    /* === Buttons === */
    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid var(--md-border) !important;
        background: var(--md-surface-2) !important;
        color: var(--md-text) !important;
        transition: all 150ms ease;
        font-weight: 500;
    }
    .stButton > button:hover {
        border-color: var(--md-border-hover) !important;
        background: var(--md-surface-3) !important;
        transform: translateY(-1px);
        box-shadow: var(--md-shadow-sm);
    }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, var(--md-primary), #5B36D4) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 2px 12px var(--md-primary-glow);
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 18px var(--md-primary-glow);
        filter: brightness(1.08);
    }

    /* === Sidebar: chat & source list buttons (left-aligned, transparent) === */
    section[data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        border: 1px solid transparent !important;
        text-align: left !important;
        justify-content: flex-start !important;
        font-weight: 400 !important;
        padding: 0.45rem 0.7rem !important;
        color: var(--md-text-soft) !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: var(--md-surface-2) !important;
        border-color: var(--md-border) !important;
        color: var(--md-text) !important;
        transform: none;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        text-align: center !important;
        justify-content: center !important;
        color: white !important;
    }

    /* === Inputs / selects === */
    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div,
    .stDateInput input {
        background: var(--md-surface-2) !important;
        border: 1px solid var(--md-border) !important;
        border-radius: 10px !important;
        color: var(--md-text) !important;
    }
    .stTextInput input:focus,
    .stTextArea textarea:focus,
    .stDateInput input:focus {
        border-color: var(--md-primary) !important;
        box-shadow: 0 0 0 3px var(--md-primary-soft) !important;
    }

    /* === File uploader (drag-drop card) === */
    [data-testid="stFileUploader"] section {
        background: var(--md-surface-2) !important;
        border: 1.5px dashed var(--md-border-hover) !important;
        border-radius: var(--md-radius) !important;
        padding: 1rem !important;
        transition: all 150ms ease;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: var(--md-primary) !important;
        background: var(--md-surface-3) !important;
    }
    [data-testid="stFileUploader"] section button {
        background: var(--md-primary) !important;
        color: white !important;
        border: none !important;
    }

    /* === Checkboxes (sources list) === */
    div[data-testid="stCheckbox"] {
        padding: 0.4rem 0.55rem;
        border-radius: 10px;
        transition: background 120ms ease;
    }
    div[data-testid="stCheckbox"]:hover { background: var(--md-surface-2); }
    div[data-testid="stCheckbox"] label { color: var(--md-text-soft) !important; font-size: 0.88rem; }

    /* === Chat bubbles === */
    [data-testid="stChatMessage"] {
        background: var(--md-surface) !important;
        border: 1px solid var(--md-border);
        border-radius: var(--md-radius-lg);
        padding: 1rem 1.1rem !important;
        margin-bottom: 0.75rem !important;
        box-shadow: var(--md-shadow-sm);
    }
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: linear-gradient(135deg, rgba(124, 77, 255, 0.10), rgba(124, 77, 255, 0.04)) !important;
        border-color: rgba(124, 77, 255, 0.25);
    }

    /* === Expanders (citations, debug) === */
    .stExpander {
        background: var(--md-surface) !important;
        border: 1px solid var(--md-border) !important;
        border-radius: 10px !important;
        margin-top: 0.5rem;
    }
    .stExpander summary {
        font-size: 0.85rem !important;
        color: var(--md-text-soft) !important;
        padding: 0.5rem 0.8rem !important;
    }
    .stExpander summary:hover { color: var(--md-text) !important; }

    /* === Info / status banners === */
    div[data-testid="stAlert"] {
        background: var(--md-surface) !important;
        border: 1px solid var(--md-border) !important;
        border-radius: var(--md-radius) !important;
        color: var(--md-text) !important;
    }

    /* === Metrics === */
    [data-testid="stMetric"] {
        background: var(--md-surface-2);
        border: 1px solid var(--md-border);
        border-radius: 10px;
        padding: 0.5rem 0.7rem;
    }
    [data-testid="stMetricValue"] { color: var(--md-text) !important; font-size: 1rem !important; }
    [data-testid="stMetricLabel"] { color: var(--md-text-muted) !important; font-size: 0.72rem !important; }

    /* === Code blocks (citation contents) === */
    .stCode, code, pre { font-size: 0.82rem !important; }

    /* === Captions === */
    [data-testid="stCaptionContainer"] {
        color: var(--md-text-muted) !important;
        font-size: 0.78rem !important;
    }

    /* === Popover (model picker) === */
    div[data-testid="stPopover"] button {
        background: var(--md-surface-2) !important;
        border: 1px solid var(--md-border) !important;
        border-radius: 999px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: var(--md-text-soft) !important;
        padding: 0.35rem 0.85rem !important;
        height: auto !important;
        white-space: nowrap !important;
    }
    div[data-testid="stPopover"] button:hover {
        border-color: var(--md-primary) !important;
        color: var(--md-text) !important;
        background: var(--md-surface-3) !important;
    }

    /* === The composer (custom input bar) === */
    .composer-card {
        background: var(--md-surface) !important;
        border: 1px solid var(--md-border) !important;
        border-radius: var(--md-radius-lg) !important;
        padding: 0.6rem 0.6rem 0.4rem 0.6rem !important;
        box-shadow: var(--md-shadow);
        transition: border-color 150ms ease;
    }
    .composer-card:focus-within {
        border-color: var(--md-primary) !important;
        box-shadow: 0 0 0 3px var(--md-primary-soft), var(--md-shadow);
    }
    .composer-card .stTextArea textarea {
        background: transparent !important;
        border: none !important;
        font-size: 0.95rem !important;
        min-height: 60px !important;
        resize: none !important;
        padding: 0.35rem 0.5rem !important;
    }
    .composer-card .stTextArea textarea:focus { box-shadow: none !important; }
    .composer-card label { display: none !important; }

    /* Hide the surrounding white border of the form */
    .stForm { border: none !important; padding: 0 !important; background: transparent !important; }

    /* === Send button (gradient circle) === */
    button.composer-send,
    [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, var(--md-primary), #5B36D4) !important;
        color: white !important;
        border: none !important;
        border-radius: 999px !important;
        height: 38px !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 12px var(--md-primary-glow);
    }
    [data-testid="stFormSubmitButton"] button:hover {
        filter: brightness(1.1);
        box-shadow: 0 4px 18px var(--md-primary-glow);
    }

    /* === Divider === */
    hr { border-color: var(--md-border) !important; opacity: 0.5; }

    /* === Make the active chat row "stand out" === */
    .active-chat-marker { color: var(--md-primary) !important; }

    /* === Source cards (under each assistant answer) =================== */
    .sources-row { margin-top: 0.6rem; margin-bottom: 0.4rem; }
    .sources-row .sources-label {
        font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;
        color: var(--md-text-muted); margin-bottom: 0.4rem; font-weight: 600;
    }
    /* Style every BUTTON inside an assistant chat message as a "source card" */
    [data-testid="stChatMessage"] .stButton > button {
        background: var(--md-surface-2) !important;
        border: 1px solid var(--md-border) !important;
        border-radius: 10px !important;
        padding: 0.55rem 0.7rem !important;
        text-align: left !important;
        justify-content: flex-start !important;
        font-weight: 500 !important;
        font-size: 0.78rem !important;
        line-height: 1.25 !important;
        color: var(--md-text-soft) !important;
        height: auto !important;
        min-height: 56px !important;
        white-space: normal !important;
        word-break: break-word !important;
        transition: all 150ms ease;
    }
    [data-testid="stChatMessage"] .stButton > button:hover {
        border-color: var(--md-primary) !important;
        background: var(--md-surface-3) !important;
        color: var(--md-text) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(124, 77, 255, 0.15);
    }

    /* === Dialog (citation modal) tweaks === */
    div[data-testid="stDialog"] [data-testid="stMarkdown"] code,
    div[data-testid="stDialog"] pre {
        font-size: 0.82rem !important;
    }
    div[data-testid="stDialog"] h1,
    div[data-testid="stDialog"] h2,
    div[data-testid="stDialog"] h3 {
        margin-top: 0.4rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_init_state()


# ===========================================================================
# SIDEBAR — Sources + Chats
# ===========================================================================
with st.sidebar:
    # === Sources ===
    st.markdown("### 📚 Sources")
    if settings.is_hosted:
        # Read-only hosted deployment: disable uploads to keep the instance
        # within memory + storage limits.
        st.caption(
            "🔒 Uploads are disabled on the hosted version. "
            "To add documents, run the app locally — see the README."
        )
        uploaded_files = None
    else:
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="pdf_uploader",
        )

    # Synchronous ingest — blocks the UI for the duration of the upload but is
    # rock-solid on Windows (the background-thread version had Docling init
    # races). Live progress shown via st.status(...).
    enable_vision_upload = st.checkbox(
        "🎨 Describe chart images (uses Gemini vision)",
        value=False,
        help="OFF (default) = text + tables only, faster + no daily quota risk. "
             "Turn ON when Gemini vision daily quota has reset (midnight Pacific).",
        key="enable_vision_upload",
    )
    if uploaded_files:
        # Lazy import — only pulled in when an actual upload happens.
        from rag_system.ingest.pipeline import ingest_one

        new_files = [
            f for f in uploaded_files
            if f.file_id not in st.session_state._processed_uploads
        ]
        for uf in new_files:
            saved_path = _save_uploaded_pdf(uf)
            with st.status(
                f"📥 Ingesting **{uf.name}** …", expanded=True
            ) as status:
                stage_box = st.empty()
                progress = st.progress(0.0, text="starting")
                log_box = st.empty()
                log_lines: list[str] = []

                # Map pipeline events to live UI updates
                def _cb(event: str, payload: dict) -> None:
                    import datetime as _dt
                    ts = _dt.datetime.now().strftime("%H:%M:%S")
                    if event == "start":
                        log_lines.append(f"[{ts}] start: {payload.get('file', uf.name)}")
                        stage_box.caption("📄 parsing…")
                        progress.progress(0.05, text="parsing")
                    elif event == "parse_start":
                        total = payload.get("total", 0)
                        log_lines.append(f"[{ts}] parsing {total} pages (batch={payload.get('batch_size')})")
                    elif event == "parse_batch":
                        end = payload.get("end", 0)
                        total = payload.get("total", 1) or 1
                        pct = 0.05 + 0.55 * (end / total)
                        progress.progress(min(0.60, pct),
                                          text=f"parsing {end}/{total} pages")
                        log_lines.append(
                            f"[{ts}] parsed pages {payload.get('start')}-{end}/{total} "
                            f"({payload.get('elapsed_s', 0):.1f}s)"
                        )
                    elif event == "parse_done":
                        stage_box.caption(
                            f"📄 parse done — {payload.get('pages')} pages, "
                            f"{payload.get('images')} images"
                        )
                        progress.progress(0.60, text="parse done")
                    elif event == "vision_start":
                        log_lines.append(f"[{ts}] vision: describing {payload.get('total')} images")
                        stage_box.caption("📈 vision (rate-limited)…")
                    elif event == "vision_progress":
                        d, t = payload.get("done", 0), payload.get("total", 1)
                        progress.progress(0.60 + 0.15 * (d / max(t, 1)),
                                          text=f"vision {d}/{t}")
                    elif event == "vision_done":
                        log_lines.append(
                            f"[{ts}] vision done: kept {payload.get('described')}/{payload.get('total')}"
                        )
                    elif event == "chunk_done":
                        log_lines.append(
                            f"[{ts}] chunked: {payload.get('total')} chunks "
                            f"{dict(payload.get('by_type', {}))}"
                        )
                        progress.progress(0.75, text="chunked")
                    elif event == "embed_start":
                        log_lines.append(
                            f"[{ts}] embedding {payload.get('total')} chunks (local BGE)"
                        )
                        stage_box.caption("🧬 embedding…")
                    elif event == "embed_progress":
                        d, t = payload.get("done", 0), payload.get("total", 1)
                        progress.progress(0.75 + 0.20 * (d / max(t, 1)),
                                          text=f"embed {d}/{t}")
                    elif event == "embed_done":
                        log_lines.append(f"[{ts}] embedded {payload.get('total')} chunks")
                        progress.progress(0.95, text="upserting")
                        stage_box.caption("☁ upserting to Snowflake…")
                    elif event == "upsert_done":
                        log_lines.append(
                            f"[{ts}] snowflake: {payload.get('doc_status')} · "
                            f"{payload.get('chunks')} chunks stored"
                        )
                        progress.progress(1.0, text="done")
                    elif event == "done":
                        log_lines.append(f"[{ts}] DONE in {payload.get('elapsed_s')}s")
                    elif event == "error":
                        log_lines.append(
                            f"[{ts}] ERROR ({payload.get('stage')}): {payload.get('message')}"
                        )
                    log_box.text("\n".join(log_lines[-8:]))

                try:
                    result = ingest_one(
                        saved_path,
                        with_vision=enable_vision_upload,
                        vision_call_budget=20,
                        progress_cb=_cb,
                    )
                    status.update(
                        label=f"✅ Ingested **{uf.name}** — "
                              f"{result.get('chunks', 0)} chunks in "
                              f"{result.get('elapsed_s', 0)}s",
                        state="complete", expanded=False,
                    )
                except Exception as e:
                    status.update(
                        label=f"❌ Failed: **{uf.name}** — {type(e).__name__}: {e}",
                        state="error",
                    )
            st.session_state._processed_uploads.add(uf.file_id)
        if new_files:
            st.cache_data.clear()
            st.rerun()

    # ============ Active jobs panel (no-op now that uploads are synchronous) ===
    # Kept as a stub in case we re-enable the background JobManager later.
    def _render_jobs_panel():
        return
        # Original fragment-based implementation lives in git history; restore
        # by reverting this stub + re-importing get_job_manager / JOB_STATUS_*.
        jobs = []  # mgr.list()
        if not jobs:
            return

        # Summary header
        n_running = sum(1 for j in jobs if j.status == JOB_STATUS_RUNNING)
        n_queued = sum(1 for j in jobs if j.status == JOB_STATUS_QUEUED)
        n_done = sum(1 for j in jobs if j.status == JOB_STATUS_DONE)
        n_err = sum(1 for j in jobs if j.status == JOB_STATUS_ERROR)
        summary_bits = []
        if n_running: summary_bits.append(f"⏳ {n_running} running")
        if n_queued:  summary_bits.append(f"⏸ {n_queued} queued")
        if n_done:    summary_bits.append(f"✅ {n_done} done")
        if n_err:     summary_bits.append(f"❌ {n_err} error")
        st.markdown(f"##### Ingestion jobs · {' · '.join(summary_bits)}")
        st.caption("Jobs run **one at a time** to avoid memory + rate-limit issues.")

        any_changes = False
        for job in jobs:
            snap = job.snapshot()
            status = snap["status"]

            if status == JOB_STATUS_RUNNING:
                icon, color_state = "⏳", "running"
                header_extra = f"{snap['current_stage']} · {snap['elapsed_s']:.0f}s"
            elif status == JOB_STATUS_QUEUED:
                pos = mgr.queue_position(snap["id"])
                icon, color_state = "⏸", "running"
                header_extra = f"queued #{pos} · waiting {snap['wait_s']:.0f}s"
            elif status == JOB_STATUS_DONE:
                icon, color_state = "✅", "complete"
                header_extra = f"done in {snap['elapsed_s']:.0f}s"
            else:
                icon, color_state = "❌", "error"
                header_extra = f"failed after {snap['elapsed_s']:.0f}s"

            with st.status(
                f"{icon}  **{snap['file_name']}**  ·  {header_extra}",
                state=color_state,
                expanded=(status == JOB_STATUS_RUNNING),
            ):
                if status == JOB_STATUS_QUEUED:
                    st.caption(f"Waiting for the current job to finish. "
                               f"Position in queue: {mgr.queue_position(snap['id'])}")
                else:
                    pct = max(0.0, min(1.0, snap["progress"]))
                    st.progress(pct, text=f"{int(pct*100)}%")

                    rows = []
                    if snap["parse_total_pages"]:
                        rows.append(
                            f"📄 parse: {snap['parse_done_pages']}/{snap['parse_total_pages']} pages"
                        )
                    if snap["vision_total"]:
                        rows.append(
                            f"📈 vision: {snap['vision_done']}/{snap['vision_total']} images"
                        )
                    if snap["embed_total"]:
                        rows.append(
                            f"🧬 embed: {snap['embed_done']}/{snap['embed_total']} chunks"
                        )
                    if rows:
                        st.caption(" · ".join(rows))

                    for line in snap["log_lines"][-6:]:
                        st.caption(line)

                if status == JOB_STATUS_DONE and snap["result"]:
                    r = snap["result"]
                    st.success(
                        f"Stored {r.get('chunks', 0)} chunks "
                        f"({r.get('doc_status', '?')}) in "
                        f"{r.get('elapsed_s', 0)}s"
                    )
                    if st.button("Dismiss", key=f"dismiss_{snap['id']}"):
                        mgr.remove(snap["id"])
                        any_changes = True
                elif status == JOB_STATUS_ERROR:
                    st.error(snap["error"] or "Unknown error")
                    if st.button("Dismiss", key=f"dismiss_{snap['id']}"):
                        mgr.remove(snap["id"])
                        any_changes = True
        if any_changes:
            st.cache_data.clear()
            st.rerun()

    _render_jobs_panel()

    docs = _list_documents()

    if not docs:
        st.caption("No documents yet. Upload a PDF to get started.")
    else:
        sel_cols = st.columns(2)
        with sel_cols[0]:
            if st.button("Select all", use_container_width=True, key="sel_all"):
                st.session_state.selected_doc_ids = None
                # Also flip every checkbox's stored state, otherwise Streamlit
                # ignores our value= and keeps the previous check state.
                for d in docs:
                    st.session_state[f"doc_{d['doc_id']}"] = True
                st.rerun()
        with sel_cols[1]:
            if st.button("Clear", use_container_width=True, key="sel_none"):
                st.session_state.selected_doc_ids = []
                for d in docs:
                    st.session_state[f"doc_{d['doc_id']}"] = False
                st.rerun()

        all_ids = [d["doc_id"] for d in docs]
        if st.session_state.selected_doc_ids is None:
            current_selected = set(all_ids)
        else:
            current_selected = set(st.session_state.selected_doc_ids)

        new_selected: set[str] = set()
        for d in docs:
            meta = []
            if d["company"]:        meta.append(d["company"])
            if d["version_label"]:  meta.append(d["version_label"])
            if d["page_count"]:     meta.append(f"{d['page_count']}p")
            help_str = " · ".join(meta + [f"{d['n_chunks']} chunks"])

            # Checkbox + delete icon on the same row
            cb_col, del_col = st.columns([9, 1], gap="small")
            with cb_col:
                checked = st.checkbox(
                    d["name"],
                    value=d["doc_id"] in current_selected,
                    key=f"doc_{d['doc_id']}",
                    help=help_str,
                )
            with del_col:
                if st.button("🗑", key=f"deldoc_{d['doc_id']}",
                             help="Delete from Snowflake"):
                    _confirm_delete_document(d)
            if checked:
                new_selected.add(d["doc_id"])

        if new_selected == set(all_ids):
            st.session_state.selected_doc_ids = None
        else:
            st.session_state.selected_doc_ids = list(new_selected)

        n_active = (
            len(all_ids) if st.session_state.selected_doc_ids is None
            else len(st.session_state.selected_doc_ids)
        )
        st.caption(f"**{n_active}** of {len(all_ids)} selected")

    st.divider()

    # === Chats ===
    chat_header = st.columns([5, 3])
    with chat_header[0]:
        st.markdown("### 💬 Chats")
    with chat_header[1]:
        if st.button("➕ New", use_container_width=True, type="primary", key="new_chat_btn"):
            _new_chat()
            st.rerun()

    for cid in st.session_state.chat_order:
        chat = st.session_state.chats[cid]
        is_active = cid == st.session_state.active_chat_id
        prefix = "▸ " if is_active else "  "
        cols = st.columns([8, 1], gap="small")
        with cols[0]:
            if st.button(
                f"{prefix}{chat.title}",
                key=f"open_{cid}",
                use_container_width=True,
            ):
                st.session_state.active_chat_id = cid
                st.rerun()
        with cols[1]:
            if st.button("🗑", key=f"del_{cid}", help="Delete chat"):
                _delete_chat(cid)
                st.rerun()


# ===========================================================================
# MAIN — chat stream + composer
# ===========================================================================
chat = _active_chat()

# Header
st.markdown(f"### {chat.title}")
docs_selected = (
    len(docs) if (not docs or st.session_state.selected_doc_ids is None)
    else len(st.session_state.selected_doc_ids)
)
total_docs = len(docs) if docs else 0
st.caption(
    f"📎 scope: {docs_selected}/{total_docs} source(s) · "
    f"top-k {st.session_state.top_k}"
    + (" · 🆕 recency boost" if st.session_state.prefer_recent else "")
)

# Render existing messages
for msg_idx, msg in enumerate(chat.messages):
    with st.chat_message(msg.role):
        st.markdown(msg.content)
        if msg.role == "assistant":
            # --- Source cards (Perplexity / NotebookLM-style) ---
            if msg.citations:
                st.markdown(
                    f"<div class='sources-row'>"
                    f"<div class='sources-label'>Sources ({len(msg.citations)})</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                # Lay out 4 cards per row
                CARDS_PER_ROW = 4
                for row_start in range(0, len(msg.citations), CARDS_PER_ROW):
                    row_cites = msg.citations[row_start:row_start + CARDS_PER_ROW]
                    cols = st.columns(CARDS_PER_ROW, gap="small")
                    for i, c in enumerate(row_cites):
                        with cols[i]:
                            if st.button(
                                _source_card_label(c),
                                key=f"src_{chat.id}_{msg_idx}_{c.n}",
                                use_container_width=True,
                                help=c.text[:240] + ("…" if len(c.text) > 240 else ""),
                            ):
                                _show_citation_dialog(c)

            # --- Metrics row (compact, below sources) ---
            mcols = st.columns(4)
            mcols[0].caption(f"⏱ {msg.latency_ms} ms total")
            mcols[1].caption(f"🔍 {msg.retrieval_ms} ms retrieval")
            mcols[2].caption(f"🧠 {msg.generation_ms} ms gen")
            mcols[3].caption(f"📎 {len(msg.citations)} sources")

            if st.session_state.show_debug and msg.retrieved:
                with st.expander(
                    f"🔬 Retrieval debug ({len(msg.retrieved)} chunks)",
                    expanded=False,
                ):
                    for i, r in enumerate(msg.retrieved, 1):
                        tags = []
                        if r.dense_rank:   tags.append(f"dense#{r.dense_rank}")
                        if r.lexical_rank: tags.append(f"keyword#{r.lexical_rank}")
                        header = (
                            f"#{i} · score={r.score:.4f} · {' '.join(tags) or '—'} · "
                            f"{r.company or '?'} {r.version_label or ''} · "
                            f"p.{r.page_number} ({r.chunk_type})"
                        )
                        with st.expander(header, expanded=False):
                            st.code(r.text, language="markdown")

# Empty-state hint
if not chat.messages:
    if docs:
        st.info(
            "💡 Ask any question about your selected sources. Examples:\n\n"
            "- *Summarize the key points of this document.*\n"
            "- *What are the most important numbers mentioned?*\n"
            "- *Compare claims across the selected sources.*\n"
            "- *List any figures or charts and what they show.*"
        )
    else:
        st.info("💡 Upload a PDF in the **Sources** panel on the left to get started.")


# ===========================================================================
# COMPOSER — text area + bottom-right model picker + send (ChatGPT/Claude style)
# ===========================================================================
st.markdown("<div class='composer-card'>", unsafe_allow_html=True)

with st.form(key="composer_form", clear_on_submit=True, border=False):
    user_input = st.text_area(
        "msg",
        key="composer_input",
        placeholder="Ask anything about your documents…  (Ctrl+Enter to send)",
        label_visibility="collapsed",
        height=68,
    )
    # Bottom row: model picker on the LEFT, send button on the RIGHT
    bottom_left, _spacer, bottom_right = st.columns([6, 4, 2], gap="small")
    with bottom_left:
        st.caption(f"🤖 **{st.session_state.provider}** · `{st.session_state.model}`")
    with bottom_right:
        send_clicked = st.form_submit_button("Send ➤", use_container_width=True, type="primary")

st.markdown("</div>", unsafe_allow_html=True)

# Model picker row — appears BELOW the composer card, right-aligned, pill-shaped
picker_l, picker_r = st.columns([8, 4])
with picker_r:
    inner_l, inner_r = st.columns(2)
    with inner_l:
        with st.popover(f"⚙ Model", use_container_width=True):
            st.markdown("##### Provider & model")
            avail = available_llm_providers() or ["cerebras"]
            new_provider = st.selectbox(
                "Provider", avail,
                index=avail.index(st.session_state.provider) if st.session_state.provider in avail else 0,
                key="popover_provider",
            )
            models = PROVIDER_DEFAULT_MODELS.get(new_provider, [""])
            cur_model = (
                st.session_state.model if st.session_state.model in models else models[0]
            )
            new_model = st.selectbox(
                "Model", models, index=models.index(cur_model), key="popover_model",
            )
            if new_provider != st.session_state.provider or new_model != st.session_state.model:
                st.session_state.provider = new_provider
                st.session_state.model = new_model
                st.rerun()
    with inner_r:
        with st.popover(f"⚙ Retrieval", use_container_width=True):
            st.markdown("##### Retrieval settings")
            new_top_k = st.slider(
                "Top-K chunks", 3, 15, st.session_state.top_k, key="popover_topk",
            )
            new_recent = st.checkbox(
                "Prefer most recent version",
                value=st.session_state.prefer_recent,
                key="popover_recent",
            )
            new_debug = st.checkbox(
                "Show retrieval debug panel",
                value=st.session_state.show_debug,
                key="popover_debug",
            )
            if (
                new_top_k != st.session_state.top_k
                or new_recent != st.session_state.prefer_recent
                or new_debug != st.session_state.show_debug
            ):
                st.session_state.top_k = new_top_k
                st.session_state.prefer_recent = new_recent
                st.session_state.show_debug = new_debug
                st.rerun()


# ===========================================================================
# Handle send
# ===========================================================================
if send_clicked and user_input and user_input.strip():
    chat.messages.append(ChatMessage(role="user", content=user_input.strip()))
    _autorename_if_needed(chat, user_input.strip())

    filters = RetrievalFilters(
        doc_ids=st.session_state.selected_doc_ids or [],
        prefer_recent=st.session_state.prefer_recent,
    )
    try:
        ans = query(
            user_input.strip(),
            filters=filters,
            provider=st.session_state.provider,
            model=st.session_state.model,
            top_k=st.session_state.top_k,
            max_tokens=1500,
        )
    except Exception as e:
        chat.messages.append(ChatMessage(
            role="assistant",
            content=f"❌ Query failed: `{type(e).__name__}`: {e}",
        ))
        st.rerun()
    chat.messages.append(ChatMessage(
        role="assistant",
        content=ans.answer,
        citations=ans.citations,
        retrieved=ans.retrieved,
        latency_ms=ans.latency_ms,
        retrieval_ms=ans.retrieval_ms,
        generation_ms=ans.generation_ms,
        llm_provider=ans.llm_provider,
        llm_model=ans.llm_model,
    ))
    st.rerun()
