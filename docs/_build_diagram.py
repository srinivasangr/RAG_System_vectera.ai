"""Generate docs/architecture.drawio with embedded vendor logos.

Run from repo root:   python docs/_build_diagram.py

Logos are read from docs/assets/logos/, base64-encoded, and inlined as
draw.io `shape=image` cells so the .drawio file is self-contained — anyone
opening it in https://app.diagrams.net sees the logos immediately without
needing the asset directory.

The diagram has two clearly separated swim-lanes:
  * INGEST plane (top)  — offline, batch
  * QUERY  plane (bot)  — online, per request
plus a central Snowflake column shared by both.

Edit the LAYOUT section if you want to move boxes; everything else is
generated.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGOS_DIR = REPO_ROOT / "docs" / "assets" / "logos"
OUT_FILE  = REPO_ROOT / "docs" / "architecture.drawio"


# ---------------------------------------------------------------------------
# Logo registry — short alias → filename. Add new ones here.
# ---------------------------------------------------------------------------
LOGOS = {
    "snowflake":  "snowflake-color.png",
    "streamlit":  "streamlit.svg",
    "docling":    "docling.svg",
    "cerebras":   "Cerebras_logo.svg.png",
    "gemini":     "Google-gemini-icon.svg.png",
    "hf":         "huggingface.png",
    "github":     "github-svgrepo-com.svg",
    "pdf":        "pdf-file-type.svg",
    "openai":     "openai.svg",
}


def _data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None:
        mime = "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


# Pre-encode every logo once
ENCODED = {alias: _data_uri(LOGOS_DIR / fname) for alias, fname in LOGOS.items()}


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------
def _logo_cell(cid: str, alias: str, x: int, y: int, w: int = 80, h: int = 80, label: str = "") -> str:
    """A square 'logo' cell with optional caption below."""
    return dedent(f"""\
        <mxCell id="{cid}" value="{label}" style="shape=image;verticalLabelPosition=bottom;
            labelBackgroundColor=#FFFFFF;verticalAlign=top;
            fontSize=11;fontStyle=1;
            imageAspect=1;aspect=fixed;
            image={ENCODED[alias]}" vertex="1" parent="1">
          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>
        </mxCell>""").replace("\n", " ")


def _box(cid: str, label: str, x: int, y: int, w: int, h: int,
         fill: str, stroke: str, font_size: int = 12, font_color: str = "#000000") -> str:
    return dedent(f"""\
        <mxCell id="{cid}" value="{label}" style="rounded=1;whiteSpace=wrap;html=1;
            fillColor={fill};strokeColor={stroke};
            fontSize={font_size};fontColor={font_color};align=center;verticalAlign=middle" vertex="1" parent="1">
          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>
        </mxCell>""").replace("\n", " ")


def _swimlane(cid: str, title: str, x: int, y: int, w: int, h: int,
              fill: str, stroke: str) -> str:
    return dedent(f"""\
        <mxCell id="{cid}" value="{title}" style="rounded=0;whiteSpace=wrap;html=1;
            fillColor={fill};strokeColor={stroke};fillOpacity=20;
            verticalAlign=top;fontSize=14;fontStyle=1;align=left;
            spacingLeft=12;spacingTop=8" vertex="1" parent="1">
          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>
        </mxCell>""").replace("\n", " ")


def _text(cid: str, label: str, x: int, y: int, w: int = 200, h: int = 20,
          font_size: int = 11, font_color: str = "#444444", bold: bool = False) -> str:
    style = (f"text;html=1;align=center;verticalAlign=middle;"
             f"fontSize={font_size};fontColor={font_color}")
    if bold:
        style += ";fontStyle=1"
    return dedent(f"""\
        <mxCell id="{cid}" value="{label}" style="{style}" vertex="1" parent="1">
          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>
        </mxCell>""").replace("\n", " ")


def _arrow(cid: str, src: str, tgt: str, label: str = "",
           stroke: str = "#666666", dashed: bool = False, width: int = 2) -> str:
    style = (f"endArrow=classic;html=1;rounded=0;"
             f"strokeColor={stroke};strokeWidth={width};")
    if dashed:
        style += "dashed=1;dashPattern=8 4;"
    label_xml = (
        f'<mxCell id="{cid}-lbl" value="{label}" style="edgeLabel;html=1;align=center;verticalAlign=middle;'
        f'fontSize=10;fontColor={stroke};background=#FFFFFF" vertex="1" connectable="0" parent="{cid}">'
        f'<mxGeometry x="-0.05" relative="1" as="geometry"><mxPoint as="offset"/></mxGeometry></mxCell>'
        if label else ""
    )
    return dedent(f"""\
        <mxCell id="{cid}" style="{style}" edge="1" parent="1" source="{src}" target="{tgt}">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>{label_xml}""")


# ---------------------------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------------------------
# Canvas is 1600 x 1100. Two horizontal swim-lanes:
#   Ingest (top, y=120..420)    — offline, batch
#   Query  (bottom, y=560..880) — online, per request
# Snowflake column lives between them so both connect to it.
# ---------------------------------------------------------------------------
INGEST_Y, QUERY_Y = 150, 600
LANE_H = 320

CELLS: list[str] = []

# Title
CELLS.append(_text("title",  "RAG System — High-Level Architecture",
                   600, 30, 600, 30, font_size=22, bold=True, font_color="#222222"))
CELLS.append(_text("subtitle", "Snowflake-backed RAG over PDF investor decks · local embeddings · swappable LLM providers",
                   400, 64, 1000, 22, font_size=12, font_color="#666666"))

# Swim-lanes
CELLS.append(_swimlane("lane-ingest", "INGEST PLANE (offline, batch — runs when a PDF is added)",
                       60, INGEST_Y, 1480, LANE_H, "#FFE5B4", "#B46504"))
CELLS.append(_swimlane("lane-query",  "QUERY PLANE (online, per chat message)",
                       60, QUERY_Y, 1480, LANE_H, "#D5E8D4", "#82B366"))

# --- Ingest row: PDFs → Docling → Chunk → BGE → Snowflake (Gemini side-arrow optional)
y_logo = INGEST_Y + 90
CELLS.append(_logo_cell("pdf-in",     "pdf",        120,  y_logo, 80, 80, "PDF docs"))
CELLS.append(_logo_cell("docling",    "docling",    320,  y_logo, 80, 80, "Docling"))
CELLS.append(_box("chunk-box", "Page-aware chunker&#10;~800 tok / 100 overlap&#10;prose · table · chart",
                  500, y_logo - 10, 200, 100, "#FFFFFF", "#B46504", 11))
CELLS.append(_logo_cell("bge-ing",    "hf",         770,  y_logo, 80, 80, "BGE-base&#10;(embed)"))
CELLS.append(_logo_cell("snow-ing",   "snowflake", 1020,  y_logo, 80, 80, "Snowflake&#10;upsert"))
CELLS.append(_logo_cell("gemini-ing", "gemini",    1240,  y_logo, 80, 80, "Gemini Vision&#10;(optional, charts)"))

# Ingest arrows
CELLS.append(_arrow("a-pdf-doc",   "pdf-in",  "docling",   "parse"))
CELLS.append(_arrow("a-doc-chunk", "docling", "chunk-box"))
CELLS.append(_arrow("a-chunk-bge", "chunk-box","bge-ing"))
CELLS.append(_arrow("a-bge-snow",  "bge-ing", "snow-ing",  "768d vectors"))
CELLS.append(_arrow("a-doc-gem",   "docling", "gemini-ing","chart images", stroke="#aaaaaa", dashed=True))
CELLS.append(_arrow("a-gem-snow",  "gemini-ing","snow-ing", "descriptions", stroke="#aaaaaa", dashed=True))

# --- Query row: User → Streamlit → BGE query → Snowflake → Cerebras → Answer
y_q = QUERY_Y + 90
CELLS.append(_box("user-box", "👤  Analyst",
                  120, y_q + 5, 80, 80, "#DAE8FC", "#6C8EBF", 11))
CELLS.append(_logo_cell("streamlit",  "streamlit",  280, y_q, 80, 80, "Streamlit&#10;UI"))
CELLS.append(_logo_cell("bge-q",      "hf",         480, y_q, 80, 80, "BGE-base&#10;(embed query)"))
CELLS.append(_logo_cell("snow-q",     "snowflake",  680, y_q, 80, 80, "Snowflake&#10;cosine + LIKE"))
CELLS.append(_box("rrf-box", "RRF fuse&#10;→ top-K chunks",
                  880, y_q, 140, 80, "#FFFFFF", "#82B366", 11))
CELLS.append(_logo_cell("cerebras",   "cerebras",  1080, y_q, 80, 80, "Cerebras&#10;gpt-oss-120b"))
CELLS.append(_box("answer-box", "Answer&#10;+ [N] citations",
                  1260, y_q, 140, 80, "#DAE8FC", "#6C8EBF", 11))

# Query arrows
CELLS.append(_arrow("q1", "user-box",  "streamlit",  "question"))
CELLS.append(_arrow("q2", "streamlit", "bge-q"))
CELLS.append(_arrow("q3", "bge-q",     "snow-q",     "vector"))
CELLS.append(_arrow("q4", "snow-q",    "rrf-box",    "candidates"))
CELLS.append(_arrow("q5", "rrf-box",   "cerebras",   "prompt + sources"))
CELLS.append(_arrow("q6", "cerebras",  "answer-box"))
CELLS.append(_arrow("q7", "answer-box","streamlit",  "render", stroke="#aaaaaa"))

# --- Hosting strip (top right) ---
CELLS.append(_logo_cell("gh-repo",   "github",    1420, 100, 56, 56, "GitHub"))

# --- Storage cluster (cross-cutting Snowflake detail box) ---
CELLS.append(_box("sf-detail",
                  "Snowflake (RAG_DB.RAG_SCHEMA)&#10;documents · chunks (VECTOR(768)) · chunk_images · query_log",
                  340, 990, 920, 60,
                  "#CCE5FF", "#1C6EA4", 12))
CELLS.append(_arrow("sf-d-up",  "sf-detail", "snow-ing", stroke="#1c6ea4", dashed=True, width=1))
CELLS.append(_arrow("sf-d-dn",  "sf-detail", "snow-q",   stroke="#1c6ea4", dashed=True, width=1))


# ---------------------------------------------------------------------------
# Assemble the .drawio XML
# ---------------------------------------------------------------------------
XML = f"""<mxfile host="app.diagrams.net" agent="rag-system-diagram-generator" version="24.7.17">
  <diagram id="rag-architecture" name="High-Level Architecture">
    <mxGraphModel dx="1422" dy="824" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1600" pageHeight="1100" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {chr(10).join(CELLS)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""

OUT_FILE.write_text(XML, encoding="utf-8")
print(f"wrote {OUT_FILE}  ({len(XML):,} bytes)")
print(f"  embedded {len(ENCODED)} logos: {', '.join(ENCODED)}")
