"""Generate two architecture artefacts from the same source-of-truth layout:

  1) docs/architecture.drawio                 — editable, with embedded logos
  2) docs/assets/architecture_overview.png    — flat PNG for inline README rendering

Both share the LAYOUT block below, so a node moved here updates both outputs.

Run from repo root:    python docs/_build_diagram.py
"""

from __future__ import annotations

import base64
import math
import mimetypes
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGOS_DIR = REPO_ROOT / "docs" / "assets" / "logos"
OUT_DRAWIO = REPO_ROOT / "docs" / "architecture.drawio"
OUT_PNG    = REPO_ROOT / "docs" / "assets" / "architecture_overview.png"


# ---------------------------------------------------------------------------
# Logo registry
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
}


# ---------------------------------------------------------------------------
# LAYOUT — single source of truth for both outputs
# ---------------------------------------------------------------------------
@dataclass
class Node:
    id: str
    kind: str           # "logo" | "box"
    label: str
    sublabel: str = ""  # second line under the label (PNG only — drawio uses \n in label)
    logo: str | None = None
    x: int = 0          # top-left in canvas coords
    y: int = 0
    w: int = 120
    h: int = 120
    fill: str = "#FFFFFF"
    stroke: str = "#666666"


@dataclass
class Edge:
    src: str
    tgt: str
    label: str = ""
    dashed: bool = False
    color: str = "#666666"


# Canvas
CANVAS_W = 1600
CANVAS_H = 1100

INGEST_Y, QUERY_Y = 180, 620
LANE_H = 320

NODES: list[Node] = [
    # ----- Ingest plane (top) -----
    Node("pdf",     "logo", "PDF docs",          logo="pdf",       x=130,  y=INGEST_Y + 80, w=100, h=100),
    Node("docling", "logo", "Docling",           sublabel="parser",     logo="docling",   x=320,  y=INGEST_Y + 80, w=100, h=100),
    Node("chunk",   "box",  "Page-aware chunker",sublabel="~800 tok · prose / table / chart",
                                                                          x=490,  y=INGEST_Y + 95, w=240, h=80,
                                                                          fill="#FFFFFF", stroke="#B46504"),
    Node("bge_ing", "logo", "BGE-base",          sublabel="embed",      logo="hf",        x=800,  y=INGEST_Y + 80, w=100, h=100),
    Node("snow_ing","logo", "Snowflake",         sublabel="upsert",     logo="snowflake", x=1010, y=INGEST_Y + 80, w=100, h=100),
    Node("gemini",  "logo", "Gemini Vision",     sublabel="(optional)", logo="gemini",    x=1230, y=INGEST_Y + 80, w=100, h=100),

    # ----- Query plane (bottom) -----
    Node("user",    "box",  "Analyst",           sublabel="👤",          x=120,  y=QUERY_Y + 95, w=100, h=80,
                                                                          fill="#DAE8FC", stroke="#6C8EBF"),
    Node("streamlit","logo","Streamlit",         sublabel="UI",         logo="streamlit", x=290,  y=QUERY_Y + 80, w=100, h=100),
    Node("bge_q",   "logo", "BGE-base",          sublabel="embed query",logo="hf",        x=470,  y=QUERY_Y + 80, w=100, h=100),
    Node("snow_q",  "logo", "Snowflake",         sublabel="cosine + LIKE",logo="snowflake", x=660,  y=QUERY_Y + 80, w=100, h=100),
    Node("rrf",     "box",  "RRF fuse",          sublabel="→ top-K chunks",
                                                                          x=850, y=QUERY_Y + 95, w=160, h=80,
                                                                          fill="#FFFFFF", stroke="#82B366"),
    Node("cerebras","logo", "Cerebras",          sublabel="gpt-oss-120b", logo="cerebras", x=1060, y=QUERY_Y + 80, w=100, h=100),
    Node("answer",  "box",  "Answer",            sublabel="+ [N] citations",
                                                                          x=1240, y=QUERY_Y + 95, w=160, h=80,
                                                                          fill="#DAE8FC", stroke="#6C8EBF"),

    # ----- Hosting -----
    Node("github",  "logo", "GitHub",            sublabel="auto-deploy",logo="github",    x=1430, y=110,            w=80,  h=80),

    # ----- Snowflake detail (cross-cutting) -----
    Node("sf_detail","box", "Snowflake — RAG_DB.RAG_SCHEMA",
                            sublabel="documents · chunks (VECTOR 768) · chunk_images · query_log",
                                                                          x=340, y=990, w=920, h=70,
                                                                          fill="#CCE5FF", stroke="#1C6EA4"),
]

EDGES: list[Edge] = [
    # Ingest
    Edge("pdf",     "docling",  "parse"),
    Edge("docling", "chunk"),
    Edge("chunk",   "bge_ing"),
    Edge("bge_ing", "snow_ing", "768d vectors"),
    Edge("docling", "gemini",   "chart images", dashed=True, color="#888888"),
    Edge("gemini",  "snow_ing", "descriptions", dashed=True, color="#888888"),

    # Query
    Edge("user",      "streamlit", "question"),
    Edge("streamlit", "bge_q"),
    Edge("bge_q",     "snow_q",    "vector"),
    Edge("snow_q",    "rrf",       "candidates"),
    Edge("rrf",       "cerebras",  "prompt + sources"),
    Edge("cerebras",  "answer"),
    Edge("answer",    "streamlit", "render", color="#888888"),

    # Snowflake detail
    Edge("sf_detail", "snow_ing", "", dashed=True, color="#1C6EA4"),
    Edge("sf_detail", "snow_q",   "", dashed=True, color="#1C6EA4"),
]


# ---------------------------------------------------------------------------
# PART 1 — draw.io XML (with embedded logos)
# ---------------------------------------------------------------------------
def _data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None:
        mime = "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


ENCODED = {alias: _data_uri(LOGOS_DIR / fname) for alias, fname in LOGOS.items()}


def _drawio_style_logo(alias: str) -> str:
    # draw.io style strings MUST be single-line, no extra whitespace anywhere
    return (
        "shape=image;"
        "verticalLabelPosition=bottom;labelBackgroundColor=#FFFFFF;"
        "verticalAlign=top;fontSize=11;fontStyle=1;"
        "imageAspect=1;aspect=fixed;"
        f"image={ENCODED[alias]}"
    )


def _drawio_style_box(fill: str, stroke: str) -> str:
    return (
        "rounded=1;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};"
        "fontSize=12;align=center;verticalAlign=middle"
    )


def _drawio_style_lane(fill: str, stroke: str) -> str:
    return (
        "rounded=0;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};fillOpacity=20;"
        "verticalAlign=top;fontSize=14;fontStyle=1;align=left;"
        "spacingLeft=12;spacingTop=8"
    )


def _drawio_style_edge(color: str, dashed: bool, width: int = 2) -> str:
    s = f"endArrow=classic;html=1;rounded=0;strokeColor={color};strokeWidth={width};"
    if dashed:
        s += "dashed=1;dashPattern=8 4;"
    return s


def _drawio_style_text(font_size: int, color: str, bold: bool = False) -> str:
    s = (
        "text;html=1;align=center;verticalAlign=middle;"
        f"fontSize={font_size};fontColor={color}"
    )
    if bold:
        s += ";fontStyle=1"
    return s


def _build_drawio() -> str:
    cells: list[str] = []

    # Title
    cells.append(
        f'<mxCell id="title" value="RAG System — High-Level Architecture" '
        f'style="{_drawio_style_text(22, "#222222", True)}" vertex="1" parent="1">'
        f'<mxGeometry x="600" y="30" width="600" height="30" as="geometry"/></mxCell>'
    )
    cells.append(
        f'<mxCell id="subtitle" value="Snowflake-backed RAG over PDF investor decks · '
        f'local embeddings · swappable LLM providers" '
        f'style="{_drawio_style_text(12, "#666666")}" vertex="1" parent="1">'
        f'<mxGeometry x="400" y="64" width="1000" height="22" as="geometry"/></mxCell>'
    )

    # Lanes
    cells.append(
        f'<mxCell id="lane-ingest" value="INGEST PLANE (offline, batch — runs when a PDF is added)" '
        f'style="{_drawio_style_lane("#FFE5B4", "#B46504")}" vertex="1" parent="1">'
        f'<mxGeometry x="60" y="{INGEST_Y}" width="1480" height="{LANE_H}" as="geometry"/></mxCell>'
    )
    cells.append(
        f'<mxCell id="lane-query" value="QUERY PLANE (online, per chat message)" '
        f'style="{_drawio_style_lane("#D5E8D4", "#82B366")}" vertex="1" parent="1">'
        f'<mxGeometry x="60" y="{QUERY_Y}" width="1480" height="{LANE_H}" as="geometry"/></mxCell>'
    )

    # Nodes
    for n in NODES:
        if n.kind == "logo":
            label = n.label + (f"\\n{n.sublabel}" if n.sublabel else "")
            style = _drawio_style_logo(n.logo)
        else:
            label = n.label + (f"\\n{n.sublabel}" if n.sublabel else "")
            style = _drawio_style_box(n.fill, n.stroke)
        cells.append(
            f'<mxCell id="{n.id}" value="{label}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{n.x}" y="{n.y}" width="{n.w}" height="{n.h}" as="geometry"/></mxCell>'
        )

    # Edges
    for i, e in enumerate(EDGES):
        eid = f"e{i}"
        style = _drawio_style_edge(e.color, e.dashed)
        cells.append(
            f'<mxCell id="{eid}" style="{style}" edge="1" parent="1" '
            f'source="{e.src}" target="{e.tgt}">'
            f'<mxGeometry relative="1" as="geometry"/></mxCell>'
        )
        if e.label:
            cells.append(
                f'<mxCell id="{eid}-lbl" value="{e.label}" '
                f'style="edgeLabel;html=1;align=center;verticalAlign=middle;'
                f'fontSize=10;fontColor={e.color};background=#FFFFFF" '
                f'vertex="1" connectable="0" parent="{eid}">'
                f'<mxGeometry x="-0.05" relative="1" as="geometry">'
                f'<mxPoint as="offset"/></mxGeometry></mxCell>'
            )

    return (
        '<mxfile host="app.diagrams.net" agent="rag-diagram-generator" version="24.7.17">'
        '<diagram id="rag-architecture" name="High-Level Architecture">'
        f'<mxGraphModel dx="1422" dy="824" grid="1" gridSize="10" guides="1" tooltips="1" '
        f'connect="1" arrows="1" fold="1" page="1" pageScale="1" '
        f'pageWidth="{CANVAS_W}" pageHeight="{CANVAS_H}" math="0" shadow="0">'
        '<root><mxCell id="0"/><mxCell id="1" parent="0"/>'
        + "".join(cells) +
        '</root></mxGraphModel></diagram></mxfile>'
    )


# ---------------------------------------------------------------------------
# PART 2 — PNG via Pillow (flat poster for inline README rendering)
# ---------------------------------------------------------------------------
def _load_logo_image(alias: str, target_size: int) -> Image.Image | None:
    """Load a logo file as RGBA, resized to fit within target_size×target_size."""
    path = LOGOS_DIR / LOGOS[alias]
    if not path.exists():
        return None
    if path.suffix.lower() == ".svg":
        # Pillow doesn't open SVG natively. Try cairosvg if available; otherwise
        # fall back to a text label.
        try:
            import cairosvg  # noqa: F401
            from io import BytesIO
            png_bytes = cairosvg.svg2png(
                url=str(path), output_width=target_size, output_height=target_size,
            )
            return Image.open(BytesIO(png_bytes)).convert("RGBA")
        except Exception:
            return None
    img = Image.open(path).convert("RGBA")
    # Keep aspect ratio
    img.thumbnail((target_size, target_size), Image.LANCZOS)
    return img


def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    # Try common fonts; fall back to default bitmap font (still readable).
    candidates_bold = ["arialbd.ttf", "DejaVuSans-Bold.ttf", "Helvetica-Bold.ttf"]
    candidates_reg  = ["arial.ttf", "DejaVuSans.ttf", "Helvetica.ttf"]
    names = candidates_bold if bold else candidates_reg
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_arrow(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int,
                color: str = "#666666", dashed: bool = False, width: int = 2):
    if dashed:
        # Manual dashing
        n_dashes = max(1, int(math.hypot(x2 - x1, y2 - y1) / 12))
        for i in range(n_dashes):
            t1, t2 = i / n_dashes, (i + 0.5) / n_dashes
            sx, sy = x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1
            ex, ey = x1 + (x2 - x1) * t2, y1 + (y2 - y1) * t2
            draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
    else:
        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    # Arrowhead
    ang = math.atan2(y2 - y1, x2 - x1)
    head = 12
    p1 = (x2 - head * math.cos(ang - math.pi / 6),
          y2 - head * math.sin(ang - math.pi / 6))
    p2 = (x2 - head * math.cos(ang + math.pi / 6),
          y2 - head * math.sin(ang + math.pi / 6))
    draw.polygon([(x2, y2), p1, p2], fill=color)


def _node_center(n: Node) -> tuple[int, int]:
    return (n.x + n.w // 2, n.y + n.h // 2)


def _node_anchor(n: Node, side: str) -> tuple[int, int]:
    """Return a point on the given edge of the node (for cleaner arrow joins)."""
    cx, cy = _node_center(n)
    if side == "left":   return (n.x, cy)
    if side == "right":  return (n.x + n.w, cy)
    if side == "top":    return (cx, n.y)
    if side == "bottom": return (cx, n.y + n.h)
    return (cx, cy)


def _pick_sides(src: Node, tgt: Node) -> tuple[str, str]:
    """Decide which sides to anchor on based on relative position."""
    sx, sy = _node_center(src); tx, ty = _node_center(tgt)
    if abs(tx - sx) >= abs(ty - sy):
        return ("right", "left") if tx >= sx else ("left", "right")
    return ("bottom", "top") if ty >= sy else ("top", "bottom")


def _build_png() -> None:
    bg = Image.new("RGBA", (CANVAS_W, CANVAS_H), (255, 255, 255, 255))
    draw = ImageDraw.Draw(bg)

    # ---- Title ----
    title_font = _get_font(28, bold=True)
    sub_font   = _get_font(14)
    draw.text((CANVAS_W // 2 - 280, 20), "RAG System — High-Level Architecture",
              fill="#222222", font=title_font)
    draw.text((CANVAS_W // 2 - 360, 60),
              "Snowflake-backed RAG over PDF investor decks · local embeddings · "
              "swappable LLM providers",
              fill="#666666", font=sub_font)

    # ---- Lanes ----
    def _lane(x, y, w, h, fill, stroke, label):
        # Semi-transparent fill
        overlay = Image.new("RGBA", (w, h), fill + "33")  # alpha 0x33
        bg.alpha_composite(overlay, (x, y))
        # Border
        draw.rectangle([x, y, x + w, y + h], outline=stroke, width=2)
        # Title strip
        lane_font = _get_font(16, bold=True)
        draw.text((x + 14, y + 8), label, fill=stroke, font=lane_font)

    _lane(60, INGEST_Y, 1480, LANE_H, "#FFE5B4", "#B46504",
          "INGEST PLANE (offline, batch — runs when a PDF is added)")
    _lane(60, QUERY_Y, 1480, LANE_H, "#D5E8D4", "#82B366",
          "QUERY PLANE (online, per chat message)")

    # ---- Nodes ----
    label_font = _get_font(14, bold=True)
    sub_font_s = _get_font(11)

    node_by_id = {n.id: n for n in NODES}

    for n in NODES:
        if n.kind == "box":
            # Filled rounded-ish rect (just rectangle here)
            draw.rectangle([n.x, n.y, n.x + n.w, n.y + n.h], fill=n.fill, outline=n.stroke, width=2)
            # Label (centered, 1 or 2 lines)
            lines = [n.label] + ([n.sublabel] if n.sublabel else [])
            line_h = 18
            total_h = line_h * len(lines)
            for i, line in enumerate(lines):
                lw = draw.textlength(line, font=label_font if i == 0 else sub_font_s)
                draw.text(
                    (n.x + (n.w - lw) // 2,
                     n.y + (n.h - total_h) // 2 + i * line_h),
                    line, fill="#222222",
                    font=label_font if i == 0 else sub_font_s,
                )
        else:
            # Logo cell
            logo_size = min(n.w, n.h) - 30  # leave room for label below
            img = _load_logo_image(n.logo, logo_size) if n.logo else None
            if img is not None:
                lx = n.x + (n.w - img.width) // 2
                ly = n.y + 5
                bg.alpha_composite(img, (lx, ly))
                label_y = ly + img.height + 4
            else:
                # Fallback: draw a placeholder rectangle
                draw.rectangle([n.x + 10, n.y + 10, n.x + n.w - 10, n.y + n.h - 28],
                              outline=n.stroke, width=1)
                label_y = n.y + n.h - 24

            lw = draw.textlength(n.label, font=label_font)
            draw.text((n.x + (n.w - lw) // 2, label_y),
                      n.label, fill="#222222", font=label_font)
            if n.sublabel:
                sw = draw.textlength(n.sublabel, font=sub_font_s)
                draw.text((n.x + (n.w - sw) // 2, label_y + 17),
                          n.sublabel, fill="#666666", font=sub_font_s)

    # ---- Edges ----
    edge_label_font = _get_font(11)
    for e in EDGES:
        src = node_by_id[e.src]; tgt = node_by_id[e.tgt]
        side_s, side_t = _pick_sides(src, tgt)
        x1, y1 = _node_anchor(src, side_s)
        x2, y2 = _node_anchor(tgt, side_t)
        _draw_arrow(draw, x1, y1, x2, y2, color=e.color, dashed=e.dashed, width=2)
        if e.label:
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2 - 8
            lw = draw.textlength(e.label, font=edge_label_font)
            # White background for legibility
            draw.rectangle(
                [mx - lw / 2 - 4, my - 2, mx + lw / 2 + 4, my + 14],
                fill="#FFFFFF",
            )
            draw.text((mx - lw / 2, my), e.label, fill=e.color, font=edge_label_font)

    bg.convert("RGB").save(OUT_PNG, "PNG", optimize=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DRAWIO.write_text(_build_drawio(), encoding="utf-8")
    print(f"wrote {OUT_DRAWIO}  ({OUT_DRAWIO.stat().st_size:,} bytes)")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    _build_png()
    print(f"wrote {OUT_PNG}  ({OUT_PNG.stat().st_size:,} bytes)")
    print(f"  embedded logos: {', '.join(LOGOS)}")


if __name__ == "__main__":
    main()
