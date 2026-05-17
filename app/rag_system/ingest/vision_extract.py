"""Vision-LLM pass: turn each chart/figure image into a rich text description.

Strategy:
  - Skip images that are too small (logos, decorative bullets) — they waste
    a vision call and add noise. Threshold: < 200px on the shortest side.
  - Skip images with extreme aspect ratios (banners, dividers).
  - For each remaining image, ask Gemini Flash for a structured description
    (axes, trend, key numbers). If the model says NOT_A_CHART, drop it.

Free-tier rate limits (Gemini 2.5 Flash: 250 RPD) are respected by
filtering aggressively and by exposing a max-calls cap from the caller.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from rag_system.ingest.parse import PageImage
from rag_system.llm_providers import get_vision

log = logging.getLogger(__name__)

# Free-tier Gemini 2.5 Flash is actually 5 RPM (the docs we relied on earlier
# said 10; the API enforces 5). With client-side throttling in the provider we
# only run 1 vision call at a time and let the RateLimiter sleep between calls.
MAX_CONCURRENT_VISION = 1


CHART_PROMPT = """\
You are extracting structured information from a figure in a financial \
investor presentation.

Examine the image. If it is NOT a data figure (e.g. it is a logo, photograph, \
decorative graphic, or pure text), reply with exactly:

NOT_A_CHART

Otherwise, describe the figure in Markdown with these sections — be concise \
and specific, do not invent numbers you cannot see:

**Figure type:** (bar chart / line chart / pie chart / table / diagram / map / other)
**Title or topic:** (if visible)
**Axes / categories:** (x-axis label, y-axis label, units, time range)
**Key data points:** (the specific numbers visible — list them)
**Trend or takeaway:** (one sentence; only what the figure actually shows)
"""

MIN_SIDE_PX = 200
MAX_ASPECT_RATIO = 6.0  # skip very wide banners and very tall sidebars


def _should_describe(img: PageImage) -> bool:
    short = min(img.width, img.height)
    long = max(img.width, img.height)
    if short < MIN_SIDE_PX:
        return False
    if long / max(short, 1) > MAX_ASPECT_RATIO:
        return False
    return True


def _describe_one(im: PageImage, vision) -> tuple[PageImage, str | None]:
    """Worker: describe a single image. Returns (image, description-or-None)."""
    try:
        desc = vision.describe_image(
            image_bytes=im.png_bytes,
            prompt=CHART_PROMPT,
            mime_type="image/png",
        )
    except Exception as e:
        log.warning("vision failed p%s img%s: %s", im.page_number, im.image_index, e)
        return (im, None)
    if not desc or desc.strip().upper().startswith("NOT_A_CHART"):
        return (im, None)
    return (im, desc.strip())


def describe_images(
    images: list[PageImage],
    *,
    max_calls: int | None = None,
    sleep_between: float = 0.0,   # legacy param, ignored when parallel
    progress_cb=None,
    max_workers: int = MAX_CONCURRENT_VISION,
) -> dict[tuple[int, int], tuple[str, bytes, int, int]]:
    """Return (page_number, image_index) → (description, png_bytes, width, height).

    Vision calls run concurrently (max_workers threads). Filtered-out and
    NOT_A_CHART images are absent from the map.

    progress_cb (optional): called as progress_cb("vision_progress",
    {"done": int, "total": int, "described": int}) after each result.
    """
    if not images:
        return {}

    candidates = [im for im in images if _should_describe(im)]
    if max_calls is not None:
        candidates = candidates[:max_calls]

    if not candidates:
        return {}

    if progress_cb:
        progress_cb("vision_start", {"total": len(candidates)})

    vision = get_vision()
    out: dict[tuple[int, int], tuple[str, bytes, int, int]] = {}
    done = 0
    described = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_describe_one, im, vision) for im in candidates]
        for fut in as_completed(futures):
            im, desc = fut.result()
            done += 1
            if desc:
                described += 1
                out[(im.page_number, im.image_index)] = (
                    desc, im.png_bytes, im.width, im.height,
                )
            if progress_cb:
                progress_cb("vision_progress", {
                    "done": done, "total": len(candidates), "described": described,
                })

    if progress_cb:
        progress_cb("vision_done", {"total": len(candidates), "described": described})
    return out
