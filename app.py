"""Chromatic Sync â€” Garment Color Detection & Outfit Compatibility Analysis.

Pipeline (upload mode only):
  1. Upload a garment image
  2. GrabCut background removal
  3. K-Means dominant color extraction on masked pixels
  4. HSV histogram for color distribution
  5. Color harmony recommendations (complementary / analogous / triadic)
  6. Retrieve compatible outfit images from DeepFashion2 by HSV histogram similarity
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from modules.preprocessing import (
    load_image,
    resize_for_analysis,
    grabcut_foreground,
    classify_wear_type_from_mask,
)
from modules.segmentation import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_IMAGE_LIMIT,
    WearType,
    CATEGORY_MAP,
    list_training_images,
    mask_coverage,
)
from modules.color_analysis import (
    extract_dominant_colors,
    compute_hsv_histogram,
    DominantColor,
)
from modules.color_harmony import compare_palettes
from modules.recommendations import recommend_colors
from modules.retrieval import load_or_build_index, retrieve_compatible

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Page config
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

st.set_page_config(
    page_title="Chromatic Sync",
    page_icon="ðŸŽ¨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CSS â€” injected via a single clean <style> block
# Use st.html() so Streamlit never sanitises it
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

CSS = """
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap"
      rel="stylesheet">
<style>
:root {
  --bg-primary:    #0d0d14;
  --bg-secondary:  #13131f;
  --bg-card:       rgba(255,255,255,0.05);
  --border-glass:  rgba(255,255,255,0.10);
  --accent-violet: #7c3aed;
  --accent-teal:   #0d9488;
  --accent-rose:   #e11d48;
  --text-primary:  #f0f0f8;
  --text-muted:    #8888aa;
  --radius-lg:     16px;
  --radius-md:     10px;
  --shadow-glow:   0 0 40px rgba(124,58,237,0.25);
}

/* â”€â”€ Global â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
html, body, .stApp {
  font-family: 'Outfit', 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif !important;
  background: var(--bg-primary) !important;
  color: var(--text-primary) !important;
}
.main .block-container { padding-top: 2rem !important; }

/* â”€â”€ Sidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
section[data-testid="stSidebar"] {
  background: var(--bg-secondary) !important;
  border-right: 1px solid var(--border-glass) !important;
}

/* â”€â”€ Headings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
h1,h2,h3,h4 { color: var(--text-primary) !important; font-family: 'Outfit', 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif !important; }
h1 { font-weight: 800 !important; font-size: 2.4rem !important; letter-spacing: -0.5px; }
h2 { font-weight: 700 !important; }
h3 { font-weight: 600 !important; }

/* â”€â”€ Glass card â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.glass-card {
  background: var(--bg-card);
  border: 1px solid var(--border-glass);
  border-radius: var(--radius-lg);
  padding: 1.4rem 1.6rem;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  margin-bottom: 1rem;
  transition: box-shadow 0.3s ease;
}
.glass-card:hover { box-shadow: var(--shadow-glow); }

/* â”€â”€ Badge â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 14px; border-radius: 999px;
  font-size: 0.78rem; font-weight: 600; letter-spacing: 0.5px;
  text-transform: uppercase; white-space: nowrap;
}
.badge-top    { background:rgba(124,58,237,.20); color:#c4b5fd; border:1px solid rgba(124,58,237,.4); }
.badge-bottom { background:rgba(13,148,136,.20); color:#5eead4; border:1px solid rgba(13,148,136,.4); }
.badge-full   { background:rgba(225,29,72,.20);  color:#fda4af; border:1px solid rgba(225,29,72,.4); }
.badge-unknown{ background:rgba(100,100,130,.20);color:#a8a8cc; border:1px solid rgba(100,100,130,.4);}

/* â”€â”€ Score ring â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.score-ring-wrapper {
  display:flex; flex-direction:column;
  align-items:center; justify-content:center; padding:1rem;
}
.score-ring {
  width:130px; height:130px; border-radius:50%;
  display:flex; flex-direction:column; align-items:center; justify-content:center;
}
.score-label {
  font-size:0.72rem; font-weight:600; letter-spacing:1px;
  text-transform:uppercase; color:var(--text-muted); margin-top:4px;
}

/* â”€â”€ Swatches â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.swatch-row { display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; }
.swatch {
  width:48px; height:48px; border-radius:10px;
  border:2px solid rgba(255,255,255,.12);
  transition:transform .2s, box-shadow .2s; cursor:pointer; flex-shrink:0;
}
.swatch:hover { transform:scale(1.18); box-shadow:0 4px 16px rgba(0,0,0,.4); }
.swatch-lg {
  height:64px; border-radius:12px;
  border:2px solid rgba(255,255,255,.12);
  margin-bottom:8px; transition:transform .2s;
}
.swatch-lg:hover { transform:scaleY(1.06); }

/* â”€â”€ Histogram bars â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.hist-bar-outer {
  background:rgba(255,255,255,.06); border-radius:4px;
  height:10px; overflow:hidden; margin:2px 0;
}
.hist-bar-inner { height:100%; border-radius:4px; transition:width .6s; }

/* â”€â”€ Gallery item â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.gallery-item {
  border-radius:var(--radius-md); overflow:hidden;
  border:1px solid var(--border-glass); background:var(--bg-card);
  transition:transform .25s, box-shadow .25s;
}
.gallery-item:hover { transform:translateY(-4px); box-shadow:0 12px 32px rgba(0,0,0,.5); }

/* â”€â”€ Harmony chip â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.harmony-chip {
  display:inline-block; padding:3px 12px; border-radius:999px;
  font-size:0.72rem; font-weight:600; letter-spacing:.4px;
}

/* â”€â”€ Animated gradient headline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
@keyframes gradient-shift {
  0%   { background-position:0%   50%; }
  50%  { background-position:100% 50%; }
  100% { background-position:0%   50%; }
}
.gradient-text {
  background: linear-gradient(270deg,#7c3aed,#0d9488,#e11d48,#f59e0b);
  background-size: 400% 400%;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: gradient-shift 6s ease infinite;
  font-weight: 800;
}

/* â”€â”€ Streamlit component overrides â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.stButton > button {
  background: linear-gradient(135deg,var(--accent-violet),#5b21b6) !important;
  color: white !important; border: none !important;
  border-radius: var(--radius-md) !important;
  font-weight: 600 !important; font-family: 'Outfit', 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif !important;
  padding: 0.55rem 1.4rem !important;
  transition: opacity .2s, transform .15s !important;
}
.stButton > button:hover { opacity:.88 !important; transform:translateY(-1px) !important; }

div[data-testid="metric-container"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-glass) !important;
  border-radius: var(--radius-md) !important;
  padding: 0.75rem 1rem !important;
}

div[data-testid="stFileUploaderDropzone"] {
  background: var(--bg-card) !important;
  border: 1.5px dashed var(--border-glass) !important;
  border-radius: var(--radius-lg) !important;
}

.stProgress > div > div > div > div {
  background: linear-gradient(90deg,var(--accent-violet),var(--accent-teal)) !important;
}

div[data-testid="stAlert"] {
  border-radius: var(--radius-md) !important;
  font-family: 'Outfit', 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif !important;
}

details { border-radius: var(--radius-md) !important; }

div[data-baseweb="tab-list"] {
  background: var(--bg-secondary) !important;
  border-radius: 10px !important; gap: 4px !important;
}

input, .stNumberInput input, .stTextInput input {
  font-family: 'Outfit', sans-serif !important;
  background: var(--bg-card) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border-glass) !important;
  border-radius: var(--radius-md) !important;
}

small, .stCaption { color: var(--text-muted) !important; }
p { color: var(--text-primary) !important; font-family:'Outfit', 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif !important; }
label { color: var(--text-primary) !important; font-family:'Outfit', 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif !important; }
hr { border-color: var(--border-glass) !important; }
</style>
"""

# st.html is the correct way to inject raw HTML/CSS without sanitisation
st.html(CSS)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Constants & helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

WEAR_BADGE: dict[WearType, str] = {
    WearType.TOP:     '<span class="badge badge-top">👕 Top Wear</span>',
    WearType.BOTTOM:  '<span class="badge badge-bottom">👖 Bottom Wear</span>',
    WearType.FULL:    '<span class="badge badge-full">👗 Full / Dress</span>',
    WearType.UNKNOWN: '<span class="badge badge-unknown">❓ Unknown</span>',
}

HARMONY_COLORS: dict[str, str] = {
    "complementary": "#7c3aed",
    "analogous":     "#0d9488",
    "triadic":       "#f59e0b",
    "neutral":       "#6b7280",
}


def _score_color(score: int) -> str:
    if score >= 85:
        return "#10b981"
    if score >= 65:
        return "#f59e0b"
    return "#e11d48"


def _score_ring_html(score: int) -> str:
    color = _score_color(score)
    conic = f"conic-gradient({color} {score * 3.6}deg, rgba(255,255,255,0.06) 0deg)"
    return f"""
    <div class="score-ring-wrapper">
      <div class="score-ring" style="background:{conic};box-shadow:0 0 28px {color}55;">
        <div style="width:96px;height:96px;border-radius:50%;background:var(--bg-primary);
                    display:flex;flex-direction:column;align-items:center;justify-content:center;">
          <span style="font-size:2rem;font-weight:800;color:{color};line-height:1;">{score}</span>
          <span style="font-size:0.65rem;color:var(--text-muted);font-weight:600;letter-spacing:1px;">/ 100</span>
        </div>
      </div>
      <div class="score-label">Compatibility</div>
    </div>"""


def _palette_html(colors: list[DominantColor]) -> str:
    swatches = "".join(
        f'<div class="swatch" style="background:{c.hex_code};" '
        f'title="{c.name.title()} Â· {c.hex_code} Â· {c.proportion:.1%}"></div>'
        for c in colors
    )
    return f'<div class="swatch-row">{swatches}</div>'


def _hue_histogram_html(hist: np.ndarray) -> str:
    import colorsys
    bars = []
    for i, val in enumerate(hist):
        hue = i / 36.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        color = "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))
        width = round(val * 100)
        bars.append(
            f'<div class="hist-bar-outer">'
            f'<div class="hist-bar-inner" style="width:{width}%;background:{color};"></div>'
            f'</div>'
        )
    return '<div style="padding:4px 0;">' + "".join(bars) + "</div>"


def _render_color_card(color: DominantColor) -> None:
    st.markdown(
        f'<div class="swatch-lg" style="background:{color.hex_code};"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**{color.name.title()}**  \n`{color.hex_code}`  \n"
        f"<span style='color:var(--text-muted);font-size:0.8rem;'>"
        f"RGB {color.rgb[0]}, {color.rgb[1]}, {color.rgb[2]} · {color.proportion:.1%}</span>",
        unsafe_allow_html=True,
    )


def _render_rec_card(rec) -> None:
    chip_color = HARMONY_COLORS.get(rec.harmony_type, "#555")
    st.markdown(
        f'<div class="swatch-lg" style="background:{rec.hex_code};"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**{rec.label}**  \n`{rec.hex_code}`  \n"
        f'<span class="harmony-chip" style="background:{chip_color}22;color:{chip_color};'
        f'border:1px solid {chip_color}55;">{rec.harmony_type.capitalize()}</span>',
        unsafe_allow_html=True,
    )
    st.caption(rec.reason)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Sidebar
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

with st.sidebar:
    st.markdown(
        '<h2 class="gradient-text" style="margin-bottom:0;">Chromatic Sync</h2>'
        '<p style="color:var(--text-muted);font-size:0.85rem;margin-top:2px;">'
        'Garment Color Detection &amp; Outfit Compatibility</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    cluster_count = st.slider(
        "Dominant Colors (K)", min_value=2, max_value=8, value=5,
        help="Number of K-Means clusters for dominant color extraction.",
    )
    show_histogram = st.checkbox("Show HSV color histogram", value=True)
    show_retrieval = st.checkbox("Show compatible outfit retrieval", value=True)

    st.divider()

    retrieval_limit = st.slider(
        "Retrieval index size", min_value=100, max_value=2000,
        value=500, step=100,
        help="Smaller = faster indexing. Index is cached after first build.",
    )

    st.divider()
    st.markdown(
        '<p style="color:var(--text-muted);font-size:0.75rem;line-height:1.6;">'
        '🎓 <b>Techniques</b><br>'
        '· K-Means clustering<br>'
        '· HSV / CIELAB color spaces<br>'
        '· GrabCut background removal<br>'
        '· Color harmony theory<br>'
        '· Bhattacharyya histogram retrieval</p>',
        unsafe_allow_html=True,
    )

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Header
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

st.markdown(
    '<h1 class="gradient-text">Chromatic Sync</h1>'
    '<p style="color:var(--text-muted);font-size:1.05rem;margin-top:-6px;">'
    'Upload a garment · Detect color &amp; type · Get perfect outfit matches</p>',
    unsafe_allow_html=True,
)
st.divider()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Upload section
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

upload_col, how_col = st.columns([2, 1])
with upload_col:
    uploaded_file = st.file_uploader(
        "Drop a garment image here",
        type=["jpg", "jpeg", "png", "webp"],
        help="JPEG / PNG / WebP up to 15 MB.",
        key="main_upload",
    )
with how_col:
    st.markdown(
        '<div class="glass-card">'
        '<h3 style="margin-top:0;">ℹ️ How it works</h3>'
        '<ol style="color:var(--text-muted);font-size:0.85rem;padding-left:16px;margin:0;">'
        '<li>GrabCut removes the background</li>'
        '<li>K-Means extracts dominant colors</li>'
        '<li>HSV histogram shows color spread</li>'
        '<li>Color harmony theory for suggestions</li>'
        '<li>Retrieve outfit matches from DF2</li>'
        '</ol></div>',
        unsafe_allow_html=True,
    )

if uploaded_file is None:
    st.markdown(
        '<div class="glass-card" style="text-align:center;padding:3rem;">'
        '<p style="font-size:3rem;margin:0;">👚</p>'
        '<p style="color:var(--text-muted);font-size:1rem;margin-top:8px;">'
        'Upload a garment image to get started</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Load & preprocess
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

try:
    image = load_image(uploaded_file.getvalue())
except (TypeError, ValueError) as err:
    st.error(str(err))
    st.stop()

image_resized = resize_for_analysis(image)

st.markdown("### 🖼️ Image Processing")
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

col_orig, col_gc, col_info = st.columns(3)

with col_orig:
    st.markdown("**Original Image**")
    st.image(image_resized, width='stretch')

with col_gc:
    st.markdown("**GrabCut Foreground**")
    with st.spinner("Removing background…"):
        try:
            gc_result = grabcut_foreground(image_resized, iterations=5)
            st.image(gc_result, width='stretch')
            full_mask = Image.fromarray(np.array(gc_result)[:, :, 3])
        except Exception as exc:
            st.warning(f"GrabCut failed ({exc}) — using full image.")
            full_mask = Image.new("L", image_resized.size, 255)

with col_info:
    st.markdown("**Garment Region Info**")
    coverage = mask_coverage(full_mask)
    st.metric("Foreground coverage", f"{coverage:.1%}")

    # Infer wear type from mask geometry
    detected_wear_type = classify_wear_type_from_mask(full_mask)
    st.markdown(WEAR_BADGE[detected_wear_type], unsafe_allow_html=True)
    wear_type_options = [WearType.TOP, WearType.BOTTOM, WearType.FULL, WearType.UNKNOWN]
    detected_index = wear_type_options.index(detected_wear_type)
    confirmed_wear_type = st.selectbox(
        "Confirm garment type",
        wear_type_options,
        index=detected_index,
        format_func=lambda item: item.value,
        help="The automatic result uses mask geometry. Choose the correct type if the image is ambiguous.",
    )

    if detected_wear_type == WearType.UNKNOWN:
        st.caption(
            "Could not classify garment type from mask geometry. "
            "Try an image where the garment clearly occupies most of the frame."
        )
    elif confirmed_wear_type != detected_wear_type:
        st.caption(
            f"Automatic estimate: {detected_wear_type.value}. "
            "The confirmed type will be used for outfit retrieval."
        )
    else:
        st.caption(
            f"Detected via GrabCut mask analysis: foreground centroid position "
            f"and height coverage ({coverage:.1%}) indicate {confirmed_wear_type.value.lower()}."
        )

st.markdown('</div>', unsafe_allow_html=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Color Analysis
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

st.markdown("### 🎨 Color Analysis")

try:
    with st.spinner("Extracting dominant colors via K-Means…"):
        colors = extract_dominant_colors(image_resized, full_mask, cluster_count)
except ValueError as err:
    st.error(str(err))
    st.stop()

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("#### Dominant Colors")
color_cols = st.columns(len(colors))
for col, color in zip(color_cols, colors):
    with col:
        _render_color_card(color)
st.markdown('</div>', unsafe_allow_html=True)

if show_histogram:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### HSV Hue Histogram")
    st.caption("36 hue bins × 10° each. Chromatic pixels only (saturation > 0.15). Bars colored by actual hue.")
    hist = compute_hsv_histogram(image_resized, full_mask)
    st.markdown(_hue_histogram_html(hist.bins), unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.metric("Mean Saturation", f"{hist.saturation_mean:.2f}")
    m2.metric("Mean Brightness (Value)", f"{hist.value_mean:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Color Harmony Recommendations
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

st.markdown("### 🌈 Color Harmony Suggestions")
st.caption("Derived from the dominant color using complementary, analogous, and triadic color wheel theory.")

recs = recommend_colors(colors)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
rec_cols = st.columns(5)
for col, rec in zip(rec_cols, recs):
    with col:
        _render_rec_card(rec)
st.markdown('</div>', unsafe_allow_html=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Compatible Outfit Retrieval
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if show_retrieval:
    st.markdown("### 🔍 Compatible Outfit Retrieval")
    st.caption(
        "Nearest-neighbor search over DeepFashion2 HSV histograms using Bhattacharyya distance. "
        "Index is cached on disk after the first build — subsequent runs are instant."
    )

    dataset_root = DEFAULT_DATASET_ROOT
    if not dataset_root.is_dir():
        st.info(
            f"DeepFashion2 dataset not found at `{dataset_root}`. "
            "Update `DEFAULT_DATASET_ROOT` in `modules/segmentation.py` to enable retrieval."
        )
    else:
        try:
            image_paths = list_training_images(dataset_root, limit=retrieval_limit)
        except FileNotFoundError as err:
            st.error(str(err))
            st.stop()

        # Check if cache already exists
        from modules.retrieval import _cache_path
        cache_file = _cache_path(dataset_root, retrieval_limit)
        cache_exists = cache_file.exists()

        if not cache_exists:
            st.info(
                f"First run: building color index for {retrieval_limit} images. "
                "This takes ~1–2 min. Subsequent runs load instantly from cache."
            )

        progress_placeholder = st.empty()
        with st.spinner("Loading color index…" if cache_exists else "Building color index (first run)…"):
            bar = progress_placeholder.progress(0) if not cache_exists else None

            def _prog(i: int, total: int, name: str) -> None:
                if bar is not None:
                    bar.progress(i / total, text=f"Indexing {i}/{total}: {name}")

            index = load_or_build_index(
                dataset_root, image_paths,
                progress_callback=_prog,
            )

        if bar is not None:
            progress_placeholder.empty()

        st.success(f"✅ Index ready — {len(index)} garment records from {retrieval_limit} images.")

        results = retrieve_compatible(colors, confirmed_wear_type, index, top_k=8)

        if not results:
            st.info("No compatible results found in the index.")
        else:
            st.markdown(
                f"<p style='color:var(--text-muted);font-size:0.85rem;'>"
                f"Showing top {len(results)} compatible garments (sorted by color similarity)</p>",
                unsafe_allow_html=True,
            )
            gallery_cols = st.columns(4)
            for gi, (img_path, dist, wt, cat_id) in enumerate(results):
                with gallery_cols[gi % 4]:
                    st.markdown('<div class="gallery-item">', unsafe_allow_html=True)
                    try:
                        thumb = Image.open(img_path).convert("RGB")
                        thumb = resize_for_analysis(thumb, max_side=300)
                        st.image(thumb, width='stretch')
                        cat_name = CATEGORY_MAP.get(cat_id, ("Unknown",))[0]
                        similarity = round(1 / (1 + dist) * 100, 1)
                        st.markdown(
                            f"<p style='padding:6px 8px 8px;margin:0;"
                            f"font-size:0.78rem;color:var(--text-muted);'>"
                            f"{cat_name}<br>"
                            f"<span style='color:#10b981;font-weight:600;'>"
                            f"Similarity: {similarity}%</span></p>",
                            unsafe_allow_html=True,
                        )
                    except Exception:
                        st.caption("⚠️ Could not load image")
                    st.markdown('</div>', unsafe_allow_html=True)
