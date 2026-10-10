"""Dashboard visual theme (FR-RPT-004): the validated default palette from the
`dataviz` design-system skill, as plain Python constants. Both `app/charts.py`
(Plotly) and `app/main.py` (custom CSS for stat tiles/cards) read from here, so
there is exactly one place a color means something.

The categorical slots are a **fixed order** -- a variant's color never changes
when a filter changes which variants are visible (anti-pattern: "recolor on
filter"). `VARIANT_ORDER` is the canonical assignment; `variant_color()` looks a
variant up by name, falling back to a stable hash-based slot for names outside the
expected V0-V5 set (e.g. a user's own variant names via `fsa ingest`).
"""

from __future__ import annotations

# Categorical palette, light mode (8 slots, fixed order -- dataviz skill palette.md).
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Canonical variant -> slot assignment (synthetic study default). A variant name
# outside this list gets a stable slot via `variant_color()`'s hash fallback.
VARIANT_ORDER = ["V0", "V1", "V2", "V3", "V4", "V5"]

# Sequential ramp (one hue, light -> dark), for magnitude (heatmaps).
SEQUENTIAL_BLUE = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]

# Status palette (fixed; never reused for a categorical series).
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"

# Chrome & ink (light mode -- Streamlit's default dashboard background is light).
SURFACE = "#fcfcfb"
PAGE_PLANE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BORDER = "rgba(11,11,11,0.10)"


def variant_color(name: str) -> str:
    """This variant's fixed categorical color -- the same color everywhere,
    regardless of which other variants are currently selected/visible."""
    if name in VARIANT_ORDER:
        return CATEGORICAL[VARIANT_ORDER.index(name) % len(CATEGORICAL)]
    return CATEGORICAL[hash(name) % len(CATEGORICAL)]


def dashboard_css() -> str:
    """Custom CSS for stat tiles and recommendation cards, injected once via
    `st.markdown(..., unsafe_allow_html=True)`. Streamlit's own chrome (buttons,
    inputs, tabs) is left at its native styling -- only the custom components below
    need it."""
    return f"""
    <style>
    .av-kpi-row {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }}
    .av-kpi-tile {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
        padding: 0.9rem 1.1rem; min-width: 150px; flex: 1;
    }}
    .av-kpi-label {{
        color: {INK_SECONDARY}; font-size: 0.8rem; margin-bottom: 0.3rem;
    }}
    .av-kpi-value {{
        color: {INK_PRIMARY}; font-size: 1.6rem; font-weight: 600; line-height: 1.1;
    }}
    .av-badge {{
        display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600; margin-right: 0.4rem;
    }}
    .av-badge-good {{ background: rgba(12,163,12,0.12); color: {STATUS_GOOD}; }}
    .av-badge-warning {{ background: rgba(250,178,25,0.18); color: #8a5a00; }}
    .av-rec-card {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-left: 4px solid {STATUS_GOOD};
        border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.7rem;
    }}
    .av-rec-card.hypothesis {{ border-left-color: {STATUS_WARNING}; }}
    .av-rec-caveat {{ color: {INK_MUTED}; font-size: 0.85rem; margin-top: 0.3rem; }}
    </style>
    """
