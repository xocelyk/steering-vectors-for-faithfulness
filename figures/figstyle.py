#!/usr/bin/env python3
"""Unified figure style for the paper artifacts.

One palette for every figure. TEAL/RED are the two heatmap valence poles
(TEAL = improved/positive, RED = worsened/negative). For categorical legends
(models, datasets) colors are assigned in the group's natural list order as
blue, yellow, red, so every legend reads blue -> yellow -> red.

Import this at the top of each figure script instead of touching mpl_config
colors directly, so the whole figure set stays consistent.
"""
import colorsys
import os
import mpl_config
import numpy as np
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_hex

mpl_config.use("default")
# Pure-white background everywhere (figure, axes, saved PNG). The default theme
# uses a warm off-white (#F5F4F2); we override via a derived theme so that
# mpl_config.figure()/apply()/save() all pick up white.
mpl_config.use(mpl_config.get_theme("default").derive("default_white", bg="#FFFFFF"))

# --- FONT SWITCH: FIG_FONT env, like the FIG_PALETTE switch below ---
#   cmu -- CMU Sans Serif (Computer Modern sans companion; matches the paper's
#          LaTeX look) + mathtext "cm" so $\Delta$/$\alpha$ render as CM math.
# Applied AFTER the mpl_config.use() calls above (no figure script calls use()
# again later, so these rcParams stick for the whole render).
FONT = os.environ.get("FIG_FONT", "cmu")
if FONT == "cmu":
    from pathlib import Path as _Path
    import matplotlib.font_manager as _fm
    import matplotlib.pyplot as _plt
    _CMU_DIRS = [
        _Path(__file__).resolve().parent / "fonts",   # bundled copies (OFL 1.1)
        _Path("/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/cm-unicode"),
    ]
    _cmu_found = []
    for _stem in ("cmunss", "cmunsx", "cmunsi", "cmunso"):   # regular/bold/oblique/bold-oblique
        for _d in _CMU_DIRS:
            _f = _d / f"{_stem}.otf"
            if _f.exists():
                _fm.fontManager.addfont(str(_f))
                _cmu_found.append(_f)
                break
    if not _cmu_found:
        print(f"warning: CMU Sans Serif not found in {[str(d) for d in _CMU_DIRS]}; "
              "figures will fall back to DejaVu Sans", flush=True)
    _plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
        "mathtext.fontset": "cm",
        # CMU Sans Serif has no U+2212 MINUS SIGN; use ASCII hyphen-minus in tick
        # labels instead of falling back to a mismatched DejaVu glyph
        "axes.unicode_minus": False,
    })
else:
    raise SystemExit(f"FIG_FONT={FONT!r} unknown; only 'cmu' is supported")

# --- core colors ---
# NOTE: var names are historical/non-literal. TEAL = the palette's BLUE, GOLD = the
# YELLOW/ORANGE, RED = the red. TEAL/RED double as the heatmap valence poles
# (blue=improved, red=worsened). In the categorical model/dataset legends the colors
# are assigned in list order (blue, yellow, red), so e.g. the LARGEST model is red.
#
# PALETTE SWITCH: set FIG_PALETTE to pick a scheme. "current" (the default) is the
# palette the paper has used so far, byte-identical output. Alternatives are
# colorblind-safe (deutan-checked) and keep the blue->yellow->red legend order and
# blue-positive/red-negative heatmap semantics:
#   current   -- bright sky-blue / bright yellow / dusty red (status quo)
#   okabeito  -- Okabe-Ito: blue #0072B2 / orange #E69F00 / vermillion #D55E00
#   tolbright -- Paul Tol bright: blue #4477AA / yellow #CCBB44 / red #EE6677
#   petroff   -- Petroff (CMS): blue #3F90DA / orange #FFA90E / deep red #BD1F01
#   anthropic -- warm/bookish: slate #52748F / kraft sand #D4A27F / terracotta #C96442
#   paper     -- paper default: blue #3E86C6 / ochre #C49F42 / rust #C84D2F
PALETTE = os.environ.get("FIG_PALETTE", "paper")

_PALETTES = {
    # status quo (unchanged from before the switch existed)
    "current":   dict(TEAL="#43A1E4", GOLD="#F7D02E", RED="#E36450",
                      INK="#1F6FB0", GRAY="#8C8C8C"),
    # Okabe-Ito (Wong 2011): the classic CVD-safe set
    "okabeito":  dict(TEAL="#0072B2", GOLD="#E69F00", RED="#D55E00",
                      INK="#01518C", GRAY="#999999"),
    # Paul Tol "bright" qualitative scheme
    "tolbright": dict(TEAL="#4477AA", GOLD="#CCBB44", RED="#EE6677",
                      INK="#2E5A87", GRAY="#BBBBBB"),
    # Petroff 2021 (matplotlib "petroff10" subset): widest lightness spread
    "petroff":   dict(TEAL="#3F90DA", GOLD="#FFA90E", RED="#BD1F01",
                      INK="#2A6DAF", GRAY="#94A4A2"),
    # Anthropic house style: dusty slate blue / warm kraft sand / terracotta
    # (Claude coral family), warm gray neutral. Deutan-checked (worst adjacent
    # pair dE 20.3); grayscale-legible via lightness spread slate < terracotta < sand.
    "anthropic": dict(TEAL="#52748F", GOLD="#D4A27F", RED="#C96442",
                      INK="#3D586E", GRAY="#A39E93"),
    # Paper default: bright true blue, ochre, rust; the ochre is lightened to
    # pass deutan vs the rust. Deutan-checked (worst adjacent pair dE 16.5
    # ochre-rust; the blue-ochre and blue-rust pairs are wider).
    "paper":     dict(TEAL="#3E86C6", GOLD="#C49F42", RED="#C84D2F",
                      INK="#2A5E8C", GRAY="#A39E93"),
}
if PALETTE not in _PALETTES:
    raise SystemExit(f"FIG_PALETTE={PALETTE!r} unknown; pick one of {sorted(_PALETTES)}")
_P = _PALETTES[PALETTE]
# FIG_TEAL / FIG_THIRD / FIG_RED override single series slots on top of whatever
# FIG_PALETTE selects — used to audition colors while the rest stays fixed, e.g.
# FIG_PALETTE=paper FIG_THIRD="#C49F42". Heatmap poles do NOT track these
# overrides (they are tuned per palette below), so use them for series-color
# auditions only.
TEAL = os.environ.get("FIG_TEAL", _P["TEAL"])    # palette blue (heatmap "improved" pole; 1st categorical / smallest model)
RED = os.environ.get("FIG_RED", _P["RED"])       # palette red (heatmap "worsened" pole; last categorical / largest model)
GOLD = os.environ.get("FIG_THIRD", _P["GOLD"])   # palette yellow/orange (middle categorical)
INK = _P["INK"]       # dark blue: net markers, emphasis (harmonizes with TEAL)
GRAY = _P["GRAY"]     # neutral fourth category
OFFWHITE = "#F5F4F2"
EDGE = "#444"
GRID = "#888"
HEADER_FILL = "#E7E4DF"

# --- 3-category palette: colors assigned DOWN each group's natural list order ---
# Each categorical legend keeps its own semantic order (models by size 4B->9B->12B;
# datasets bbh->gpqa->mmlu) and colors are assigned along that order as
# TEAL(blue) -> GOLD(yellow) -> RED(red). So every such legend reads blue, yellow, red
# WITHOUT re-sorting the legend. (Consequence: the largest model, 12B, is red.)
MODEL_COLORS = {"gemma-3-4b-it": TEAL, "qwen3.5-9b": GOLD, "gemma-3-12b-it": RED}
DATASET_COLORS = {"bbh": TEAL, "gpqa": GOLD, "mmlu": RED}

# --- four-category cue palette (CUES order stanford,xml,grader,insider -> blue,yellow,red,gray) ---
CUE_COLORS = {"stanford": TEAL, "xml": GOLD, "grader": RED, "insider": GRAY}

# --- heatmap support ---
# Black text on EVERY cell. The maps below top out at the medium-luminance
# primaries (not dark), so black stays legible (contrast > 4.5) on every tile.
HEAT_TEXT = "#000000"   # black, on every heatmap cell
TILE = "#FFFFFF"        # separator drawn between tiles

# Colormaps are anchored to the SAME primaries used everywhere else, so the
# heatmap poles are literally our TEAL / RED — not a separate green.

def _cap_sat(hexcolor, frac):
    """Scale a color's HSV saturation by `frac`, keeping hue and brightness.
    Used to soften the most-saturated heatmap poles so the densest cells read
    gentler without shifting hue."""
    h, s, v = colorsys.rgb_to_hsv(*to_rgb(hexcolor))
    return to_hex(colorsys.hsv_to_rgb(h, s * frac, v))

# Heatmap poles capped to 70% of full saturation (densest cells read softer).
HEAT_SAT_CAP = 0.70


def _truncated(name, lo, hi):
    """Slice a named matplotlib colormap to [lo, hi] so its extremes stay light
    enough for black cell text (contrast > 4.5 verified at both poles)."""
    base = matplotlib.colormaps[name]
    return LinearSegmentedColormap.from_list(f"{name}_{lo}_{hi}", base(np.linspace(lo, hi, 256)))


if PALETTE == "current":
    # sequential (all-positive magnitudes): near-white -> bright blue (top end slightly
    # desaturated so black cell text keeps contrast > 4.5)
    CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#F2F7FC", "#C2DEF1", "#88BEE6", _cap_sat("#4A93C9", HEAT_SAT_CAP)])
    # diverging (signed deltas): RED -> near-white -> TEAL (poles track the variables above)
    CMAP_DIV = LinearSegmentedColormap.from_list("div_blue_red", [_cap_sat(RED, HEAT_SAT_CAP), "#FAF7F2", _cap_sat(TEAL, HEAT_SAT_CAP)])
elif PALETTE == "okabeito":
    CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#F2F7FC", "#C4DCEE", "#85B9DE", "#4292C6"])
    # ColorBrewer RdBu, truncated so black text passes at both poles (red low, blue high)
    CMAP_DIV = _truncated("RdBu", 0.18, 0.84)
elif PALETTE == "tolbright":
    CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#F2F6FA", "#C2E4EF", "#98CAE1", "#6EA6CD", "#4A7BB7"])
    # Paul Tol "sunset" diverging scheme, reversed (red low -> blue high), with the two
    # darkest end steps dropped so black cell text passes at both poles, and the
    # cream-yellow midpoint replaced by neutral near-white so near-zero cells stay
    # sign-legible (pale red vs pale blue, not yellow vs green)
    CMAP_DIV = LinearSegmentedColormap.from_list("tol_sunset_rb", [
        "#DD3D2D", "#F67E4B", "#FDB366", "#F7F7F5",
        "#98CAE1", "#6EA6CD", "#4A7BB7"])
elif PALETTE == "petroff":
    CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#F2F7FC", "#C6DEF1", "#8ABFE7", "#3F90DA"])
    # matplotlib coolwarm, reversed (red low, blue high), truncated for black-text contrast
    CMAP_DIV = _truncated("coolwarm_r", 0.10, 0.90)
elif PALETTE == "anthropic":
    # sequential: near-white -> slate blue, top pole lightened vs the series slate so
    # black cell text keeps contrast > 4.5 (#6E8FAF is 6.2:1)
    CMAP_SEQ = LinearSegmentedColormap.from_list("seq_slate", ["#F4F7FA", "#CFDDE8", "#9FBBCF", "#6E8FAF"])
    # diverging: terracotta (Claude coral #D97757, 6.7:1) -> warm near-white ->
    # lightened slate (#6E8FAF, 6.2:1); blue = positive, terracotta reads as the red pole
    CMAP_DIV = LinearSegmentedColormap.from_list("div_slate_terracotta", ["#D97757", "#FAF7F4", "#6E8FAF"])
elif PALETTE == "paper":
    # sequential: near-white -> lightened blue (series blue #3E86C6 is 4.0:1 for
    # black text, so the heatmap pole is lifted to #4E93CE, 6.4:1)
    CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#F3F7FB", "#C9DCEE", "#8CB9E0", "#4E93CE"])
    # diverging: lightened rust (#D3603F, 5.5:1; series rust #C84D2F is 4.56:1) ->
    # warm near-white -> lightened blue (#4E93CE, 6.4:1). Blue = positive pole,
    # rust = the red pole, matching the captions' "blue > 0, red < 0".
    CMAP_DIV = LinearSegmentedColormap.from_list("div_blue_rust", ["#D3603F", "#FAF8F5", "#4E93CE"])
