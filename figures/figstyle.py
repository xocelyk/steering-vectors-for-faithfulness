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
import mpl_config
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_hex

mpl_config.use("default")
# Pure-white background everywhere (figure, axes, saved PNG). The default theme
# uses a warm off-white (#F5F4F2); we override via a derived theme so that
# mpl_config.figure()/apply()/save() all pick up white.
mpl_config.use(mpl_config.get_theme("default").derive("default_white", bg="#FFFFFF"))

# --- core colors ---
# NOTE: var names are historical/non-literal. TEAL = bright sky-BLUE, GOLD = bright
# YELLOW, RED = vermillion. TEAL/RED double as the heatmap valence poles
# (blue=improved, red=worsened). In the categorical model/dataset legends the colors
# are assigned in list order (blue, yellow, red), so e.g. the LARGEST model is red.
TEAL = "#43A1E4"      # bright sky-blue  (heatmap "improved" pole; 1st categorical / smallest model)
RED = "#E36450"       # dusty red: slightly lifted value, lower saturation (muted), hue kept red (heatmap "worsened" pole; last categorical / largest model)
GOLD = "#F7D02E"      # bright yellow     (middle categorical; truer-to-yellow, not amber/gold)
INK = "#1F6FB0"       # dark blue: net markers, emphasis (harmonizes with TEAL)
GRAY = "#8C8C8C"      # neutral fourth category
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
# sequential (all-positive magnitudes): near-white -> bright blue (top end slightly desaturated
# so black cell text keeps contrast > 4.5)
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#F2F7FC", "#C2DEF1", "#88BEE6", _cap_sat("#4A93C9", HEAT_SAT_CAP)])
# diverging (signed deltas): RED -> near-white -> TEAL (poles track the variables above)
CMAP_DIV = LinearSegmentedColormap.from_list("div_blue_red", [_cap_sat(RED, HEAT_SAT_CAP), "#FAF7F2", _cap_sat(TEAL, HEAT_SAT_CAP)])
