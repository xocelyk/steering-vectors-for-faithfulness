"""Minimal shim for the `mpl_config` package (the real one lives in the
iCloud-stuck `mpl-themes` repo). Implements only the 4 entry points the figure
scripts use: figure(), save(), use(), get_theme().derive().

Formatting is approximate by design (the user deprioritized exact styling); the
actual palette lives in figstyle.py, so colors are faithful. Aspect ratios and
fonts here are reasonable stand-ins and can be tuned later.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt

_BG = "#FFFFFF"

# Register Google Sans Code (static Regular/Bold/Italic instanced from the
# variable font; matplotlib only picks up the default 400 weight from a varfont,
# so bold panel titles need a real static Bold). Fall back silently if the files
# are absent so the scripts still run on a machine without them.
_FONT_DIR = Path(__file__).resolve().parent / "fonts"
for _f in sorted(_FONT_DIR.glob("GoogleSansCode-*.ttf")):
    try:
        _fm.fontManager.addfont(str(_f))
    except Exception:
        pass


class _Theme:
    def __init__(self, name="default", bg="#FFFFFF"):
        self.name = name
        self.bg = bg

    def derive(self, name, bg=None, **kw):
        return _Theme(name, bg if bg is not None else self.bg)


def get_theme(name="default"):
    return _Theme(name)


def use(theme="default"):
    """Apply a clean, roughly-on-brand rcParam set. Accepts a name or _Theme."""
    global _BG
    if isinstance(theme, _Theme):
        _BG = theme.bg
    plt.rcParams.update({
        "figure.facecolor": _BG,
        "axes.facecolor": _BG,
        "savefig.facecolor": _BG,
        "font.family": "monospace",
        "font.monospace": ["Google Sans Code", "DejaVu Sans Mono"],
        "font.size": 11.5,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "grid.color": "#888888",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "legend.frameon": False,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def figure(nrows=1, ncols=1, width=6.0, height=None, widths=None, heights=None,
           sharex=False, sharey=False, **kw):
    """Wrapper over plt.subplots. width/height in inches; widths/heights map to
    gridspec width_ratios/height_ratios."""
    if height is None:
        # flatter for multi-column rows, taller for single panels
        height = round(width * 0.42 * nrows, 2) if ncols > 1 else round(width * 0.62, 2)
    gridspec_kw = {}
    if widths is not None:
        gridspec_kw["width_ratios"] = widths
    if heights is not None:
        gridspec_kw["height_ratios"] = heights
    # constrained_layout auto-reserves room for suptitles, panel titles,
    # colorbars and inter-panel gaps -> no overlaps / no label collisions.
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height),
                             sharex=sharex, sharey=sharey,
                             gridspec_kw=gridspec_kw or None,
                             layout="constrained")
    fig.patch.set_facecolor(_BG)
    return fig, axes


def save(fig, path, png=False, pdf=False, dpi=150, **kw):
    """Save fig to path + .png / .pdf (path given without extension)."""
    if not (png or pdf):
        png = True
    if png:
        fig.savefig(str(path) + ".png", dpi=dpi, bbox_inches="tight", facecolor=_BG)
    if pdf:
        fig.savefig(str(path) + ".pdf", bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
