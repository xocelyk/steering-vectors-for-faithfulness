#!/usr/bin/env python3
"""Figures for understanding the steering-vector transfer results.
Reads figures/agg.json, writes PNGs to the paper's artifacts/ folder.
"""
import json, collections
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mpl_config
import figstyle as S

TEAL, RED, GOLD, INK, GRAY, EDGE = S.TEAL, S.RED, S.GOLD, S.INK, S.GRAY, S.EDGE
CMAP, DCMAP = S.CMAP_SEQ, S.CMAP_DIV
# --- recovery shim: pre-rename color aliases used by fig1/fig4/fig5 (not in paper).
# The rename edits (GREEN->TEAL, BLUE->INK, CLAY->RED) were lost in transcript recovery.
GREEN, BLUE, CLAY = TEAL, INK, RED

HERE = Path(__file__).resolve().parent
from paths import artifacts_dir
OUT = artifacts_dir()
OUT.mkdir(parents=True, exist_ok=True)
rows = json.load(open(HERE / "agg.json"))
A = "5.0"

def get(method=None, model=None, scenario=None, ed=None, ec=None):
    return [r for r in rows if r["alpha"] == A
            and (not method or r["method"] == method)
            and (not model or r["model"] == model)
            and (not scenario or r["scenario"] == scenario)
            and (not ed or r["eval_dataset"] == ed)
            and (not ec or r["eval_cue"] == ec)]

MODELS = ["gemma-3-4b-it", "qwen3.5-9b", "gemma-3-12b-it"]  # sorted by size: 4B, 9B, 12B
MLAB = {"gemma-3-4b-it": "Gemma-3 4B", "gemma-3-12b-it": "Gemma-3 12B", "qwen3.5-9b": "Qwen-3.5 9B"}
CUES = ["stanford", "xml", "grader", "insider"]
CLAB = {"stanford": "Stanford", "xml": "XML", "grader": "Grader", "insider": "Unethical"}
DS = ["bbh", "gpqa", "mmlu"]
DLAB = {"bbh": "BBH", "gpqa": "GPQA", "mmlu": "MMLU"}
METHODS = ["contrastive", "synthetic", "opt-specific", "opt-generic"]
METHLAB = {"contrastive": "Contrastive", "synthetic": "Synthetic",
           "opt-specific": "Opt.\nspecific", "opt-generic": "Opt.\ngeneric"}
mean = lambda x: sum(x) / len(x) if x else float("nan")

def matched_cells(method, model):
    cells = []
    for cue in CUES:
        cells += get(method, model, f"gpqa_{cue}", "gpqa", cue)
    return cells

Z = 1.645  # 90% two-sided normal quantile

def ci_prop(p, n):
    """90% Wald half-width for a proportion."""
    if p is None or not n:
        return 0.0
    return Z * (p * (1 - p) / n) ** 0.5

def ci_net(b, c, n):
    """90% half-width for paired net change = (b-c)/n (b=converted, c=regressed)."""
    if not n:
        return 0.0
    var = ((b + c) / n - ((b - c) / n) ** 2) / n
    return Z * max(var, 0.0) ** 0.5

def cell_conv(c):
    return c["conversion"], ci_prop(c["conversion"], c["n_unfaith_base"])

def cell_net(c):
    return c["steer_faith"] - c["base_faith"], ci_net(c["n_converted"], c["n_regressed"], c["n"])

def pooled_matched(method, model):
    """Pool raw counts across the 4 matched-cue GPQA cells for one (method, model)."""
    cells = matched_cells(method, model)
    nconv = sum(c["n_converted"] for c in cells)
    nunf = sum(c["n_unfaith_base"] for c in cells)
    nregr = sum(c["n_regressed"] for c in cells)
    nfaith = sum(c["n_faith_base"] for c in cells)
    n = sum(c["n"] for c in cells)
    conv = nconv / nunf if nunf else float("nan")
    regr = nregr / nfaith if nfaith else float("nan")
    return dict(conv=conv, conv_ci=ci_prop(conv, nunf),
                regr=regr, regr_ci=ci_prop(regr, nfaith),
                net=(nconv - nregr) / n, net_ci=ci_net(nconv, nregr, n))

# --- recovery shim: pre-rename helper used by fig1/fig4 (not in paper).
# Pools matched GPQA cells and returns the requested rate, equivalent to the
# pooled_matched()-based code the lost rename edits would have produced.
def agg_matched(method, model, key):
    pm = pooled_matched(method, model)
    if key == "conversion": return pm["conv"]
    if key == "regression": return pm["regr"]
    cells = matched_cells(method, model)
    n = sum(c["n"] for c in cells)
    if n and key in ("steer_faith", "base_faith"):
        return sum(c[key] * c["n"] for c in cells) / n
    return float("nan")

# ============================================================ FIG 1
# Conversion vs regression vs net, per model, faceted by method.
def fig1():
    fig, axes = mpl_config.figure(1, 4, width=11, sharey=True)
    x = np.arange(len(MODELS))
    w = 0.38
    for ax, method in zip(axes, METHODS):
        conv = [agg_matched(method, m, "conversion") for m in MODELS]
        regr = [agg_matched(method, m, "regression") for m in MODELS]
        net = [agg_matched(method, m, "steer_faith") - agg_matched(method, m, "base_faith") for m in MODELS]
        ax.bar(x - w/2, conv, w, label="Converted\n(unfaith$\\to$faith)", color=GREEN, edgecolor="#444", zorder=3)
        ax.bar(x + w/2, regr, w, label="Regressed\n(faith$\\to$unfaith)", color=RED, edgecolor="#444", zorder=3)
        # net markers
        ax.plot(x, net, "D", color=BLUE, markeredgecolor="#444", markersize=7, zorder=5, label="$\\Delta_{\\mathrm{ack}}$")
        ax.axhline(0, color="#444", lw=0.9, zorder=2)
        ax.set_title(method.replace("-", " ").title().replace("Opt ", "Opt-"), fontsize=11.5)
        ax.set_xticks(x)
        ax.set_xticklabels([MLAB[m].replace(" ", "\n") for m in MODELS], fontsize=9)
        ax.set_ylim(-0.1, 0.62)
    axes[0].set_ylabel("Rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.13))
    # figure title removed; caption is in the LaTeX \caption
    mpl_config.save(fig, str(OUT / "fig1_conversion_vs_regression"), png=True)
    plt.close(fig)

# ============================================================ FIG 2
# Cross-cue transfer heatmaps (contrastive), 3 models side by side.
def _style_heat_axes(ax, rowlab, collab):
    """Square tiles, white separators, no ticks/spines/frame."""
    ax.set_xticks(range(len(collab)))
    ax.set_xticklabels(collab, fontsize=11)
    ax.set_yticks(range(len(rowlab))); ax.set_yticklabels(rowlab, fontsize=11)
    ax.set_xticks(np.arange(-.5, len(collab), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rowlab), 1), minor=True)
    # thin white gutters between tiles (solid; global theme grid is dashed)
    ax.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=1.0)
    ax.set_axisbelow(False)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

def heatmap(ax, M, rowlab, collab, title, CI=None, fmt="{:.2f}", vmin=0, vmax=None):
    vmax = vmax or np.nanmax(M)
    im = ax.imshow(M, cmap=CMAP, vmin=vmin, vmax=vmax, aspect="equal")
    _style_heat_axes(ax, rowlab, collab)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v): continue
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    color=S.HEAT_TEXT, fontsize=12, fontweight="normal")
    ax.set_title(title, fontsize=11.5, pad=8)
    return im

def fig2():
    fig, axes = mpl_config.figure(1, 3, width=11.5, height=4.4)
    im = None
    for ax, model in zip(axes, MODELS):
        M = np.full((4, 4), np.nan)
        for i, tc in enumerate(CUES):
            for j, ec in enumerate(CUES):
                c = get("contrastive", model, f"gpqa_{tc}", "gpqa", ec)
                if c and c[0]["conversion"] is not None:
                    M[i, j] = c[0]["conversion"]
        im = heatmap(ax, M, [CLAB[c] for c in CUES], [CLAB[c] for c in CUES],
                     MLAB[model], vmin=0, vmax=0.6)
    axes[0].set_ylabel("Train cue")
    for ax in axes: ax.set_xlabel("Eval cue")
    mpl_config.save(fig, str(OUT / "fig2_crosscue_heatmaps"), png=True)
    plt.close(fig)

# Diverging colormap for signed deltas: red (worse) -> off-white (0) -> teal (better).
# Use the shared, saturation-capped map from figstyle so the delta heatmaps match
# the sequential ones (don't rebuild a local uncapped map here).
DCMAP = S.CMAP_DIV

def heatmap_div(ax, M, rowlab, collab, title, V, CI=None, fmt="{:+.2f}", aspect="equal",
                cellfs=10.5):
    # "+0"/"-0"-style cells collapse to unsigned zero so rounding noise
    # doesn't read as a signed effect.
    im = ax.imshow(M, cmap=DCMAP, vmin=-V, vmax=V, aspect=aspect)
    _style_heat_axes(ax, rowlab, collab)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v): continue
            txt = fmt.format(v)
            if txt in ("+0", "-0", "+0.00", "-0.00"):
                txt = txt.lstrip("+-")
            ax.text(j, i, txt, ha="center", va="center", color=S.HEAT_TEXT,
                    fontsize=cellfs, fontweight="normal")
    ax.set_title(title, fontsize=11.5, pad=8)
    return im

def fig2_delta():
    # Build matrices first to set a shared symmetric scale across models.
    mats = {}; cis = {}
    for model in MODELS:
        M = np.full((4, 4), np.nan); C = np.full((4, 4), np.nan)
        for i, tc in enumerate(CUES):
            for j, ec in enumerate(CUES):
                c = get("contrastive", model, f"gpqa_{tc}", "gpqa", ec)
                if c:
                    M[i, j], C[i, j] = cell_net(c[0])
        mats[model] = M; cis[model] = C
    V = np.nanmax([np.nanmax(np.abs(m)) for m in mats.values()])
    V = np.ceil(V * 20) / 20  # round up to nearest 0.05
    # Wide, short strip for the main body: square tiles (aspect equal leaves a
    # little gap between the three panels), no colorbar (every cell prints its value).
    fig, axes = mpl_config.figure(1, 3, width=11.5, height=3.2)
    for ax, model in zip(axes, MODELS):
        heatmap_div(ax, mats[model], [CLAB[c] for c in CUES],
                    [CLAB[c] for c in CUES], MLAB[model], V, CI=cis[model], aspect="equal")
    axes[0].set_ylabel("Train cue")
    for ax in axes: ax.set_xlabel("Eval cue")
    mpl_config.save(fig, str(OUT / "fig2b_crosscue_delta"), png=True, pdf=True)
    plt.close(fig)

# ============================================================ FIG 3
# Cross-dataset transfer heatmaps (contrastive, Stanford cue), 3 models.
def fig3():
    scen = {"bbh": "stanford_bbh", "gpqa": "gpqa_stanford", "mmlu": "stanford_mmlu"}
    fig, axes = mpl_config.figure(1, 3, width=11.5)
    im = None
    for ax, model in zip(axes, MODELS):
        M = np.full((3, 3), np.nan)
        for i, td in enumerate(DS):
            for j, ed in enumerate(DS):
                c = get("contrastive", model, scen[td], ed, "stanford")
                if c and c[0]["conversion"] is not None:
                    M[i, j] = c[0]["conversion"]
        im = heatmap(ax, M, [DLAB[d] for d in DS], [DLAB[d] for d in DS],
                     MLAB[model], vmin=0, vmax=0.85)
    axes[0].set_ylabel("Train dataset")
    for ax in axes: ax.set_xlabel("Eval dataset")
    mpl_config.save(fig, str(OUT / "fig3_crossdataset_heatmaps"), png=True)
    plt.close(fig)

def fig3_delta():
    scen = {"bbh": "stanford_bbh", "gpqa": "gpqa_stanford", "mmlu": "stanford_mmlu"}
    mats = {}; cis = {}
    for model in MODELS:
        M = np.full((3, 3), np.nan); C = np.full((3, 3), np.nan)
        for i, td in enumerate(DS):
            for j, ed in enumerate(DS):
                c = get("contrastive", model, scen[td], ed, "stanford")
                if c:
                    M[i, j], C[i, j] = cell_net(c[0])
        mats[model] = M; cis[model] = C
    V = np.nanmax([np.nanmax(np.abs(m)) for m in mats.values()])
    V = np.ceil(V * 20) / 20
    # Match the main-body cross-cue heatmap: square tiles, flat height, no colorbar.
    fig, axes = mpl_config.figure(1, 3, width=11.5, height=3.2)
    im = None
    for ax, model in zip(axes, MODELS):
        im = heatmap_div(ax, mats[model], [DLAB[d] for d in DS],
                         [DLAB[d] for d in DS], MLAB[model], V, CI=cis[model])
    axes[0].set_ylabel("Train dataset")
    for ax in axes: ax.set_xlabel("Eval dataset")
    mpl_config.save(fig, str(OUT / "fig3b_crossdataset_delta"), png=True, pdf=True)
    plt.close(fig)

# ============================================================ FIG 4
# Method comparison: matched conversion by method, grouped by model.
def fig4():
    fig, ax = mpl_config.figure(width=7)
    x = np.arange(len(METHODS))
    w = 0.25
    colors = S.MODEL_COLORS
    for k, model in enumerate(MODELS):
        vals = [agg_matched(m, model, "conversion") for m in METHODS]
        ax.bar(x + (k - 1) * w, vals, w, label=MLAB[model], color=colors[model],
               edgecolor="#444", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([METHLAB[m] for m in METHODS], fontsize=10.5)
    ax.set_ylabel("Matched-setting conversion rate")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=3, fontsize=9)
    ax.set_ylim(0, 0.5)
    # figure title removed; caption is in the LaTeX \caption
    mpl_config.save(fig, str(OUT / "fig4_method_comparison"), png=True)
    plt.close(fig)

def pooled_panel(ds, method, model):
    """Matched cells for one dataset panel: GPQA pools the 4 cues; BBH/MMLU are the
    matched Stanford-cue scenario (cues were varied only on GPQA)."""
    if ds == "gpqa":
        cells = matched_cells(method, model)
    else:
        cells = get(method, model, f"stanford_{ds}", ds, "stanford")
    nconv = sum(c["n_converted"] for c in cells)
    nregr = sum(c["n_regressed"] for c in cells)
    n = sum(c["n"] for c in cells)
    return dict(net=(nconv - nregr) / n if n else float("nan"),
                net_ci=ci_net(nconv, nregr, n))

def fig4_delta():
    # Main-text figure: matched GPQA only, pooled over the four cues (the only
    # dataset where all four cues were run). Wide, short aspect ratio: full
    # \columnwidth in the paper but reduced height to save vertical page space.
    fig, ax = mpl_config.figure(width=7, height=2.5)
    x = np.arange(len(METHODS))
    w = 0.25
    colors = S.MODEL_COLORS
    for k, model in enumerate(MODELS):
        pm = [pooled_matched(m, model) for m in METHODS]
        vals = [p["net"] for p in pm]; errs = [p["net_ci"] for p in pm]
        ax.bar(x + (k - 1) * w, vals, w, yerr=errs, label=MLAB[model], color=colors[model],
               edgecolor="#444", zorder=3, error_kw=dict(elinewidth=1.1, ecolor="#444"))
    ax.axhline(0, color="#444", lw=0.9, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([METHLAB[m] for m in METHODS], fontsize=10.5)
    # Metric is cue acknowledgment: the underlying `faithfulness_score` field is
    # scored as "1 if the response acknowledges the cue" (transfer judge rubric),
    # so label it to match the caption/section ("cue acknowledgment").
    ax.set_ylabel("$\\Delta_{\\mathrm{ack}}$")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=3, fontsize=9)
    ax.set_ylim(-0.07, 0.12)
    # figure title removed; caption is in the LaTeX \caption
    mpl_config.save(fig, str(OUT / "fig4b_method_comparison_delta"), png=True, pdf=True)
    plt.close(fig)

def fig4_delta_datasets():
    # Appendix companion (fig:method-datasets): matched BBH and MMLU, Stanford cue
    # (the only cue varied on those datasets), shared y so the MMLU > BBH ordering
    # on Gemma-3 12B is read directly off the bars.
    PANELS = [("bbh", "BBH (Stanford cue)"), ("mmlu", "MMLU (Stanford cue)")]
    fig, axes = mpl_config.figure(1, len(PANELS), width=8, height=2.6, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    x = np.arange(len(METHODS))
    w = 0.25
    colors = S.MODEL_COLORS
    for ax, (ds, title) in zip(axes, PANELS):
        for k, model in enumerate(MODELS):
            pm = [pooled_panel(ds, m, model) for m in METHODS]
            vals = [p["net"] for p in pm]; errs = [p["net_ci"] for p in pm]
            ax.bar(x + (k - 1) * w, vals, w, yerr=errs, label=MLAB[model], color=colors[model],
                   edgecolor="#444", zorder=3, error_kw=dict(elinewidth=1.1, ecolor="#444"))
        ax.axhline(0, color="#444", lw=0.9, zorder=2)
        ax.set_xticks(x); ax.set_xticklabels([METHLAB[m] for m in METHODS], fontsize=9)
        ax.set_title(title, fontsize=11.5)
    axes[0].set_ylabel("$\\Delta_{\\mathrm{ack}}$")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3, fontsize=9)
    axes[0].set_ylim(-0.09, 0.36)
    mpl_config.save(fig, str(OUT / "fig4c_method_comparison_bbh_mmlu"), png=True, pdf=True)
    plt.close(fig)

METHOD_COLORS = {"contrastive": TEAL, "synthetic": GOLD,
                 "opt-specific": RED, "opt-generic": GRAY}
METHLEG = {"contrastive": "Contrastive", "synthetic": "Synthetic",
           "opt-specific": "Opt. specific", "opt-generic": "Opt. generic"}

def fig4_delta_model_dataset_method():
    # Main-text figure (fig:net): dot-and-whisker of the matched effect,
    # one panel per model, evaluation dataset on the x-axis, one color per
    # construction method (subsumes the former fig4b/fig4c method bars).
    # Stanford cue throughout -- the one cue run on all three datasets -- so
    # the panels are like-for-like; the GPQA cue breakdown is fig4e.
    SCEN = {"gpqa": "gpqa_stanford", "bbh": "stanford_bbh", "mmlu": "stanford_mmlu"}
    PANEL_DS = ["gpqa", "bbh", "mmlu"]
    fig, axes = mpl_config.figure(1, 3, width=11, height=3.4, sharey=True)
    x = np.arange(len(PANEL_DS))
    for ax, model in zip(axes, MODELS):
        for k, method in enumerate(METHODS):
            nets, cis = zip(*[cell_net(get(method, model, SCEN[ds], ds, "stanford")[0])
                              for ds in PANEL_DS])
            ax.errorbar(x + (k - 1.5) * 0.16, nets, yerr=cis,
                        fmt="o", linestyle="none", zorder=3,
                        label=METHLEG[method] if model == MODELS[0] else None,
                        color=METHOD_COLORS[method], markeredgecolor="#444",
                        markersize=7, ecolor="#444", elinewidth=1.1, capsize=2.5)
        ax.axhline(0, color="#444", lw=0.9, zorder=2)
        ax.set_xticks(x); ax.set_xticklabels([DLAB[d] for d in PANEL_DS], fontsize=13.5)
        ax.set_title(MLAB[model], fontsize=15)
        ax.set_xlim(-0.5, len(PANEL_DS) - 0.5)
        ax.tick_params(axis="y", labelsize=13)
    axes[0].set_ylabel("$\\Delta_{\\mathrm{ack}}$", fontsize=14)
    axes[0].set_ylim(-0.13, 0.36)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center",
               ncol=4, fontsize=12.5, title="Construction method", title_fontsize=12.5)
    mpl_config.save(fig, str(OUT / "fig4d_delta_model_dataset_method"), png=True, pdf=True)
    plt.close(fig)

def fig4_gpqa_cue_breakdown():
    # Appendix companion to fig:net (fig:gpqa-cues): GPQA per-cue effects plus
    # the pooled-over-cues estimate, same panel/color scheme as fig4d. Shows
    # why fig:net's GPQA (Stanford) cells are noisier than the pooled numbers
    # quoted in the text: per-cue n is ~140 vs ~550 pooled, and on the smaller
    # models the per-cue effects are mixed in sign and cancel when pooled.
    XPOS = CUES + ["pooled"]
    XLAB = [CLAB[c] for c in CUES] + ["Pooled"]
    fig, axes = mpl_config.figure(1, 3, width=11, height=2.8, sharey=True)
    x = np.arange(len(XPOS))
    for ax, model in zip(axes, MODELS):
        for k, method in enumerate(METHODS):
            vals = [cell_net(get(method, model, f"gpqa_{c}", "gpqa", c)[0]) for c in CUES]
            pm = pooled_matched(method, model)
            vals.append((pm["net"], pm["net_ci"]))
            nets, cis = zip(*vals)
            ax.errorbar(x + (k - 1.5) * 0.16, nets, yerr=cis,
                        fmt="o", linestyle="none", zorder=3,
                        label=METHLEG[method] if model == MODELS[0] else None,
                        color=METHOD_COLORS[method], markeredgecolor="#444",
                        markersize=6, ecolor="#444", elinewidth=1.1, capsize=2.5)
        ax.axhline(0, color="#444", lw=0.9, zorder=2)
        ax.axvline(len(CUES) - 0.5, color="#bbb", lw=0.8, ls=":", zorder=1)
        ax.set_xticks(x); ax.set_xticklabels(XLAB, fontsize=11)
        ax.set_title(MLAB[model], fontsize=13)
        ax.set_xlim(-0.5, len(XPOS) - 0.5)
        ax.tick_params(axis="y", labelsize=11.5)
    axes[0].set_ylabel("$\\Delta_{\\mathrm{ack}}$", fontsize=12.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center",
               ncol=4, fontsize=11, title="Construction method", title_fontsize=11)
    mpl_config.save(fig, str(OUT / "fig4e_gpqa_cue_breakdown"), png=True, pdf=True)
    plt.close(fig)

# ============================================================ FIG 5
# Unified vs specific (contrastive, Gemma-12B). Cue panel + dataset panel.
def fig5():
    fig, (axc, axd) = mpl_config.figure(1, 2, width=9)
    model = "gemma-3-12b-it"
    # cues
    x = np.arange(len(CUES)); w = 0.38
    spec = [get("contrastive", model, f"gpqa_{c}", "gpqa", c)[0]["conversion"] for c in CUES]
    unif = [get("contrastive", model, "gpqa_all", "gpqa", c)[0]["conversion"] for c in CUES]
    axc.bar(x - w/2, spec, w, label="Cue-specific", color=BLUE, edgecolor="#444", zorder=3)
    axc.bar(x + w/2, unif, w, label="Unified (all cues)", color=CLAY, edgecolor="#444", zorder=3)
    axc.set_xticks(x); axc.set_xticklabels([CLAB[c] for c in CUES], fontsize=9)
    axc.set_ylabel("Conversion rate"); axc.set_title("Across cues (GPQA)", fontsize=11.5)
    axc.legend(fontsize=9, loc="best"); axc.set_ylim(0, 0.65)
    # datasets
    scen = {"bbh": "stanford_bbh", "gpqa": "gpqa_stanford", "mmlu": "stanford_mmlu"}
    xd = np.arange(len(DS))
    specd = [get("contrastive", model, scen[d], d, "stanford")[0]["conversion"] for d in DS]
    unifd = [get("contrastive", model, "stanford_all", d, "stanford")[0]["conversion"] for d in DS]
    axd.bar(xd - w/2, specd, w, label="Dataset-specific", color=BLUE, edgecolor="#444", zorder=3)
    axd.bar(xd + w/2, unifd, w, label="Unified (all datasets)", color=CLAY, edgecolor="#444", zorder=3)
    axd.set_xticks(xd); axd.set_xticklabels([DLAB[d] for d in DS], fontsize=9)
    axd.set_title("Across datasets (Stanford cue)", fontsize=11.5)
    axd.legend(fontsize=9, loc="best"); axd.set_ylim(0, 0.9)
    # figure title removed; caption is in the LaTeX \caption
    mpl_config.save(fig, str(OUT / "fig5_unified_vs_specific"), png=True)
    plt.close(fig)

def fig5_delta():
    fig, (axc, axd) = mpl_config.figure(1, 2, width=9)
    model = "gemma-3-12b-it"
    ekw = dict(elinewidth=1.1, ecolor="#444")
    # cues
    x = np.arange(len(CUES)); w = 0.38
    spec, spec_ci = zip(*[cell_net(get("contrastive", model, f"gpqa_{c}", "gpqa", c)[0]) for c in CUES])
    unif, unif_ci = zip(*[cell_net(get("contrastive", model, "gpqa_all", "gpqa", c)[0]) for c in CUES])
    axc.bar(x - w/2, spec, w, yerr=spec_ci, label="Cue-specific", color=TEAL, edgecolor="#444", zorder=3, error_kw=ekw)
    axc.bar(x + w/2, unif, w, yerr=unif_ci, label="Unified (all cues)", color=RED, edgecolor="#444", zorder=3, error_kw=ekw)
    axc.axhline(0, color="#444", lw=0.9, zorder=2)
    axc.set_xticks(x); axc.set_xticklabels([CLAB[c] for c in CUES], fontsize=9)
    axc.set_ylabel("$\\Delta_{\\mathrm{ack}}$"); axc.set_title("Across cues (GPQA)", fontsize=11.5)
    axc.legend(fontsize=9, loc="best"); axc.set_ylim(-0.10, 0.32)
    # datasets
    scen = {"bbh": "stanford_bbh", "gpqa": "gpqa_stanford", "mmlu": "stanford_mmlu"}
    xd = np.arange(len(DS))
    specd, specd_ci = zip(*[cell_net(get("contrastive", model, scen[d], d, "stanford")[0]) for d in DS])
    unifd, unifd_ci = zip(*[cell_net(get("contrastive", model, "stanford_all", d, "stanford")[0]) for d in DS])
    axd.bar(xd - w/2, specd, w, yerr=specd_ci, label="Dataset-specific", color=TEAL, edgecolor="#444", zorder=3, error_kw=ekw)
    axd.bar(xd + w/2, unifd, w, yerr=unifd_ci, label="Unified (all datasets)", color=RED, edgecolor="#444", zorder=3, error_kw=ekw)
    axd.axhline(0, color="#444", lw=0.9, zorder=2)
    axd.set_xticks(xd); axd.set_xticklabels([DLAB[d] for d in DS], fontsize=9)
    axd.set_title("Across datasets (Stanford cue)", fontsize=11.5)
    axd.legend(fontsize=9, loc="best"); axd.set_ylim(-0.10, 0.45)
    # figure title removed; caption is in the LaTeX \caption
    mpl_config.save(fig, str(OUT / "fig5b_unified_vs_specific_delta"), png=True)
    plt.close(fig)

# Only the figures the paper includes are regenerated; the others (fig1, fig2,
# fig3, fig4, fig4b, fig4c, fig5, fig5b) are retired from the paper and their
# functions are kept for reference but not run.
for f in (fig2_delta, fig3_delta, fig4_delta_model_dataset_method, fig4_gpqa_cue_breakdown):
    f()
    print(f"done {f.__name__}")
print(f"\nFigures written to {OUT}")
