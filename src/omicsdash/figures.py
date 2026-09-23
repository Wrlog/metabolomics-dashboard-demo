"""Figures for the dashboard.

Every figure is rendered once per theme and returned as PNG bytes, so the
report can embed a light and a dark copy and let CSS pick. Nothing here writes
to disk or depends on the report layout.

House style, applied through `_style`: recessive grid and axes, no chart
junk, a legend whenever more than one series is on screen, and direct labels
on the few marks worth naming rather than a number on every point.
"""

from __future__ import annotations

import io

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from .pipeline import CurationResult, PCAResult  # noqa: E402
from .simulate import OUTCOMES, Study  # noqa: E402
from .theme import Theme  # noqa: E402

DPI = 160
MARKER = 44  # ~7.5px diameter, above the 8px-ish floor once antialiased


def _style(theme: Theme, ax: plt.Axes) -> None:
    ax.set_facecolor(theme.surface)
    ax.figure.set_facecolor(theme.surface)
    ax.grid(True, color=theme.grid, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.border)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=theme.text_secondary, labelsize=9, length=0)
    ax.xaxis.label.set_color(theme.text_secondary)
    ax.yaxis.label.set_color(theme.text_secondary)
    ax.title.set_color(theme.text_primary)


def _legend(theme: Theme, ax: plt.Axes, **kwargs) -> None:
    leg = ax.legend(
        frameon=False, fontsize=9, labelcolor=theme.text_secondary, **kwargs
    )
    for text in leg.get_texts():
        text.set_color(theme.text_secondary)


def _png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def qc_drift(study: Study, theme: Theme) -> bytes:
    """Per-sample median intensity against injection order.

    The pooled QC samples should sit on a flat line; any slope in them is
    instrument drift rather than biology, which is what the curation step
    filters on.
    """
    median = np.log2(study.abundance.median(axis=1))
    order = study.samples["run_order"]
    is_qc = study.samples["sample_type"] == "qc"

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    _style(theme, ax)
    ax.scatter(
        order[~is_qc], median[~is_qc], s=MARKER, c=theme.series[0],
        edgecolors=theme.surface, linewidths=1.0, label="Study sample", zorder=2,
    )
    ax.scatter(
        order[is_qc], median[is_qc], s=MARKER + 18, c=theme.series[1],
        marker="D", edgecolors=theme.surface, linewidths=1.0,
        label="Pooled QC", zorder=3,
    )
    if is_qc.sum() >= 2:
        slope, intercept = np.polyfit(order[is_qc], median[is_qc], 1)
        xs = np.array([order.min(), order.max()])
        ax.plot(xs, slope * xs + intercept, color=theme.series[1],
                linewidth=2, linestyle="--", zorder=4,
                label=f"QC trend ({slope:+.3f} log2/injection)")
    ax.set_xlabel("Injection order")
    ax.set_ylabel("Median log2 intensity")
    _legend(theme, ax, loc="upper left", ncol=1)
    return _png(fig)


def pca_scores(components: PCAResult, curated: CurationResult, theme: Theme) -> bytes:
    """PC1 against PC2, split by acquisition batch."""
    scores = components.scores
    run_order = curated.samples["run_order"]
    batch = np.where(run_order > run_order.median(), "Batch 2", "Batch 1")

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    _style(theme, ax)
    for i, name in enumerate(["Batch 1", "Batch 2"]):
        sel = batch == name
        ax.scatter(
            scores.loc[sel, "PC1"], scores.loc[sel, "PC2"], s=MARKER,
            c=theme.series[i], edgecolors=theme.surface, linewidths=1.0,
            label=name, zorder=2,
        )
    ax.axhline(0, color=theme.border, linewidth=1, zorder=1)
    ax.axvline(0, color=theme.border, linewidth=1, zorder=1)
    ax.set_xlabel(f"PC1 ({components.explained[0] * 100:.1f}% of variance)")
    ax.set_ylabel(f"PC2 ({components.explained[1] * 100:.1f}% of variance)")
    _legend(theme, ax, loc="best")
    return _png(fig)


def scree(components: PCAResult, theme: Theme) -> bytes:
    """Variance explained by each retained component."""
    pct = components.explained * 100
    labels = [f"PC{i + 1}" for i in range(len(pct))]

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    _style(theme, ax)
    ax.grid(False, axis="x")
    bars = ax.bar(labels, pct, color=theme.series[0], width=0.62, zorder=2)
    for bar, value in zip(bars, pct, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + max(pct) * 0.03,
            f"{value:.1f}%", ha="center", va="bottom",
            color=theme.text_secondary, fontsize=9,
        )
    ax.set_ylabel("Variance explained (%)")
    ax.set_ylim(0, max(pct) * 1.22)
    return _png(fig)


def volcano(assoc: pd.DataFrame, outcome: str, theme: Theme,
            alpha: float = 0.05, n_label: int = 6) -> bytes:
    """Effect size against evidence, one point per feature.

    Direction is the only thing colour encodes here, so it uses the diverging
    pair with grey for "no call" rather than a categorical ramp.
    """
    est = assoc["estimate"].to_numpy()
    logp = -np.log10(assoc["p_value"].to_numpy().clip(min=1e-300))
    sig = assoc["significant"].to_numpy()

    colors = np.where(
        ~sig, theme.neutral,
        np.where(est < 0, theme.diverging_low, theme.diverging_high),
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    _style(theme, ax)
    ax.scatter(est, logp, s=MARKER, c=colors, edgecolors=theme.surface,
               linewidths=0.8, zorder=2)

    # One dashed guide at the significance boundary, labelled in words.
    if sig.any():
        cutoff = logp[sig].min()
        ax.axhline(cutoff, color=theme.text_muted, linewidth=1.2,
                   linestyle="--", zorder=1)
        ax.text(ax.get_xlim()[0], cutoff, f"  FDR {alpha:g}", va="bottom",
                ha="left", color=theme.text_muted, fontsize=9)

    for fid in assoc.index[sig][:n_label]:
        row = assoc.loc[fid]
        ax.annotate(
            fid, (row["estimate"], -np.log10(max(row["p_value"], 1e-300))),
            textcoords="offset points", xytext=(6, 4), fontsize=8.5,
            color=theme.text_primary,
        )

    ax.axvline(0, color=theme.border, linewidth=1, zorder=1)
    ax.set_xlabel(f"log2 abundance change per SD of {OUTCOMES[outcome].lower()}")
    ax.set_ylabel("-log10 p-value")

    # Identity is direction, and direction is also given by the x-axis sign,
    # but a legend keeps it from being colour-alone.
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                   markerfacecolor=theme.diverging_low,
                   markeredgecolor=theme.surface, label="Lower with outcome"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                   markerfacecolor=theme.diverging_high,
                   markeredgecolor=theme.surface, label="Higher with outcome"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                   markerfacecolor=theme.neutral,
                   markeredgecolor=theme.surface, label="Not significant"),
    ]
    _legend(theme, ax, handles=handles, loc="upper left")
    return _png(fig)


def forest(assoc: pd.DataFrame, outcome: str, theme: Theme,
           top_n: int = 10) -> bytes:
    """Point estimate and 95% interval for the strongest associations."""
    top = assoc.head(top_n).iloc[::-1]
    if top.empty:
        top = assoc.head(1)
    y = np.arange(len(top))
    est = top["estimate"].to_numpy()
    ci = 1.96 * top["std_error"].to_numpy()
    colors = [
        theme.diverging_low if e < 0 else theme.diverging_high for e in est
    ]

    fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(top) + 1.4))
    _style(theme, ax)
    ax.grid(False, axis="y")
    ax.axvline(0, color=theme.text_muted, linewidth=1.2, linestyle="--", zorder=1)
    for i, (e, c, colour) in enumerate(zip(est, ci, colors, strict=True)):
        ax.plot([e - c, e + c], [i, i], color=colour, linewidth=2, zorder=2,
                solid_capstyle="round")
        ax.plot([e], [i], marker="o", markersize=7, color=colour,
                markeredgecolor=theme.surface, markeredgewidth=1.2, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(top.index, fontsize=9)
    ax.set_xlabel(f"log2 abundance change per SD of {OUTCOMES[outcome].lower()}")
    ax.set_ylim(-0.8, len(top) - 0.2)
    return _png(fig)
