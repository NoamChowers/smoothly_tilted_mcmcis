"""Shared, print-sized plotting style for the article figures."""

from __future__ import annotations

import shutil

import matplotlib as mpl


TEXTWIDTH_IN = 390.0 / 72.27
COLORS = {"smooth": "#4C72B0", "step": "#C44E52", "samc": "#55A868", "baseline": "#7F7F7F", "rule": "#B0B0B0"}
DASHES = {"smooth": (None, None), "step": (4.0, 1.6), "samc": (1.2, 1.2), "baseline": (0.8, 1.4)}


def apply() -> None:
    use_tex = shutil.which("latex") is not None
    mpl.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.6,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,
    })


def despine(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
