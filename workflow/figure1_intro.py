"""Reproduce the main-text introductory tilt-geometry figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import style


WINDOW = 200_000
TRACE_STRIDE = 20
TARGET_EVENT_MASS = 0.50
SEEDS = {"step": 2026080101, "smooth": 2026080102}


def normalized_tilt(f: np.ndarray, tilt: np.ndarray) -> np.ndarray:
    target = f * tilt
    return target / target.sum()


def lattice(k: int = 20, a_size: int = 6, rho: float = 0.02, alpha: float = 2.0) -> dict:
    last = k + a_size - 1
    states = np.arange(last + 1)
    f = np.exp(-rho * states**alpha)
    f /= f.sum()
    p = float(f[k:].sum())
    step_r = TARGET_EVENT_MASS / (1.0 - TARGET_EVENT_MASS) * (1.0 - p) / p
    step = normalized_tilt(f, 1.0 + (step_r - 1.0) * (states >= k))
    shortfall = np.maximum(k - states, 0.0)

    def mass(beta: float) -> float:
        return float(normalized_tilt(f, np.exp(-beta * shortfall))[states >= k].sum())

    lo, hi = 0.0, 1.0
    while mass(hi) < TARGET_EVENT_MASS:
        hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mass(mid) < TARGET_EVENT_MASS else (lo, mid)
    beta = (lo + hi) / 2
    smooth = normalized_tilt(f, np.exp(-beta * shortfall))
    return {"states": states, "k": k, "last": last, "f": f, "step": step, "smooth": smooth, "p": p, "step_r": step_r, "beta": beta}


def simulate(target: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    states = np.empty(WINDOW, dtype=np.int16)
    states[0] = rng.choice(len(target), p=target)
    moves = rng.choice(np.asarray([-1, 1], dtype=np.int16), WINDOW - 1)
    uniforms = rng.random(WINDOW - 1)
    for index in range(1, WINDOW):
        current = states[index - 1]
        proposed = current + moves[index - 1]
        if 0 <= proposed < len(target) and uniforms[index - 1] < min(1.0, target[proposed] / target[current]):
            states[index] = proposed
        else:
            states[index] = current
    return states


def build_figure(output_stem: Path) -> dict:
    style.apply()
    model = lattice()
    traces = {name: simulate(model[name], SEEDS[name]) for name in ("step", "smooth")}
    colors, dashes = style.COLORS, style.DASHES
    k, last = model["k"], model["last"]
    fig = plt.figure(figsize=(style.TEXTWIDTH_IN, 2.85))
    grid = GridSpec(2, 2, figure=fig, width_ratios=[0.43, 0.57], wspace=0.36, hspace=0.60)
    left = fig.add_subplot(grid[:, 0])
    top = fig.add_subplot(grid[0, 1])
    bottom = fig.add_subplot(grid[1, 1], sharex=top, sharey=top)
    left.set_yscale("log")
    left.axvspan(k, last, color="#E8E8E8", alpha=0.65, lw=0)
    left.plot(model["states"], model["f"], color=colors["baseline"], dashes=dashes["baseline"], label=r"baseline $f$")
    left.plot(model["states"], model["smooth"], color=colors["smooth"], dashes=dashes["smooth"], label="smooth target")
    left.plot(model["states"], model["step"], color=colors["step"], dashes=dashes["step"], label="step target")
    left.axvline(k, color=colors["rule"], lw=0.5, dashes=(3, 2))
    left.set(xlabel="state i", ylabel="probability mass", xlim=(0, last), ylim=(2e-7, 3))
    left.set_xticks([0, k, last], ["0", "k", "L_k"])
    left.legend(loc="lower left", fontsize=7.5)
    style.despine(left)
    time = np.arange(WINDOW) / 1000
    shown = slice(None, None, TRACE_STRIDE)
    for ax, name, title, width in ((top, "step", "Step target", 0.40), (bottom, "smooth", "Smooth target", 0.25)):
        ax.axhspan(k, last + 1.5, color="#E8E8E8", alpha=0.65, lw=0)
        ax.plot(time[shown], traces[name][shown], color=colors[name], lw=width, alpha=0.75, rasterized=True)
        ax.axhline(k, color=colors["rule"], lw=0.5, dashes=(3, 2))
        ax.set_ylim(-1.5, last + 1.5)
        ax.set_yticks([0, k], ["0", "k"])
        ax.text(0, 1.045, title, transform=ax.transAxes, fontsize=8.2)
        style.despine(ax)
    top.tick_params(labelbottom=False)
    bottom.set_xlabel("iteration (thousands)")
    top.set_xlim(0, WINDOW / 1000)
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"))
    plt.close(fig)
    metadata = {
        "p_k": model["p"], "target_event_mass": TARGET_EVENT_MASS,
        "r_step": model["step_r"], "beta_smooth": model["beta"], "window": WINDOW,
        "seeds": SEEDS,
        "up_crossings": {name: int(np.sum((trace[:-1] == k - 1) & (trace[1:] == k))) for name, trace in traces.items()},
    }
    output_stem.with_name(output_stem.name + "_meta").with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
