#!/usr/bin/env python3
"""Rebuild the article's computational tables and figures from cached records.

The expensive Monte Carlo trajectories are immutable inputs under
``data/reference``.  This script validates those inputs, recomputes every
reported point estimate from the scenario-level records, joins the published
one-million-resample BCa intervals, and writes manuscript-ready outputs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/jasa-mcmcis-matplotlib")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "reference"
ARTICLE_BUDGETS = (500_000, 1_000_000, 2_500_000, 5_000_000)
FAMILIES = ("gwas", "hep")
METHODS = ("mcmc_is_no_oracle", "samc")
FAMILY_LABELS = {"gwas": "GWAS-like", "hep": "HEP-like"}
METHOD_LABELS = {"mcmc_is_no_oracle": "Smooth MCMC-IS", "samc": "SAMC"}
COLORS = {"mcmc_is_no_oracle": "#2f6fb0", "samc": "#96882a"}
MARKERS = {"mcmc_is_no_oracle": "o", "samc": "s"}
RATIO_BCA = {
    ("gwas", 500_000): (0.450, 0.684),
    ("gwas", 1_000_000): (0.456, 0.631),
    ("gwas", 2_500_000): (0.475, 0.704),
    ("gwas", 5_000_000): (0.481, 0.807),
    ("hep", 500_000): (0.806, 1.272),
    ("hep", 1_000_000): (0.716, 1.006),
    ("hep", 2_500_000): (0.659, 1.066),
    ("hep", 5_000_000): (0.595, 0.887),
}


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def config_key(row: dict) -> tuple[str, str, float, str]:
    return (
        str(row["family"]),
        str(row["method"]),
        round(float(row["swap_fraction"]), 10),
        str(row.get("gamma_label", "none")),
    )


def rrmse(rows: list[dict]) -> float:
    errors = np.asarray(
        [float(row["estimate"]) / float(row["exact_p"]) - 1.0 for row in rows],
        dtype=float,
    )
    return float(np.sqrt(np.mean(errors * errors)))


def validate_main_records(records: list[dict]) -> dict:
    scenarios = {(str(row["family"]), str(row["scenario"])) for row in records}
    checkpoints = sorted({int(row["checkpoint"]) for row in records})
    configs = {config_key(row) for row in records}
    expected_checkpoints = list(range(250_000, 5_000_001, 250_000))
    if len(records) != 44_800:
        raise RuntimeError(f"Expected 44,800 main-grid records; found {len(records):,}.")
    if len(scenarios) != 160 or checkpoints != expected_checkpoints or len(configs) != 28:
        raise RuntimeError("Main-grid inventory does not match the article design.")
    return {
        "records": len(records),
        "scenarios": len(scenarios),
        "configurations": len(configs),
        "checkpoints": checkpoints,
    }


def select_configs(records: list[dict]) -> dict[tuple[str, str], tuple[str, str, float, str]]:
    checkpoints = tuple(range(500_000, 5_000_001, 250_000))
    grouped: dict[tuple[tuple[str, str, float, str], int], list[dict]] = defaultdict(list)
    for row in records:
        cp = int(row["checkpoint"])
        if cp in checkpoints:
            grouped[(config_key(row), cp)].append(row)

    selected: dict[tuple[str, str], tuple[str, str, float, str]] = {}
    for family in FAMILIES:
        for method in METHODS:
            candidates = []
            for key in sorted({key for key, _ in grouped}):
                if key[:2] != (family, method):
                    continue
                values = [rrmse(grouped[(key, cp)]) for cp in checkpoints]
                if all(len(grouped[(key, cp)]) == 80 for cp in checkpoints):
                    candidates.append((float(np.mean(values)), key))
            if not candidates:
                raise RuntimeError(f"No complete configuration for {family}/{method}.")
            selected[(family, method)] = min(candidates)[1]
    return selected


def selected_records(records: list[dict], selected: dict) -> list[dict]:
    return [row for row in records if config_key(row) == selected[(row["family"], row["method"])]]


def checkpoint_metrics(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in records:
        if int(row["checkpoint"]) >= 500_000:
            grouped[(row["family"], row["method"], int(row["checkpoint"]))].append(row)
    output = []
    for (family, method, checkpoint), rows in sorted(grouped.items(), key=lambda item: (item[0][2], item[0][0], item[0][1])):
        exact = np.asarray([float(row["exact_p"]) for row in rows])
        estimate = np.asarray([float(row["estimate"]) for row in rows])
        p0 = np.asarray([float(row["p0_reference"]) for row in rows])
        rel = estimate / exact - 1.0
        output.append(
            {
                "family": family,
                "method": method,
                "checkpoint": checkpoint,
                "checkpoint_millions": checkpoint / 1_000_000,
                "n_scenarios": len(rows),
                "rrmse": float(np.sqrt(np.mean(rel * rel))),
                "median_are": float(np.median(np.abs(rel))),
                "mean_phat_over_p": float(np.mean(estimate / exact)),
                "mean_relative_bias": float(np.mean(rel)),
                "false_negative_count": int(np.sum((exact < p0) & (estimate > p0))),
                "false_negative_rate": float(np.mean((exact < p0) & (estimate > p0))),
            }
        )
    return output


def metric_lookup(metrics: list[dict]) -> dict[tuple[str, str, int], dict]:
    return {(row["family"], row["method"], int(row["checkpoint"])): row for row in metrics}


def main_table(metrics: list[dict]) -> list[dict]:
    lookup = metric_lookup(metrics)
    rows = []
    for family in FAMILIES:
        for checkpoint in ARTICLE_BUDGETS:
            smooth = lookup[(family, "mcmc_is_no_oracle", checkpoint)]["rrmse"]
            samc = lookup[(family, "samc", checkpoint)]["rrmse"]
            low, high = RATIO_BCA[(family, checkpoint)]
            rows.append(
                {
                    "family": FAMILY_LABELS[family],
                    "budget": checkpoint,
                    "mcmc_is_rrmse": smooth,
                    "samc_rrmse": samc,
                    "rrmse_ratio": smooth / samc,
                    "ratio_bca_ci95_low": low,
                    "ratio_bca_ci95_high": high,
                    "bootstrap_resamples": 1_000_000,
                    "bootstrap_pairing": "scenario",
                }
            )
    return rows


def write_main_table_tex(path: Path, rows: list[dict]) -> None:
    lines = [
        r"\begin{tabular}{llccc}", r"\toprule",
        r"Family & Budget & MCMC--IS RRMSE & SAMC RRMSE & Ratio (95\% BCa CI) \\",
        r"\midrule",
    ]
    for row in rows:
        budget = f"{row['budget'] / 1_000_000:g}M"
        interval = f"{row['rrmse_ratio']:.3f} [{row['ratio_bca_ci95_low']:.3f}, {row['ratio_bca_ci95_high']:.3f}]"
        lines.append(
            f"{row['family']} & {budget} & {row['mcmc_is_rrmse']:.3f} & {row['samc_rrmse']:.3f} & {interval} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _style(ax: plt.Axes, ylabel: str, reference: float | None = None) -> None:
    ax.grid(axis="y", color="#d9dde0", linewidth=0.7, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Total budget (millions)")
    ax.set_ylabel(ylabel)
    if reference is not None:
        ax.axhline(reference, color="#666666", linestyle=(0, (2, 2)), linewidth=1.0)


def plot_metric(metrics: list[dict], field: str, ylabel: str, path: Path, reference: float | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.75))
    for ax, family in zip(axes, FAMILIES):
        for method in METHODS:
            rows = sorted((row for row in metrics if row["family"] == family and row["method"] == method), key=lambda row: row["checkpoint"])
            ax.plot(
                [row["checkpoint_millions"] for row in rows],
                [row[field] for row in rows],
                color=COLORS[method], marker=MARKERS[method], linewidth=1.8,
                markerfacecolor="white" if method == "samc" else COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax.set_title(FAMILY_LABELS[family])
        _style(ax, ylabel, reference)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_rrmse(metrics: list[dict], path: Path) -> None:
    cached = read_csv(REFERENCE / "main_grid" / "diagnostic_rrmse_bca_bootstrap_summary.csv")
    intervals = {
        (row["family"], row["method"], int(row["checkpoint"])): (float(row["bca_ci95_low"]), float(row["bca_ci95_high"]))
        for row in cached
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.75))
    for ax, family in zip(axes, FAMILIES):
        for method in METHODS:
            rows = sorted((row for row in metrics if row["family"] == family and row["method"] == method), key=lambda row: row["checkpoint"])
            x = np.asarray([row["checkpoint_millions"] for row in rows])
            y = np.asarray([row["rrmse"] for row in rows])
            lo = np.asarray([intervals[(family, method, row["checkpoint"])][0] for row in rows])
            hi = np.asarray([intervals[(family, method, row["checkpoint"])][1] for row in rows])
            ax.fill_between(x, lo, hi, color=COLORS[method], alpha=0.14, linewidth=0)
            ax.plot(x, y, color=COLORS[method], marker=MARKERS[method], linewidth=1.8,
                    markerfacecolor="white" if method == "samc" else COLORS[method], label=METHOD_LABELS[method])
        ax.set_title(FAMILY_LABELS[family])
        _style(ax, "Cross-scenario RRMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_ratio(records: list[dict], path: Path) -> None:
    rows = [row for row in records if int(row["checkpoint"]) == 5_000_000]
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 6.6), sharex=True, sharey=False)
    for i, family in enumerate(FAMILIES):
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            subset = [row for row in rows if row["family"] == family and row["method"] == method]
            x = np.asarray([float(row["exact_p"]) / float(row["p0_reference"]) for row in subset])
            y = np.asarray([float(row["estimate"]) / float(row["p0_reference"]) for row in subset])
            ax.scatter(x, y, s=13, alpha=0.72, color=COLORS[method])
            limit = max(1.1, float(np.max(y)) * 1.03)
            ax.plot([0, limit], [0, limit], color="#777777", linestyle=":", linewidth=1)
            ax.axhline(1.0, color="#8d3328", linewidth=1)
            ax.set_xlim(0.72, 1.02)
            ax.set_ylim(0, limit)
            ax.set_title(f"{FAMILY_LABELS[family]} — {METHOD_LABELS[method]}")
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(0.02, 0.96, f"FN: {int(np.sum(y > 1.0))}/80", transform=ax.transAxes, va="top")
    for ax in axes[-1]:
        ax.set_xlabel(r"Exact $p/p_0$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Estimated $\hat p/p_0$")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_config_mean(path: Path) -> None:
    rows = read_csv(REFERENCE / "main_grid" / "config_mean_rrmse_paired_scenario_bca.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.4))
    for ax, family in zip(axes, FAMILIES):
        subset = sorted((row for row in rows if row["family"] == family), key=lambda row: float(row["mean_cross_scenario_rrmse"]))
        y = np.arange(len(subset))
        center = np.asarray([float(row["mean_cross_scenario_rrmse"]) for row in subset])
        lo = np.asarray([float(row["bca_ci95_low"]) for row in subset])
        hi = np.asarray([float(row["bca_ci95_high"]) for row in subset])
        colors = [COLORS[row["method"]] for row in subset]
        for pos, value, lower, upper, color in zip(y, center, lo, hi, colors):
            ax.errorbar(value, pos, xerr=[[value - lower], [upper - value]], fmt="o", color=color, capsize=3)
        ax.set_yticks(y, [row["config_label"].replace("\\gamma", "gamma") for row in subset], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_title(FAMILY_LABELS[family])
        ax.set_xlabel("Mean cross-scenario RRMSE")
        ax.grid(axis="x", color="#d9dde0", linewidth=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_top_configs(path: Path) -> None:
    rows = read_csv(REFERENCE / "main_grid" / "top_mcmcis_configs_and_best_samc_rrmse_by_budget_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.15))
    for ax, family in zip(axes, FAMILIES):
        labels = sorted({row["config_label"] for row in rows if row["family"] == family})
        for index, label in enumerate(labels):
            subset = sorted((row for row in rows if row["family"] == family and row["config_label"] == label), key=lambda row: int(row["checkpoint"]))
            method = subset[0]["method"]
            ax.plot([float(row["checkpoint_millions"]) for row in subset], [float(row["cross_scenario_rrmse"]) for row in subset],
                    label=label.replace("\\gamma", "gamma"), linewidth=2 if index == 0 else 1.35,
                    linestyle="--" if method == "samc" else "-", marker=MARKERS[method], markersize=3.2)
        ax.set_title(FAMILY_LABELS[family])
        _style(ax, "Cross-scenario RRMSE")
        ax.legend(frameon=False, fontsize=7.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_wide_ratio(path: Path) -> None:
    rows = read_csv(REFERENCE / "wide_ratio" / "mcmcis_rrmse_by_pvalue_half_bca.csv")
    bands = ("significant", "non_significant")
    titles = {"significant": "Clearly significant", "non_significant": "Clearly non-significant"}
    half_colors = {"Lower p-value half": "#2f6fb0", "Upper p-value half": "#d07a2d"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    for ax, band in zip(axes, bands):
        for half in half_colors:
            subset = sorted((row for row in rows if row["threshold_band"] == band and row["p_half"] == half), key=lambda row: int(row["checkpoint"]))
            x = np.asarray([float(row["budget_millions"]) for row in subset])
            y = np.asarray([float(row["observed_rrmse"]) for row in subset])
            lo = np.asarray([float(row["bca_ci95_low"]) for row in subset])
            hi = np.asarray([float(row["bca_ci95_high"]) for row in subset])
            ax.fill_between(x, lo, hi, color=half_colors[half], alpha=0.14)
            ax.plot(x, y, color=half_colors[half], label=half, linewidth=1.8)
        ax.set_title(titles[band])
        _style(ax, "Cross-scenario RRMSE")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_obm(path: Path) -> None:
    rows = read_csv(REFERENCE / "obm" / "scenario_checkpoint_sd_summary.csv")
    fig, ax = plt.subplots(figsize=(8.1, 4.1))
    positions, values, colors = [], [], []
    for index, checkpoint in enumerate(ARTICLE_BUDGETS):
        subset = [row for row in rows if int(row["checkpoint"]) == checkpoint]
        positions.extend([3 * index + 1, 3 * index + 2])
        values.extend([
            [float(row["empirical_relative_sd"]) for row in subset],
            [float(row["aggregate_obm_relative_sd"]) for row in subset],
        ])
        colors.extend(["#777777", "#2f6fb0"])
    boxes = ax.boxplot(values, positions=positions, widths=0.75, patch_artist=True, showfliers=False)
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.65)
    ax.set_xticks([1.5, 4.5, 7.5, 10.5], ["0.5M", "1M", "2.5M", "5M"])
    ax.set_ylabel("Relative standard deviation")
    ax.set_xlabel("Total budget")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend([boxes["boxes"][0], boxes["boxes"][1]], ["Across-run SD", "OBM estimate"], frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def scenario_table() -> list[dict]:
    catalog = json.loads((ROOT / "src" / "jasa_mcmcis" / "data" / "scenarios" / "catalog.json").read_text(encoding="utf-8"))
    rows = catalog["scenarios"]
    output = []
    for family, prefix in (("GWAS-like additive dosage", "gwas_"), ("HEP-like Poisson counts", "poisson_")):
        subset = [row for row in rows if row["key"].startswith(prefix)]
        output.append({
            "family": family,
            "n": subset[0]["n"],
            "n_treated": subset[0]["n_treated"],
            "threshold_p0": subset[0]["extra"]["known_significance_threshold"],
            "scenarios": len(subset),
            "exact_p_min": min(float(row["exact_p_value"]) for row in subset),
            "exact_p_max": max(float(row["exact_p_value"]) for row in subset),
        })
    return output


def pilot_allocation_table() -> list[dict]:
    rows = []
    for budget in ARTICLE_BUDGETS:
        mcmc_production = budget - 100_000
        rows.append({
            "total_budget": budget,
            "mcmc_is_pilot": 100_000,
            "mcmc_is_burn_in": int(0.20 * mcmc_production),
            "mcmc_is_retained": mcmc_production - int(0.20 * mcmc_production),
            "samc_pilot": 10_000,
            "samc_retained": budget - 10_000,
        })
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reproduce(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    main_records = read_jsonl_gz(REFERENCE / "main_grid" / "all_records.jsonl.gz")
    inventory = validate_main_records(main_records)
    selected = select_configs(main_records)
    fixed = selected_records(main_records, selected)
    metrics = checkpoint_metrics(fixed)

    expected = {("gwas", "mcmc_is_no_oracle"): (0.05, "0.4"), ("gwas", "samc"): (0.05, "none"),
                ("hep", "mcmc_is_no_oracle"): (0.05, "0.4"), ("hep", "samc"): (0.05, "none")}
    for key, (swap, gamma) in expected.items():
        chosen = selected[key]
        if not (math.isclose(chosen[2], swap) and chosen[3] == gamma):
            raise RuntimeError(f"Configuration selection mismatch for {key}: {chosen}.")

    main_rows = main_table(metrics)
    write_csv(output_dir / "table_main_results.csv", main_rows)
    write_main_table_tex(output_dir / "table_main_results.tex", main_rows)
    write_csv(output_dir / "table_scenario_bank.csv", scenario_table())
    write_csv(output_dir / "table_pilot_production_allocation.csv", pilot_allocation_table())
    write_csv(output_dir / "diagnostic_checkpoint_metrics.csv", metrics)
    write_json(output_dir / "selected_configurations.json", {f"{key[0]}__{key[1]}": list(value) for key, value in selected.items()})

    plot_rrmse(metrics, output_dir / "diagnostic_rrmse_by_budget_bca.pdf")
    plot_metric(metrics, "false_negative_rate", "False-negative rate", output_dir / "diagnostic_false_negative_rate_by_budget.pdf")
    plot_metric(metrics, "mean_phat_over_p", r"Mean $\hat p/p$", output_dir / "diagnostic_mean_phat_over_p_by_budget.pdf", 1.0)
    plot_threshold_ratio(fixed, output_dir / "figure4_threshold_ratio_budget_5m.pdf")
    plot_config_mean(output_dir / "config_mean_rrmse_paired_scenario_bca.pdf")
    plot_top_configs(output_dir / "top_mcmcis_configs_and_best_samc_rrmse_by_budget.pdf")
    plot_wide_ratio(output_dir / "mcmcis_rrmse_by_pvalue_half_bca.pdf")
    plot_obm(output_dir / "mcmcis_obm_relative_sd_distributions.pdf")

    from figure1_intro import build_figure

    build_figure(output_dir / "figure1_intro_tilt_geometry")
    files = sorted(path for path in output_dir.iterdir() if path.is_file())
    manifest = {
        "status": "complete",
        "source_inventory": inventory,
        "selected_configurations": {f"{key[0]}__{key[1]}": list(value) for key, value in selected.items()},
        "outputs": [{"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size} for path in files],
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "reproduced")
    args = parser.parse_args()
    result = reproduce(args.output_dir.expanduser().resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
