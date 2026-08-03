#!/usr/bin/env python3
"""Re-run the repeated-chain OBM calibration reported in the supplement.

The default smoke mode uses one documented scenario, three short independent
chains, and two checkpoints. Full mode uses the archived selection of 20
scenarios, one fixed 100,000-draw pilot per scenario, 30 production chains,
and all 20 total-budget checkpoints through 5,000,000 evaluations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from jasa_mcmcis import (
    estimate_scale_T,
    iid_pilot_statistics,
    init_beta_from_iid_pilot,
    load_scenario,
    run_mcmc_is_checkpoints,
)


ROOT = Path(__file__).resolve().parents[1]
PILOT_SEED_BASE = 820_260_802
PRODUCTION_SEED_BASE = 920_260_802
FULL_CHECKPOINTS = tuple(range(250_000, 5_000_001, 250_000))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_keys(mode: str) -> list[str]:
    selection = json.loads(
        (ROOT / "data" / "reference" / "obm" / "scenario_selection.json").read_text(encoding="utf-8")
    )
    keys = [str(row["scenario"]) for row in selection]
    return keys[:1] if mode == "smoke" else keys


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    groups = sorted({(str(row["scenario"]), int(row["checkpoint"])) for row in rows})
    for scenario_key, checkpoint in groups:
        subset = [row for row in rows if row["scenario"] == scenario_key and row["checkpoint"] == checkpoint]
        estimates = np.asarray([row["estimate"] for row in subset], dtype=float)
        variances = np.asarray([row["variance_obm"] for row in subset], dtype=float)
        exact_p = float(subset[0]["exact_p"])
        empirical_sd = float(np.std(estimates, ddof=1))
        aggregate_obm_sd = float(np.sqrt(np.mean(variances)))
        output.append(
            {
                "scenario": scenario_key,
                "checkpoint": checkpoint,
                "n_chains": len(subset),
                "exact_p": exact_p,
                "p_over_p0": float(subset[0]["p_over_p0"]),
                "empirical_sd": empirical_sd,
                "aggregate_obm_sd": aggregate_obm_sd,
                "empirical_relative_sd": empirical_sd / exact_p,
                "aggregate_obm_relative_sd": aggregate_obm_sd / exact_p,
                "obm_over_empirical_sd_ratio": (
                    aggregate_obm_sd / empirical_sd if empirical_sd > 0.0 else ""
                ),
            }
        )
    return output


def run(mode: str) -> tuple[list[dict], list[dict]]:
    if mode == "smoke":
        pilot_samples, checkpoints, n_chains = 500, (2_500, 5_000), 3
    else:
        pilot_samples, checkpoints, n_chains = 100_000, FULL_CHECKPOINTS, 30

    rows: list[dict] = []
    for scenario_index, key in enumerate(selected_keys(mode)):
        scenario = load_scenario(key)
        p0 = float(scenario.extra["known_significance_threshold"])
        pilot_seed = PILOT_SEED_BASE + 1_000_000 * scenario_index
        pilot = iid_pilot_statistics(scenario.problem, pilot_samples, seed=pilot_seed)
        sigma_t = estimate_scale_T(pilot)
        beta = init_beta_from_iid_pilot(
            pilot, scenario.problem.t_obs, sigma_t, p0_reference=p0, gamma=0.4
        )
        production_checkpoints = tuple(checkpoint - pilot_samples for checkpoint in checkpoints)
        for chain_index in range(n_chains):
            production_seed = PRODUCTION_SEED_BASE + 1_000_000 * scenario_index + 1_000 * chain_index + 1
            results = run_mcmc_is_checkpoints(
                scenario.problem,
                beta=beta,
                sigma_t=sigma_t,
                checkpoint_steps=production_checkpoints,
                burn_in_fraction=0.20,
                proposal_size=0.05,
                seed=production_seed,
                estimate_variance=True,
            )
            for checkpoint, result in zip(checkpoints, results):
                rows.append(
                    {
                        "scenario": key,
                        "checkpoint": checkpoint,
                        "chain_index": chain_index,
                        "estimate": result.estimate,
                        "variance_obm": result.variance_obm,
                        "mcse_obm": result.mcse_obm,
                        "exact_p": scenario.exact_p_value,
                        "p_over_p0": scenario.exact_p_value / p0,
                        "beta": beta,
                        "sigma_t": sigma_t,
                        "pilot_seed": pilot_seed,
                        "production_seed": production_seed,
                    }
                )
    return rows, summarize(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "obm_smoke")
    args = parser.parse_args()
    rows, summary = run(args.mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "chain_checkpoint_records.csv", rows)
    write_csv(args.output_dir / "scenario_checkpoint_sd_summary.csv", summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "mode": args.mode,
                "records": len(rows),
                "summary_rows": len(summary),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
