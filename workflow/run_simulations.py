#!/usr/bin/env python3
"""Run the article simulations either as a quick smoke test or in full.

``--mode smoke`` runs two scenarios and two short checkpoints.  ``--mode full``
runs the complete 28-block near-threshold grid and both wide-ratio banks.  The
full mode is intentionally explicit and is expected to require multi-core or
cluster execution for practical turnaround.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np

from jasa_mcmcis import (
    BETA_INIT_PILOT_SAMPLES,
    CHECKPOINTS,
    GAMMAS,
    GWAS_NEAR_THRESHOLD_SCENARIO_KEYS,
    HEP_NEAR_THRESHOLD_SCENARIO_KEYS,
    MCMCIS_SWAP_FRACTIONS,
    MCMC_IS_BETA_PILOT_SEED_OFFSET,
    MCMC_IS_CHAIN_SEED_OFFSET,
    SAMC_CHAIN_SEED_OFFSET,
    SAMC_SETUP_SEED_OFFSET,
    SAMC_SWAP_FRACTIONS,
    PermutationTestProblem,
    estimate_scale_T,
    iid_pilot_statistics,
    init_beta_from_iid_pilot,
    job_seed,
    load_scenario,
    mcmc_is_chain_checkpoints,
    run_mcmc_is_checkpoints,
    run_samc_checkpoints,
    samc_chain_checkpoints,
    treated_sum,
)


ROOT = Path(__file__).resolve().parents[1]


def clean(value):
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def mcmc_rows(scenario, family: str, scenario_index: int, swap: float, gamma: float,
              checkpoints: tuple[int, ...], pilot_samples: int, base_seed: int) -> list[dict]:
    problem = scenario.problem
    started = time.perf_counter()
    pilot = iid_pilot_statistics(problem, pilot_samples, seed=base_seed + MCMC_IS_BETA_PILOT_SEED_OFFSET)
    sigma = estimate_scale_T(pilot)
    p0 = float(scenario.extra["known_significance_threshold"])
    beta = init_beta_from_iid_pilot(pilot, problem.t_obs, sigma, p0_reference=p0, gamma=gamma)
    production = tuple(checkpoint - pilot_samples for checkpoint in checkpoints)
    results = run_mcmc_is_checkpoints(
        problem, beta=beta, sigma_t=sigma, checkpoint_steps=production,
        burn_in_fraction=0.20, proposal_size=swap, seed=base_seed + MCMC_IS_CHAIN_SEED_OFFSET,
    )
    elapsed = time.perf_counter() - started
    return [{
        "method": "mcmc_is_no_oracle", "checkpoint": checkpoint, "estimate": result.estimate,
        "exact_p": scenario.exact_p_value, "family": family, "scenario": scenario.key,
        "threshold_band": scenario.extra["threshold_band"], "p0_reference": p0,
        "p_over_p0": scenario.exact_p_value / p0, "swap_fraction": swap,
        "gamma": gamma, "gamma_label": str(gamma), "beta": beta, "sigma_t": sigma,
        "ess": result.ess, "acceptance_rate": result.acceptance_rate,
        "n_weighted_samples": result.n_weighted_samples, "seed": base_seed,
        "pilot_samples": pilot_samples, "wall_time_sec_total_trajectory": elapsed,
    } for checkpoint, result in zip(checkpoints, results)]


def samc_rows(scenario, family: str, scenario_index: int, swap: float,
              checkpoints: tuple[int, ...], pilot_samples: int, base_seed: int) -> list[dict]:
    started = time.perf_counter()
    production = tuple(checkpoint - pilot_samples for checkpoint in checkpoints)
    results = run_samc_checkpoints(
        scenario.problem, checkpoint_steps=production, n_bins=101,
        lambda_min_samples=pilot_samples, proposal_size=swap,
        seed=base_seed + SAMC_CHAIN_SEED_OFFSET,
        lambda_min_seed=base_seed + SAMC_SETUP_SEED_OFFSET,
    )
    elapsed = time.perf_counter() - started
    p0 = float(scenario.extra["known_significance_threshold"])
    return [{
        "method": "samc", "checkpoint": checkpoint, "estimate": result.estimate,
        "exact_p": scenario.exact_p_value, "family": family, "scenario": scenario.key,
        "threshold_band": scenario.extra["threshold_band"], "p0_reference": p0,
        "p_over_p0": scenario.exact_p_value / p0, "swap_fraction": swap,
        "gamma": None, "gamma_label": "none", "acceptance_rate": result.acceptance_rate,
        "samc_max_rel_freq_error": result.max_abs_relative_frequency_error,
        "samc_empty_bins": int(result.empty_bin_indices.size), "samc_lambda_min": result.lambda_min,
        "seed": base_seed, "pilot_samples": pilot_samples, "wall_time_sec_total_trajectory": elapsed,
    } for checkpoint, result in zip(checkpoints, results)]


class WideScenario:
    def __init__(self, directory: Path):
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        self.key = metadata["key"]
        self.extra = metadata["extra"]
        self.exact_p_value = float(metadata["exact_p_value"])
        self.problem = PermutationTestProblem(
            x=np.load(directory / "X.npy"), y_obs=np.load(directory / "y_obs.npy"),
            statistic=treated_sum, tail="right",
        )


def wide_seed_map() -> dict[str, int]:
    path = ROOT / "data" / "reference" / "wide_ratio" / "mcmcis_records.jsonl.gz"
    seeds: dict[str, int] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            seeds.setdefault(str(row["scenario"]), int(row["seed"]))
    return seeds


def run(mode: str) -> list[dict]:
    if mode == "smoke":
        checkpoints, mcmc_pilot, samc_pilot = (2_500, 5_000), 500, 100
        jobs = [
            ("gwas", load_scenario(GWAS_NEAR_THRESHOLD_SCENARIO_KEYS[0]), 0),
            ("hep", load_scenario(HEP_NEAR_THRESHOLD_SCENARIO_KEYS[0]), 0),
        ]
        rows = []
        for family, scenario, index in jobs:
            rows.extend(mcmc_rows(scenario, family, index, 0.05, 0.4, checkpoints, mcmc_pilot, 1_000_000 + index))
            rows.extend(samc_rows(scenario, family, index, 0.05, checkpoints, samc_pilot, 2_000_000 + index))
        return rows

    rows = []
    family_keys = {"gwas": GWAS_NEAR_THRESHOLD_SCENARIO_KEYS, "hep": HEP_NEAR_THRESHOLD_SCENARIO_KEYS}
    for family, keys in family_keys.items():
        for index, key in enumerate(keys):
            scenario = load_scenario(key)
            for swap in MCMCIS_SWAP_FRACTIONS:
                for gamma in GAMMAS:
                    seed = job_seed(family, swap, "mcmc_is", scenario_index=index, gamma=gamma)
                    rows.extend(mcmc_rows(scenario, family, index, swap, gamma, CHECKPOINTS, BETA_INIT_PILOT_SAMPLES, seed))
            for swap in SAMC_SWAP_FRACTIONS:
                seed = job_seed(family, swap, "samc", scenario_index=index)
                rows.extend(samc_rows(scenario, family, index, swap, CHECKPOINTS, 10_000, seed))

    seeds = wide_seed_map()
    for band in ("significant", "non_significant"):
        root = ROOT / "data" / "wide_ratio" / band
        for index, directory in enumerate(sorted(path for path in root.iterdir() if path.is_dir())):
            scenario = WideScenario(directory)
            rows.extend(mcmc_rows(scenario, "gwas", index, 0.05, 0.4, CHECKPOINTS, 100_000, seeds[scenario.key]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "simulation_smoke.jsonl")
    args = parser.parse_args()
    rows = run(args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(clean(row), allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "mode": args.mode, "records": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
