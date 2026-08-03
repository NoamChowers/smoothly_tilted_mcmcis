from __future__ import annotations

import numpy as np

from jasa_mcmcis import (
    BETA_INIT_PILOT_SAMPLES,
    CROSS_METHOD_SCENARIO_KEYS,
    GWAS_NEAR_THRESHOLD_SCENARIO_KEYS,
    HEP_NEAR_THRESHOLD_SCENARIO_KEYS,
    LinearStatisticDPSolver,
    MCMC_IS_BETA_PILOT_SEED_OFFSET,
    MCMC_IS_CHAIN_SEED_OFFSET,
    NEAR_THRESHOLD_SCENARIO_KEYS,
    SAMC_CHAIN_SEED_OFFSET,
    SAMC_SETUP_SEED_OFFSET,
    PermutationTestProblem,
    available_scenarios,
    difference_in_means,
    estimate_scale_T,
    hard_step_beta_for_target_tail_mass,
    hard_step_r_for_target_tail_mass,
    iid_pilot_statistics,
    init_beta_from_iid_pilot,
    job_seed,
    load_scenario,
    load_scenarios,
    method_block_index,
    run_hard_step_mcmc_is,
    run_mcmc_is,
    run_mcmc_is_checkpoints,
    run_samc,
    run_samc_checkpoints,
)


def test_bundled_scenarios_match_the_cross_method_inventory() -> None:
    assert available_scenarios() == CROSS_METHOD_SCENARIO_KEYS
    assert len(GWAS_NEAR_THRESHOLD_SCENARIO_KEYS) == 80
    assert len(HEP_NEAR_THRESHOLD_SCENARIO_KEYS) == 80
    assert len(NEAR_THRESHOLD_SCENARIO_KEYS) == 160
    scenarios = load_scenarios()
    assert [scenario.key for scenario in scenarios] == list(CROSS_METHOD_SCENARIO_KEYS)
    for scenario in scenarios:
        assert scenario.exact_p_value > 0.0
        assert scenario.tail_hits > 0
        assert scenario.problem.tail == "right"


def test_linear_dp_matches_bundled_metadata_for_gwas_scenario() -> None:
    scenario = load_scenario("gwas_additive_score_near_v01_n140")
    exact = LinearStatisticDPSolver.from_scenario(scenario).compute()
    assert exact.tail_hits == scenario.tail_hits
    assert exact.n_permutations == scenario.n_permutations
    assert np.isclose(exact.p_value, scenario.exact_p_value, rtol=1e-15, atol=0.0)


def test_linear_dp_matches_bundled_metadata_for_hep_scenario() -> None:
    scenario = load_scenario("poisson_diffmeans_hep_near_v01_n200")
    exact = LinearStatisticDPSolver.from_scenario(scenario).compute()
    assert exact.tail_hits == scenario.tail_hits
    assert exact.n_permutations == scenario.n_permutations
    assert np.isclose(exact.p_value, scenario.exact_p_value, rtol=1e-15, atol=0.0)


def test_near_threshold_inventory_metadata_is_in_band() -> None:
    scenarios = load_scenarios(NEAR_THRESHOLD_SCENARIO_KEYS)
    assert len(scenarios) == 160

    for scenario in scenarios:
        p0 = float(scenario.extra["known_significance_threshold"])
        ratio = float(scenario.extra["p_over_p0"])
        assert np.isclose(ratio, scenario.exact_p_value / p0, rtol=0.0, atol=0.0)
        assert 0.75 <= ratio <= 0.99
        assert scenario.extra["threshold_band"] == "near"
        assert "near_threshold_variety" in scenario.portfolio["groups"]


def _toy_problem() -> PermutationTestProblem:
    x = np.arange(10, dtype=float)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int8)
    return PermutationTestProblem(x=x, y_obs=y, statistic=difference_in_means, tail="right")


def test_mcmcis_runs_on_toy_problem() -> None:
    problem = _toy_problem()
    result = run_mcmc_is(
        problem,
        beta=2.0,
        sigma_t=1.0,
        n_steps=4_000,
        burn_in=500,
        thin=2,
        n_chains=2,
        seed=7,
        proposal_size=1,
    )
    assert 0.0 <= result.estimate <= 1.0
    assert result.ess > 0.0
    assert result.mcse_obm is not None
    assert result.n_weighted_samples == 3500


def test_hard_step_mcmcis_uses_calibrated_step_tilt() -> None:
    problem = _toy_problem()
    p0 = 0.05
    q = 0.30
    r = hard_step_r_for_target_tail_mass(p0, q)
    beta = hard_step_beta_for_target_tail_mass(p0, q)
    result = run_hard_step_mcmc_is(
        problem,
        p0=p0,
        q=q,
        n_steps=2_000,
        burn_in=500,
        thin=2,
        n_chains=2,
        seed=11,
        proposal_size=1,
    )

    assert np.isclose(r * p0 / (1.0 - p0 + r * p0), q, rtol=1e-15, atol=0.0)
    assert np.isclose(result.beta, beta)
    assert result.tilt_mode == "step"
    assert 0.0 <= result.estimate <= 1.0


def test_samc_runs_on_toy_problem() -> None:
    problem = _toy_problem()
    result = run_samc(
        problem,
        n_steps=4_000,
        n_bins=6,
        lambda_min=-5.0,
        seed=8,
        proposal_size=1,
        trace_every=100,
    )
    assert 0.0 <= result.estimate <= 1.0
    assert result.tail_bin_index == result.visit_counts.size - 1
    assert result.theta_trace.shape[1] == result.visit_counts.size
    assert int(result.visit_counts.sum()) == 4_000
    assert result.lambda_min_evaluations == 0


def test_mcmc_is_checkpoints_matches_single_run_at_the_same_budget() -> None:
    problem = _toy_problem()
    n_steps = 4_000
    burn_in_fraction = 0.20
    single = run_mcmc_is(
        problem,
        beta=2.0,
        sigma_t=1.0,
        n_steps=n_steps,
        burn_in=int(burn_in_fraction * n_steps),
        thin=2,
        n_chains=2,
        seed=7,
        proposal_size=1,
    )
    (checkpointed,) = run_mcmc_is_checkpoints(
        problem,
        beta=2.0,
        sigma_t=1.0,
        checkpoint_steps=[n_steps],
        burn_in_fraction=burn_in_fraction,
        thin=2,
        n_chains=2,
        seed=7,
        proposal_size=1,
    )
    assert checkpointed.n_weighted_samples == single.n_weighted_samples
    assert np.isclose(checkpointed.estimate, single.estimate, rtol=0.0, atol=0.0)
    assert np.isclose(checkpointed.ess, single.ess, rtol=0.0, atol=0.0)
    assert np.isclose(checkpointed.mcse_obm, single.mcse_obm, rtol=0.0, atol=0.0)
    assert np.array_equal(checkpointed.t_samples, single.t_samples)


def test_mcmc_is_checkpoints_are_monotone_in_budget() -> None:
    problem = _toy_problem()
    results = run_mcmc_is_checkpoints(
        problem,
        beta=2.0,
        sigma_t=1.0,
        checkpoint_steps=[1_000, 2_000, 4_000],
        n_chains=2,
        seed=7,
        proposal_size=1,
    )
    assert [r.n_weighted_samples for r in results] == sorted(r.n_weighted_samples for r in results)
    assert all(0.0 <= r.estimate <= 1.0 for r in results)


def test_samc_checkpoints_matches_single_run_at_the_same_budget() -> None:
    problem = _toy_problem()
    single = run_samc(problem, n_steps=4_000, n_bins=6, lambda_min=-5.0, seed=8, proposal_size=1)
    (checkpointed,) = run_samc_checkpoints(
        problem,
        checkpoint_steps=[4_000],
        n_bins=6,
        lambda_min=-5.0,
        seed=8,
        proposal_size=1,
    )
    assert np.isclose(checkpointed.estimate, single.estimate, rtol=0.0, atol=0.0)
    assert np.array_equal(checkpointed.visit_counts, single.visit_counts)
    assert np.array_equal(checkpointed.theta_final, single.theta_final)
    assert checkpointed.acceptance_rate == single.acceptance_rate


def test_samc_checkpoints_are_monotone_in_budget() -> None:
    problem = _toy_problem()
    results = run_samc_checkpoints(
        problem,
        checkpoint_steps=[1_000, 2_000, 4_000],
        n_bins=6,
        lambda_min=-5.0,
        seed=8,
        proposal_size=1,
    )
    assert [r.n_steps for r in results] == [1_000, 2_000, 4_000]
    assert all(int(r.visit_counts.sum()) == r.n_steps for r in results)


def test_mcmc_is_checkpoints_reproduces_the_published_grid_result() -> None:
    # Ground truth: gwas_additive_score_near_v01_n140, swap_fraction=0.025,
    # gamma=0.25, method="mcmc_is", checkpoint=250,000 from the article's
    # cross-method grid (all_records.jsonl, seed 91337).
    scenario = load_scenario("gwas_additive_score_near_v01_n140")
    problem = scenario.problem
    base_seed = job_seed("gwas", swap_fraction=0.025, method="mcmc_is", gamma=0.25, scenario_index=0)
    assert base_seed == 91337

    pilot_T = iid_pilot_statistics(
        problem, n_samples=BETA_INIT_PILOT_SAMPLES, seed=base_seed + MCMC_IS_BETA_PILOT_SEED_OFFSET
    )
    sigma_t = estimate_scale_T(pilot_T)
    beta = init_beta_from_iid_pilot(
        pilot_T,
        problem.t_obs,
        sigma_t,
        p0_reference=scenario.extra["known_significance_threshold"],
        gamma=0.25,
    )
    assert np.isclose(beta, 3.716796875, rtol=0.0, atol=1e-9)

    (result,) = run_mcmc_is_checkpoints(
        problem,
        beta=beta,
        sigma_t=sigma_t,
        checkpoint_steps=[150_000],  # 250,000 minus the 100,000-sample beta pilot
        proposal_size=0.025,
        seed=base_seed + MCMC_IS_CHAIN_SEED_OFFSET,
    )
    assert result.n_weighted_samples == 120_000
    assert np.isclose(result.acceptance_rate, 0.56846, rtol=0.0, atol=1e-12)
    assert np.isclose(result.ess, 162.6644071397, rtol=0.0, atol=1e-6)
    assert np.isclose(result.estimate, 6.62e-08, rtol=0.02, atol=0.0)


def test_samc_checkpoints_reproduces_the_published_grid_result() -> None:
    # Ground truth: gwas_additive_score_near_v01_n140, swap_fraction=0.05,
    # method="samc", checkpoint=250,000 from the article's cross-method grid
    # (all_records.jsonl, seed 4,091,337).
    scenario = load_scenario("gwas_additive_score_near_v01_n140")
    problem = scenario.problem
    base_seed = job_seed("gwas", swap_fraction=0.05, method="samc", scenario_index=0)
    assert base_seed == 4_091_337

    (result,) = run_samc_checkpoints(
        problem,
        checkpoint_steps=[240_000],  # 250,000 minus the 10,000-sample lambda_0 pilot
        n_bins=101,
        lambda_min_samples=10_000,
        proposal_size=0.05,
        seed=base_seed + SAMC_CHAIN_SEED_OFFSET,
        lambda_min_seed=base_seed + SAMC_SETUP_SEED_OFFSET,
    )
    assert result.lambda_min == 24.0
    assert result.empty_bin_indices.size == 66
    assert np.isclose(result.acceptance_rate, 0.6150666667, rtol=0.0, atol=1e-9)
    assert np.isclose(result.pi0_adjustment, 0.0186704385, rtol=0.0, atol=1e-9)
    assert np.isclose(result.max_abs_relative_frequency_error, 2.1854166667, rtol=0.0, atol=1e-6)
    assert np.isclose(result.visitation_frequency[-1], 0.0291958333, rtol=0.0, atol=1e-9)
    assert np.isclose(result.estimate, 3.62e-08, rtol=0.02, atol=0.0)


def test_grid_job_seed_matches_the_published_cross_method_grid() -> None:
    # Ground truth taken from the article's run: gwas_additive_score_near_v01_n140
    # (scenario_index=0) at swap_fraction=0.025, gamma=0.25, method="mcmc_is" used
    # seed 91337 (the grid's unshifted base seed, i.e. block index 0).
    assert method_block_index("gwas", 0.025, "mcmc_is", 0.25) == 0
    assert job_seed("gwas", 0.025, "mcmc_is", scenario_index=0, gamma=0.25) == 91337
    # gwas_additive_score_near_v02_n140 (scenario_index=1) at swap_fraction=0.1,
    # method="samc" (block index 9) used seed 9_101_337.
    assert job_seed("gwas", 0.1, "samc", scenario_index=1) == 9_101_337
    # poisson_diffmeans_hep_near_v02_n200 (scenario_index=1) at swap_fraction=0.025,
    # gamma=0.25, method="mcmc_is" (block index 14, hep's first block) used
    # seed 14_101_337.
    assert job_seed("hep", 0.025, "mcmc_is", scenario_index=1, gamma=0.25) == 14_101_337
