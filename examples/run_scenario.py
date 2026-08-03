from __future__ import annotations

from jasa_mcmcis import (
    estimate_scale_T,
    iid_pilot_statistics,
    init_beta_from_iid_pilot,
    load_scenario,
    run_mcmc_is,
    run_samc,
)


def main() -> None:
    scenario = load_scenario("gwas_additive_score_near_v01_n140")
    problem = scenario.problem
    # Swap fraction of the smaller group, matching the article's cross-method grid.
    proposal_size = 0.05

    pilot_T = iid_pilot_statistics(problem, n_samples=5_000, seed=11)
    sigma_t = estimate_scale_T(pilot_T)
    beta = init_beta_from_iid_pilot(
        pilot_T,
        problem.t_obs,
        sigma_t,
        p0_reference=scenario.exact_p_value,
        gamma=1.0 / 3.0,
    )

    mcmcis = run_mcmc_is(
        problem,
        beta=beta,
        sigma_t=sigma_t,
        n_steps=50_000,
        burn_in=10_000,
        n_chains=2,
        proposal_size=proposal_size,
        seed=123,
    )
    samc = run_samc(
        problem,
        n_steps=200_000,
        n_bins=101,
        proposal_size=proposal_size,
        seed=456,
    )

    print(f"scenario: {scenario.key}")
    print(f"exact p:  {scenario.exact_p_value:.6g}")
    print(f"MCMC-IS:  {mcmcis.estimate:.6g} (ESS {mcmcis.ess:.1f})")
    print(f"SAMC:     {samc.estimate:.6g}")


if __name__ == "__main__":
    main()
