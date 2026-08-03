# JASA MCMCIS Supplement

This is a compact implementation of rare permutation-test
p-value estimation. It contains:

- MCMC importance sampling (MCMC-IS) for fixed-size binary-label permutation
  tests, using a tilted distribution and self-normalized importance weights.
- Hard-step MCMC-IS, using a known reference threshold `p0` to set a fixed
  tail multiplier.
- Stochastic Approximation Monte Carlo (SAMC) as a comparison method.
- The 160 near-threshold scenarios (80 GWAS-like, 80 HEP-like) used in the
  article's cross-method comparison grid.
- A small optional dynamic-programming exact solver for linear statistics,
  included for validation of the bundled scenarios.

Data generation and large experiment orchestration are intentionally not
included. The bundled data are the observed arrays, labels, and metadata needed
to reproduce method calls on the article scenarios.


## Load A Scenario

```python
from jasa_mcmcis import load_scenario

scenario = load_scenario("gwas_additive_score_near_v01_n140")
problem = scenario.problem

print(problem.t_obs)
print(scenario.exact_p_value)
```

The bundled keys are exactly the 160 near-threshold scenarios run in the
article's cross-method comparison grid: 80 GWAS-like scenarios (`n=140`) and
80 HEP-like Poisson-count scenarios (`n=200`), all with a true p-value between
75% and 99% of the family's known significance threshold.

```python
from jasa_mcmcis import (
    CROSS_METHOD_SCENARIO_KEYS,
    GWAS_NEAR_THRESHOLD_SCENARIO_KEYS,
    HEP_NEAR_THRESHOLD_SCENARIO_KEYS,
)

print(len(CROSS_METHOD_SCENARIO_KEYS))  # 160
print(GWAS_NEAR_THRESHOLD_SCENARIO_KEYS[:3])
print(HEP_NEAR_THRESHOLD_SCENARIO_KEYS[:3])
```

## Run MCMC-IS

```python
from jasa_mcmcis import (
    estimate_scale_T,
    iid_pilot_statistics,
    init_beta_from_iid_pilot,
    load_scenario,
    run_mcmc_is,
)

scenario = load_scenario("poisson_diffmeans_hep_near_v01_n200")
problem = scenario.problem

pilot_T = iid_pilot_statistics(problem, n_samples=20_000, seed=1)
sigma_t = estimate_scale_T(pilot_T)

beta = init_beta_from_iid_pilot(
    pilot_T,
    problem.t_obs,
    sigma_t,
    p0_reference=scenario.exact_p_value,
    gamma=1 / 3,
)

result = run_mcmc_is(
    problem,
    beta=beta,
    sigma_t=sigma_t,
    n_steps=50_000,
    burn_in=10_000,
    n_chains=2,
    proposal_size=0.05,
    seed=123,
)

print(result.estimate, result.mcse_obm, result.ess)
```

The tilt strength is set by the target tilted tail mass `q`. The article rule is
`q = p0 ** gamma` with `gamma = 1/3`; pass `gamma` directly, or pass `q_target`
to control `q` yourself. Smaller `gamma` tilts harder.

The hard-step variant uses `pi_r(y) ∝ f(y) * {1 + (r - 1) 1_A(y)}`. If
`f(A)=p0`, then `r = q(1 - p0) / (p0(1 - q))` targets `pi_r(A)=q`:

```python
from jasa_mcmcis import run_hard_step_mcmc_is

hard_step = run_hard_step_mcmc_is(
    problem,
    p0=scenario.extra["known_significance_threshold"],
    gamma=1 / 3,
    n_steps=50_000,
    burn_in=10_000,
    n_chains=2,
    proposal_size=0.05,
    seed=123,
)
```

`proposal_size` is a fraction of the smaller group size (an integer pins an
exact swap-pair count instead). The cross-method grid swept
`proposal_size in {0.025, 0.05, 0.1}` for MCMC-IS and `{0.05, 0.1}` for SAMC.

## Run SAMC

This is the non-adaptive scheme used for the article comparison: a fixed uniform
partition of `[lambda_0, t_obs)` plus the tail bin `[t_obs, inf)`, gain
`gamma_t = t0 / max(t0, t)`, and the Yu et al. (2011) Eq. (3.2) estimate with the
`pi0` empty-bin correction. There is no discarded burn-in — every iteration
updates `theta` and counts towards the visitation diagnostic.

`lambda_0` is the smallest statistic in `lambda_min_samples` iid permutations.
Those draws are reported as `lambda_min_evaluations` so they can be charged to
the sampling budget; pass `lambda_min` instead to pin it and skip the pilot.

```python
from jasa_mcmcis import load_scenario, run_samc

scenario = load_scenario("gwas_additive_score_near_v01_n140")

samc = run_samc(
    scenario.problem,
    n_steps=200_000,
    n_bins=101,
    lambda_min_samples=10_000,
    proposal_size=0.05,
    seed=123,
)

print(samc.estimate, samc.max_abs_relative_frequency_error)
print(samc.n_steps + samc.lambda_min_evaluations)  # total statistic evaluations
```

Yu et al. recommend `m = 301` subregions for a continuous statistic. The bundled
scenarios use integer-valued statistics, where a fine uniform partition leaves
most subregions unreachable, so the default is `n_bins = 101`. Check
`samc.empty_bin_indices` when changing it.

## Exact DP Validation

The exact p-values are already stored in scenario metadata. The DP solver is
included because the bundled cross-method scenarios use linear statistics, and
exact checks are useful in a statistical-methods supplement.

```python
from jasa_mcmcis import LinearStatisticDPSolver, load_scenario

scenario = load_scenario("gwas_additive_score_near_v01_n140")
exact = LinearStatisticDPSolver.from_scenario(scenario).compute()

print(exact.p_value)
print(scenario.exact_p_value)
```

This module is not required for running MCMC-IS or SAMC. It is a validation and
reproducibility aid, not part of the data-generation pipeline.

## Reproducing The Budget-Convergence Figures

`run_mcmc_is` and `run_samc` each run one fresh chain to a fixed budget. The
article's budget-convergence figures instead track a single chain across 20
checkpoints (step counts 250,000 to 5,000,000 in steps of 250,000) and, at
each checkpoint, treat a fraction of the steps taken so far as burn-in before
computing the estimate. That checkpoint sweep is orchestration code, not part
of either estimator, and is not included here. To reproduce a budget curve
from a single call to `run_mcmc_is` or `run_samc`, run one chain per
checkpoint at `n_steps` equal to the checkpoint value, with `burn_in` set to
`0.20 * n_steps` (the burn-in fraction used throughout the article).

## API Summary

- `PermutationTestProblem(x, y_obs, statistic, tail="right")`: fixed-size
  permutation-test problem.
- `run_mcmc_is(problem, beta, sigma_t, n_steps, ...)`: tilted MCMC-IS estimator.
- `run_hard_step_mcmc_is(problem, p0, q=..., gamma=..., n_steps, ...)`: hard-step
  MCMC-IS estimator with `r` set by the reference threshold and target tail mass.
- `run_samc(problem, n_steps, ...)`: SAMC estimator for right-tail tests.
- `iid_pilot_statistics`, `estimate_scale_T`, `init_beta_from_iid_pilot`,
  `q_target_from_gamma`: beta initialization helpers.
- `estimate_lambda_min(problem, n_samples, rng)`: the SAMC `lambda_0` pilot.
- `load_scenario`, `load_scenarios`, `available_scenarios`: access the bundled
  cross-method grid scenarios.
- `LinearStatisticDPSolver`: optional exact p-value validation for linear
  statistics.

## Citation And License

Released under the MIT License (see `LICENSE`). If you use this code, please
cite the accompanying article. Results in the article were produced with the
seeds shown in the examples on `numpy>=1.24`; estimates are reproducible for a
fixed seed, numpy version, and platform.
