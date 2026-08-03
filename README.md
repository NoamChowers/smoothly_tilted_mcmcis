# Reproducibility materials: Smoothly Tilted MCMC Importance Sampling

This is the self-contained computational supplement for *Smoothly Tilted MCMC
Importance Sampling for Small Tail Probabilities*. It implements MCMC
importance sampling (MCMC-IS), its hard-step variant, and stochastic
approximation Monte Carlo (SAMC); contains every simulated scenario and the
archived run-level records used by the article; and provides one-command
workflows for the article's computational tables and figures.

No confidential or third-party data are used. Every input is generated from a
documented pseudo-random seed and can be checked byte-for-byte with
`make verify-data`.

## Quick reproduction

Python 3.13 is the reference interpreter. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproducibility.txt
python -m pip install -e .
make all
```

`make all` verifies all 320 scenario data sets, runs the test suite, and writes
the computational tables and figures to `output/reproduced/`. The expected
runtime is 1--10 minutes on a standard desktop after installation. The checked
reference copies are in `output/reference/`.

To exercise both estimators on short fresh chains without using archived
records:

```bash
make smoke
```

## What each workflow does

| Command | Purpose | Expected scale |
|---|---|---|
| `make verify-data` | Rebuild and compare all scenario arrays from their seeds | <1 minute |
| `make test` | Unit and reproducibility tests | <1 minute |
| `make reproduce` | Recreate every article-facing computational table/figure from archived run-level records | 1--10 minutes |
| `make smoke` | Fresh short MCMC-IS, SAMC, and OBM end-to-end runs | <1 minute |
| `make full` | Re-run the complete near-threshold and wide-ratio simulation grid | >8 hours |
| `make full-obm` | Re-run 600 production chains for OBM calibration | >8 hours |
| `make archive` | Build the JASA submission ZIP in the parent directory | <1 minute |

The full simulations are computationally expensive because the design includes
160 near-threshold scenarios, 28 method/configuration blocks, 20 checkpoints up
to five million statistic evaluations, 160 additional wide-ratio scenarios,
and 600 independent OBM production chains. The submission therefore includes
run-level records as compressed JSON Lines so reviewers can reproduce all
reported results quickly while retaining scripts and seeds for a clean rerun.

## Article-output map

`workflow/reproduce.py` creates:

- `figure1_intro_tilt_geometry.pdf`
- `table_main_results.csv` and `.tex`
- `diagnostic_rrmse_by_budget_bca.pdf`
- `config_mean_rrmse_paired_scenario_bca.pdf`
- `top_mcmcis_configs_and_best_samc_rrmse_by_budget.pdf`
- `mcmcis_rrmse_by_pvalue_half_bca.pdf`
- `mcmcis_obm_relative_sd_distributions.pdf`
- `diagnostic_false_negative_rate_by_budget.pdf`
- `diagnostic_mean_phat_over_p_by_budget.pdf`
- `figure4_threshold_ratio_budget_5m.pdf`
- the scenario-bank and pilot/production-allocation tables

Bootstrap confidence bounds use the archived one-million-resample summaries
from the article run. Point estimates, configuration selection, diagnostic
curves, and table entries are recomputed from the included run-level records.
This avoids silently replacing the published bootstrap draw with a new random
realization.

## Package use

The installable Python package is under `src/jasa_mcmcis`. For example:

```python
from jasa_mcmcis import load_scenario, run_mcmc_is

scenario = load_scenario("gwas_additive_score_near_v01_n140")
result = run_mcmc_is(
    scenario.problem,
    beta=1.0,
    n_steps=10_000,
    burn_in=2_000,
    proposal_size=0.05,
    seed=123,
)
print(result.estimate)
```

See `workflow/README.md` for execution details, `data/README.md` and
`data/data_dictionary.csv` for input documentation, `ACC.md` for the completed
JASA Author Contributions Checklist, and `checksums.sha256` for integrity
checks. The code is licensed under the MIT License.

Verify the immutable snapshot files with:

```bash
shasum -a 256 -c checksums.sha256
```

## Repository and archival version

The standalone development and reproducibility repository is
<https://github.com/NoamChowers/smoothly_tilted_mcmcis>. For peer review, the
submission ZIP and its checksum manifest are the authoritative immutable
snapshot; record the final release tag or archive DOI here before publication.
