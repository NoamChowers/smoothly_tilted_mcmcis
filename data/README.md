# Data documentation

All data in this supplement are author-generated simulations. There are no
external, personal, restricted, or proprietary data.

`src/jasa_mcmcis/data/scenarios/` contains the 80 GWAS-like and 80 HEP-like
near-threshold scenarios used in the cross-method comparison. Each scenario has
`X.npy`, `y_obs.npy`, and `metadata.json`; the catalog supplies definitions,
exact permutation p-values, sample sizes, thresholds, and random seeds.

`data/wide_ratio/` contains 80 significant-side and 80 nonsignificant-side
GWAS-like scenarios used for the broad p/p0 analysis, in the same format.

`data/reference/main_grid/all_records.jsonl.gz` contains 44,800 checkpoint
records for the 160-scenario MCMC-IS/SAMC configuration grid.
`data/reference/wide_ratio/mcmcis_records.jsonl.gz` contains the selected
MCMC-IS configuration's wide-ratio checkpoint records. `data/reference/obm/`
contains scenario-level summaries from the 600-chain OBM experiment. CSV files
next to those records preserve the article's one-million-resample bootstrap
intervals and configuration-selection summaries.

Run `make verify-data` to reconstruct all 320 scenario arrays from their
metadata and verify exact equality. See `data_dictionary.csv` for fields and
formats. NumPy `.npy` is a standardized, non-executable array format; JSON,
JSON Lines, CSV, and gzip are open documented formats.
