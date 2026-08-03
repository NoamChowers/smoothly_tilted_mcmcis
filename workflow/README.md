# Workflow guide

The master post-processing entry point is `reproduce.py`. It validates the
archived run-level inventory, selects configurations using the article rule,
recomputes central estimates and diagnostics, applies the archived article
bootstrap bounds, and writes a SHA-256 output manifest.

`regenerate_data.py` regenerates all scenario arrays from metadata and checks
them against the copies used in the simulations. `run_simulations.py` offers a
short fresh-chain smoke mode and the complete cross-method plus wide-ratio
design. `run_obm.py` does the same for the repeated-chain OBM calibration.

The run-level records are included because a full rerun requires billions of
statistic evaluations and is not practical during routine review. A reviewer
can therefore use this progression:

1. `make all` for data verification, tests, and all article outputs.
2. `make smoke` to exercise fresh MCMC-IS, SAMC, and OBM chains.
3. `make full` and `make full-obm` only when access to long-running compute is
   available.

The original production design used six CPU cores on one machine. No GPU or
network access is required. Each block and scenario is seeded explicitly;
parallel scheduling does not alter its random stream.

Generated outputs go under `output/`. `output/reference/` is immutable
reference material; `output/reproduced/`, `output/simulation_smoke.jsonl`, and
`output/obm_smoke/` are disposable products.
