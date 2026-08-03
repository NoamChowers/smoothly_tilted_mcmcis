from __future__ import annotations

from typing import Sequence

BASE_SEED = 91337
FAMILIES: tuple[str, ...] = ("gwas", "hep")
SCENARIOS_PER_FAMILY = 80

MCMCIS_SWAP_FRACTIONS: tuple[float, ...] = (0.025, 0.05, 0.1)
SAMC_SWAP_FRACTIONS: tuple[float, ...] = (0.05, 0.1)
GAMMAS: tuple[float, ...] = (0.25, 1.0 / 3.0, 0.4, 0.5)
TILT_MODE = "smooth_hinge"

CHECKPOINTS: tuple[int, ...] = tuple(250_000 * i for i in range(1, 21))
MAX_BUDGET = CHECKPOINTS[-1]

N_BINS = 101
T0 = 1_000.0
CONVERGENCE_TOLERANCE = 20.0
BURN_IN_FRACTION = 0.20
MCMC_CHAINS = 1
MCMC_ESTIMATE_VARIANCE = False

BETA_INIT_PILOT_SAMPLES = 100_000
SAMC_LAMBDA_MIN_PILOT = 10_000

# The seed passed to a chain is `job_seed + offset`, where `job_seed` comes from
# `job_seed()` below. Beta calibration and the SAMC lambda_0 pilot reuse the
# unshifted job seed; the production chain gets a distinct offset so the two
# stages draw from independent streams.
MCMC_IS_BETA_PILOT_SEED_OFFSET = 0
MCMC_IS_CHAIN_SEED_OFFSET = 1
SAMC_SETUP_SEED_OFFSET = 0
SAMC_CHAIN_SEED_OFFSET = 3


def _method_blocks() -> tuple[tuple[str, float, str, float | None], ...]:
    """
    The (family, swap_fraction, method, gamma) grid, in the exact order the
    article's runs enumerated them. Order matters: each block's seed is offset
    by its position in this sequence.
    """
    blocks: list[tuple[str, float, str, float | None]] = []
    for family in FAMILIES:
        for swap_fraction in MCMCIS_SWAP_FRACTIONS:
            if swap_fraction in SAMC_SWAP_FRACTIONS:
                blocks.append((family, swap_fraction, "samc", None))
            for gamma in GAMMAS:
                blocks.append((family, swap_fraction, "mcmc_is", gamma))
    return tuple(blocks)


METHOD_BLOCKS: tuple[tuple[str, float, str, float | None], ...] = _method_blocks()


def method_block_index(
    family: str,
    swap_fraction: float,
    method: str,
    gamma: float | None = None,
) -> int:
    """Position of a (family, swap_fraction, method, gamma) block in the grid."""
    for index, (block_family, block_swap, block_method, block_gamma) in enumerate(METHOD_BLOCKS):
        same_gamma = (
            block_gamma is None and gamma is None
        ) or (
            block_gamma is not None and gamma is not None and abs(block_gamma - float(gamma)) < 1e-9
        )
        if (
            block_family == family
            and abs(block_swap - float(swap_fraction)) < 1e-9
            and block_method == method
            and same_gamma
        ):
            return index
    raise ValueError(f"No grid block matches family={family!r}, swap_fraction={swap_fraction!r}, "
                      f"method={method!r}, gamma={gamma!r}.")


def job_seed(
    family: str,
    swap_fraction: float,
    method: str,
    scenario_index: int,
    gamma: float | None = None,
    base_seed: int = BASE_SEED,
) -> int:
    """
    The base seed for one (family, swap_fraction, method, gamma, scenario) run.

    ``scenario_index`` is the 0-based position of the scenario within its
    family's near-threshold list (``GWAS_NEAR_THRESHOLD_SCENARIO_KEYS`` or
    ``HEP_NEAR_THRESHOLD_SCENARIO_KEYS``). Add ``MCMC_IS_CHAIN_SEED_OFFSET`` or
    ``SAMC_CHAIN_SEED_OFFSET`` to get the seed actually passed to
    ``run_mcmc_is_checkpoints``/``run_samc_checkpoints``; the unshifted value is
    the seed for that method's pilot stage.
    """
    block_index = method_block_index(family, swap_fraction, method, gamma)
    return int(base_seed) + 1_000_000 * block_index + 10_000 * int(scenario_index)


def mcmc_is_chain_checkpoints(checkpoints: Sequence[int] = CHECKPOINTS) -> tuple[int, ...]:
    """
    Reported budgets converted to chain step counts for ``run_mcmc_is_checkpoints``.

    Each reported budget includes the beta-calibration pilot
    (``BETA_INIT_PILOT_SAMPLES``); the chain itself runs for the remainder.
    """
    return tuple(int(cp) - BETA_INIT_PILOT_SAMPLES for cp in checkpoints if int(cp) - BETA_INIT_PILOT_SAMPLES > 0)


def samc_chain_checkpoints(checkpoints: Sequence[int] = CHECKPOINTS) -> tuple[int, ...]:
    """
    Reported budgets converted to chain step counts for ``run_samc_checkpoints``.

    Each reported budget includes the SAMC lambda_0 pilot
    (``SAMC_LAMBDA_MIN_PILOT``); the chain itself runs for the remainder.
    """
    return tuple(int(cp) - SAMC_LAMBDA_MIN_PILOT for cp in checkpoints if int(cp) - SAMC_LAMBDA_MIN_PILOT > 0)
