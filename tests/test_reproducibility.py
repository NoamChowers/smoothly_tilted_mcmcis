from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflow"))

from regenerate_data import verify  # noqa: E402
from reproduce import (  # noqa: E402
    REFERENCE,
    checkpoint_metrics,
    main_table,
    read_jsonl_gz,
    select_configs,
    selected_records,
    validate_main_records,
)


def test_all_scenario_arrays_regenerate_exactly():
    result = verify()
    assert result["status"] == "complete"
    assert result["n_scenarios"] == 320


def test_article_inventory_selection_and_main_table():
    records = read_jsonl_gz(REFERENCE / "main_grid" / "all_records.jsonl.gz")
    inventory = validate_main_records(records)
    assert inventory["records"] == 44_800
    assert inventory["scenarios"] == 160
    assert inventory["configurations"] == 28

    selected = select_configs(records)
    assert selected[("gwas", "mcmc_is_no_oracle")][2:] == (0.05, "0.4")
    assert selected[("hep", "samc")][2:] == (0.05, "none")

    metrics = checkpoint_metrics(selected_records(records, selected))
    table = main_table(metrics)
    assert len(table) == 8
    first = table[0]
    assert first["mcmc_is_rrmse"] == pytest.approx(0.186, abs=0.001)
    assert first["samc_rrmse"] == pytest.approx(0.334, abs=0.001)
    assert first["rrmse_ratio"] == pytest.approx(0.558, abs=0.002)
