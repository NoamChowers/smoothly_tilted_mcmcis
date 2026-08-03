#!/usr/bin/env python3
"""Regenerate every bundled scenario from its documented seed and metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from jasa_mcmcis import available_scenarios, load_scenario


ROOT = Path(__file__).resolve().parents[1]


def near_extreme_labels(x: np.ndarray, n_treated: int, downgrade_swaps: int) -> np.ndarray:
    order = np.argsort(np.asarray(x, dtype=float), kind="mergesort")
    labels = np.zeros(x.size, dtype=np.int8)
    treated = order[-n_treated:]
    labels[treated] = 1
    controls = order[:-n_treated]
    treated = treated[np.argsort(x[treated], kind="mergesort")]
    controls = controls[np.argsort(x[controls], kind="mergesort")[::-1]]
    used: set[int] = set()
    completed = 0
    for treated_index in treated.tolist():
        for control_index in controls.tolist():
            if control_index not in used and x[treated_index] > x[control_index]:
                labels[treated_index] = 0
                labels[control_index] = 1
                used.add(control_index)
                completed += 1
                break
        if completed == downgrade_swaps:
            return labels
    raise RuntimeError("Could not reconstruct the requested downgraded labeling.")


def regenerate(metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    extra = metadata["extra"]
    key = metadata["key"]
    rng = np.random.default_rng(int(extra["seed"]))
    if key.startswith("gwas_additive_score"):
        x = rng.binomial(2, float(extra["maf"]), size=int(metadata["n"])).astype(float)
        y = near_extreme_labels(x, int(metadata["n_treated"]), int(extra["downgrade_swaps"]))
        return x, y
    if key.startswith("poisson_diffmeans_hep"):
        low = rng.poisson(float(extra["lambda_low"]), size=int(extra["n_pois2"]))
        high = rng.poisson(float(extra["lambda_high"]), size=int(extra["n_pois3"]))
        x = np.concatenate([low, high]).astype(float)
        y = np.asarray([0] * len(low) + [1] * len(high), dtype=np.int8)
        return x, y
    raise ValueError(f"Unsupported scenario family: {key}")


def array_digest(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def iter_wide_scenarios():
    for band in ("significant", "non_significant"):
        root = ROOT / "data" / "wide_ratio" / band
        for directory in sorted(path for path in root.iterdir() if path.is_dir()):
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            yield directory, metadata, np.load(directory / "X.npy"), np.load(directory / "y_obs.npy")


def verify(output: Path | None = None) -> dict:
    rows = []
    for key in available_scenarios():
        scenario = load_scenario(key)
        metadata = {"key": key, "n": scenario.problem.n, "n_treated": scenario.problem.n_treated, "extra": scenario.extra}
        x, y = regenerate(metadata)
        if not np.array_equal(x, scenario.problem.x) or not np.array_equal(y, scenario.problem.y_obs):
            raise RuntimeError(f"Regeneration mismatch for {key}.")
        rows.append({"scenario": key, "x_sha256": array_digest(x), "y_sha256": array_digest(y)})
    for directory, metadata, expected_x, expected_y in iter_wide_scenarios():
        x, y = regenerate(metadata)
        if not np.array_equal(x, expected_x) or not np.array_equal(y, expected_y):
            raise RuntimeError(f"Regeneration mismatch for {metadata['key']}.")
        rows.append({"scenario": metadata["key"], "x_sha256": array_digest(x), "y_sha256": array_digest(y)})
        if output is not None:
            destination = output / metadata["extra"]["threshold_band"] / metadata["key"]
            destination.mkdir(parents=True, exist_ok=True)
            np.save(destination / "X.npy", x)
            np.save(destination / "y_obs.npy", y)
    result = {"status": "complete", "n_scenarios": len(rows), "scenarios": rows}
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "regeneration_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None, help="Optionally write regenerated wide-ratio arrays here.")
    args = parser.parse_args()
    result = verify(args.output_dir)
    print(json.dumps({"status": result["status"], "n_scenarios": result["n_scenarios"]}, indent=2))


if __name__ == "__main__":
    main()
