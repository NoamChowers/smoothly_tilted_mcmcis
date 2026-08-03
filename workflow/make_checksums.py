#!/usr/bin/env python3
"""Write SHA-256 checksums for the immutable reproducibility inputs."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "checksums.sha256"
EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__", "jasa_mcmcis.egg-info", ".venv"}
EXCLUDED_PREFIXES = {Path("output/reproduced"), Path("output/obm_smoke")}
EXCLUDED_FILES = {Path("output/simulation_smoke.jsonl"), Path("checksums.sha256")}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if relative in EXCLUDED_FILES:
        return False
    return not any(relative == prefix or prefix in relative.parents for prefix in EXCLUDED_PREFIXES)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and included(path))
    lines = [f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in files]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} checksums to {OUTPUT}")


if __name__ == "__main__":
    main()
