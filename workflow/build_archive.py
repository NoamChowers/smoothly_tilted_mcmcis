#!/usr/bin/env python3
"""Build a deterministic JASA reproducibility-materials ZIP."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from make_checksums import ROOT, included


ARCHIVE = ROOT.parent / "reproducibility_materials.zip"


def main() -> None:
    checksum_manifest = ROOT / "checksums.sha256"
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and (included(path) or path == checksum_manifest)
    )
    with ZipFile(ARCHIVE, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = Path(ROOT.name) / path.relative_to(ROOT)
            info = ZipInfo(relative.as_posix(), date_time=(2026, 8, 3, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=ZIP_DEFLATED, compresslevel=9)
    print(f"Wrote {len(files)} files to {ARCHIVE} ({ARCHIVE.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
