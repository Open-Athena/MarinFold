#!/usr/bin/env python
"""Package 29,087 saved Helico structures into auditable per-protein archives."""

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE / "_cache" / "helico_iid"
RESULTS = ROOT / "results"
PUBLISH = ROOT / "publish"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    """Create one tar per protein plus a member-level checksum index."""
    run = pd.read_csv(RESULTS / "iid1000.csv")
    if len(run) != 29_087 or not run.status.eq("ok").all():
        raise ValueError("iid1000 Helico results are incomplete")
    predictions = RESULTS / "predictions" / "iid1000"
    archive_dir = PUBLISH / "predictions"
    archive_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair_id, group in run.groupby(run.target_id.str.split("__").str[0]):
        paths = [predictions / f"{target_id}.pdb.gz" for target_id in sorted(group.target_id)]
        if len(paths) != 1003 or any(not path.exists() for path in paths):
            raise ValueError(f"{pair_id}: expected 1,003 prediction files")
        archive = archive_dir / f"{pair_id}.tar"
        with tarfile.open(archive, "w") as handle:
            for path in paths:
                handle.add(path, arcname=path.name, recursive=False)
                rows.append({
                    "target_id": path.name.removesuffix(".pdb.gz"),
                    "pair_id": pair_id, "archive": f"predictions/{archive.name}",
                    "member": path.name, "bytes": path.stat().st_size,
                    "sha256": digest(path),
                })
        print(f"packed {pair_id}: {archive.stat().st_size / 1e6:.1f} MB", flush=True)
    index = pd.DataFrame(rows).sort_values(["pair_id", "target_id"])
    index.to_parquet(PUBLISH / "prediction_index.parquet", index=False, compression="zstd")
    index.to_csv(PUBLISH / "prediction_index.csv.gz", index=False, compression="gzip")
    for name in ("iid1000.csv", "iid1000.manifest.json", "iid1000.progress.jsonl"):
        shutil.copy2(RESULTS / name, PUBLISH / name)
    shutil.copy2(ROOT / "scores.parquet", PUBLISH / "scores.parquet")
    smoke_dir = RESULTS / "predictions" / "smoke"
    if smoke_dir.exists():
        with tarfile.open(PUBLISH / "smoke_predictions.tar", "w") as handle:
            for path in sorted(smoke_dir.glob("*.pdb.gz")):
                handle.add(path, arcname=path.name, recursive=False)
        for path in RESULTS.glob("smoke.*"):
            if path.is_file():
                shutil.copy2(path, PUBLISH / path.name)
    archives = sorted(archive_dir.glob("*.tar"))
    manifest = {
        "n_targets": len(index), "n_proteins": index.pair_id.nunique(),
        "n_archives": len(archives), "structure_bytes": int(index.bytes.sum()),
        "archives": [{"path": f"predictions/{path.name}", "bytes": path.stat().st_size,
                      "sha256": digest(path)} for path in archives],
    }
    (PUBLISH / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "archives"}, indent=2))


if __name__ == "__main__":
    main()
