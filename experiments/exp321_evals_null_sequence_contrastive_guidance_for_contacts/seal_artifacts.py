"""Hash full fold-switch prediction pools before reference-based scoring."""

import hashlib
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    """Hash one artifact without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Write a deterministic manifest for unscored full fold-switch pools."""
    paths = sorted((HERE / "_cache").glob("full_*/foldswitch/*.parquet"))
    paths = [path for path in paths if not path.name.endswith(".timing.parquet")]
    if not paths:
        raise FileNotFoundError("no full fold-switch artifacts")
    rows = [
        {
            "mode": path.parent.parent.name,
            "pair_id": path.stem,
            "relative_path": str(path.relative_to(HERE / "_cache")),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]
    frame = pd.DataFrame(rows).sort_values(["mode", "pair_id"])
    frame.to_csv(HERE / "data" / "sealed_foldswitch_artifacts.csv", index=False)
    print(f"sealed {len(frame)} fold-switch prediction pools")


if __name__ == "__main__":
    main()
