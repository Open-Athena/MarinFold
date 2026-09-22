#!/usr/bin/env python
"""Hash all bulk artifacts before publishing them to the public bucket."""

import hashlib
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOTS = {
    "blind-search-v1": HERE / "_cache" / "raw",
    "iid500": HERE / "_cache" / "iid500",
    "helico-inputs": HERE / "_cache" / "helico" / "data",
    "helico-results": HERE / "_cache" / "helico" / "results",
    "joblogs": HERE / "_cache" / "joblogs",
}


def main() -> None:
    rows = []
    for group, root in ROOTS.items():
        if not root.exists():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            rows.append({"group": group, "path": str(path.relative_to(root)),
                         "bytes": path.stat().st_size, "sha256": digest})
    result = pd.DataFrame(rows).sort_values(["group", "path"])
    result.to_csv(HERE / "data" / "artifact_manifest.csv", index=False)
    print(result.groupby("group").agg(files=("path", "size"), bytes=("bytes", "sum")))


if __name__ == "__main__":
    main()
