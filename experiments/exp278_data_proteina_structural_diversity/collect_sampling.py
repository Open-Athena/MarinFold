"""Collect sampling timings with identifiers that join to refolding candidates."""

import csv
import hashlib
import io
import json
from pathlib import Path

from analyze_screen import write_csv
from launch import storage_filesystem


def main() -> None:
    fs = storage_filesystem("cw-rno2a")
    root = "s3://marin-us-east-02a/MarinFold/exp278-proteina"
    timings, assets = [], []
    for model in ["short", "long"]:
        cases = json.loads(
            fs.cat(f"{root}/screen-{model}-v1/cases.json".removeprefix("s3://"))
        )
        for case in cases:
            prefix = case["output"]
            path = prefix.removeprefix("s3://")
            if not fs.exists(path + "/complete.json"):
                continue
            rows = list(
                csv.DictReader(io.StringIO(fs.cat(path + "/timings.csv").decode()))
            )
            for row in rows:
                batch_name = f"batch-{int(row['batch_index']):05d}.npz"
                stem = (
                    "proteina-"
                    + hashlib.sha256(
                        f"{prefix}/{batch_name}/{row['sample_in_batch']}".encode()
                    ).hexdigest()[:20]
                )
                timings.append(
                    {
                        **row,
                        "stem": stem,
                        "source_prefix": prefix,
                        "source_batch": batch_name,
                        "warm_batch": int(row["batch_index"]) > 0,
                        "known_cache_fallback": prefix.endswith(
                            "short-l100-unconditional"
                        ),
                    }
                )
            assets.append(
                {
                    "source_prefix": prefix,
                    "assets": json.loads(fs.cat(path + "/assets.json")),
                }
            )
    write_csv(Path("data/sampling-timings.csv"), timings)
    Path("data/sampling-assets.json").write_text(json.dumps(assets, indent=2) + "\n")
    print(f"Collected timings for {len(timings)} candidates")


if __name__ == "__main__":
    main()
