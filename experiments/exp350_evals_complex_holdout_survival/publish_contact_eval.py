"""Publish the context-complete contact subset to the public HF bucket."""

import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MANIFEST = DATA / "foldbench_complex_contact_eval_manifest.json"
HF = Path.home() / ".local/bin/hf"


def main() -> None:
    """Upload each manifest-listed derived contact-evaluation artifact."""
    manifest = json.loads(MANIFEST.read_text())
    prefix = manifest["public_prefix"]
    for name in manifest["files"]:
        if name == "foldbench_complex_eval_targets.parquet":
            continue
        source = DATA / name
        destination = f"{prefix}/{name}"
        print(f"{source} -> {destination}", flush=True)
        subprocess.run(
            [str(HF), "buckets", "cp", str(source), destination],
            check=True,
        )
    subprocess.run(
        [str(HF), "buckets", "cp", str(MANIFEST), f"{prefix}/{MANIFEST.name}"],
        check=True,
    )


if __name__ == "__main__":
    main()
