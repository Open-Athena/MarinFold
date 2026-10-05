"""Publish the committed small result artifacts; model weights remain co-located."""

import subprocess
from pathlib import Path


def main() -> None:
    here = Path(__file__).parent
    destination = "hf://buckets/open-athena/MarinFold/data/exp347-qwen-base-contacts"
    for directory in [
        "data/e8_reference",
        "data/eval_val_pilot",
        "data/scale_smoke_timings",
        "data/full_phase_initial_timings",
    ]:
        subprocess.run(
            [
                "hf",
                "buckets",
                "sync",
                str(here / directory),
                f"{destination}/{directory}",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
