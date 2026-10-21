# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild and publish the small September slide artifacts to the public bucket."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESTINATION = (
    "hf://buckets/open-athena/MarinFold/data/exp180-progress-over-time/2026-09-30"
)


def main() -> None:
    """Regenerate and upload only the new slides and their plotted source tables."""
    subprocess.run([sys.executable, str(HERE / "plot_update_slide.py")], check=True)
    files = sorted((HERE / "plots").glob("*2026-09-30.*"))
    files += sorted((HERE / "data").glob("progress_slide_*.csv"))
    files += [HERE / "plot_update_slide.py", HERE / "publish_update_slide.py"]
    if not files:
        raise ValueError("No slide artifacts to publish")
    with tempfile.TemporaryDirectory(prefix="exp180-slide-") as staging:
        for source in files:
            target = Path(staging) / source.relative_to(HERE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        subprocess.run(
            ["hf", "buckets", "sync", staging, DESTINATION, "--no-delete"], check=True
        )


if __name__ == "__main__":
    main()
