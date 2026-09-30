"""Rebuild and publish the report plus reproducibility inputs to the HF bucket."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUBLIC = "hf://buckets/open-athena/MarinFold/data/exp345-qualitative-contact-map-errors/v1"


def main() -> None:
    """Publish only the intended public artifacts, excluding private caches."""
    for script in ("analyze.py", "plot_results.py", "build_report.py", "build_summary.py"):
        subprocess.run([sys.executable, str(HERE / script)], cwd=HERE, check=True)
    paths = [HERE / "report.html", HERE / "README.md", *sorted((HERE / "data").glob("*")),
             *sorted((HERE / "plots").glob("*"))]
    for path in paths:
        relative = path.relative_to(HERE)
        # The input bundle is a first-class top-level download for analyze.py.
        target = path.name if path.name == "report_inputs.json.gz" else str(relative)
        subprocess.run(["hf", "buckets", "cp", str(path), PUBLIC + "/" + target], check=True)
    print(PUBLIC)


if __name__ == "__main__":
    main()
