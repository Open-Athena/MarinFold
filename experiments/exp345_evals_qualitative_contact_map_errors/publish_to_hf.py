"""Rebuild and publish both reports and their anonymous reproduction artifacts."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfFileSystem

HERE = Path(__file__).resolve().parent
PUBLIC = "hf://buckets/open-athena/MarinFold/data/exp345-qualitative-contact-map-errors/v2"


def main() -> None:
    """Publish intended artifacts, then verify every object with anonymous reads."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-rebuild", action="store_true", help="Use already rebuilt and checked artifacts")
    args = parser.parse_args()
    if not args.skip_rebuild:
        for script in ("analyze.py", "download_whole_map.py", "analyze_whole_map.py", "package_whole_map.py", "summarize_whole_map.py",
                       "plot_results.py", "plot_whole_map.py", "build_report.py", "build_whole_map_report.py", "build_summary.py"):
            subprocess.run([sys.executable, str(HERE / script)], cwd=HERE, check=True)
    paths = [HERE / "report.html", HERE / "whole_map_report.html", HERE / "whole_map_raw.tar.gz", HERE / "README.md",
             *sorted((HERE / "data").glob("*")), *sorted((HERE / "plots").glob("*"))]
    stage = HERE / ".cache/public-v2"
    stage.mkdir(parents=True, exist_ok=True)
    expected = {}
    for path in paths:
        relative = path.name if path.name == "report_inputs.json.gz" else str(path.relative_to(HERE))
        dest = stage / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        expected[relative] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (stage / "artifact_manifest.json").write_text(json.dumps(expected, indent=2) + "\n")
    subprocess.run(["hf", "buckets", "sync", str(stage), PUBLIC], check=True)
    fs = HfFileSystem(token=False)
    for relative, entry in expected.items():
        raw = fs.cat_file(PUBLIC.removeprefix("hf://") + "/" + relative)
        assert len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"], relative
    print(f"Verified {len(expected)} public artifacts anonymously: {PUBLIC}")


if __name__ == "__main__":
    main()
