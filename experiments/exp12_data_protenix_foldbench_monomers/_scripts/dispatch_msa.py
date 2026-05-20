"""One-off: pre-compute MSAs for the 10-protein test set.

Reads inputs/manifest.csv, dispatches precompute_msa across all rows
via the Modal app. Idempotent (precompute_msa skips proteins whose
MSAs already exist on the Volume), so safe to re-run.

Invoke from the experiment dir::

    .venv/bin/python _scripts/dispatch_msa.py
"""

import csv
import json
import sys
from pathlib import Path

# Make the experiment dir importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modal_app


def main() -> None:
    inputs = Path("inputs")
    manifest = list(csv.DictReader((inputs / "manifest.csv").open()))
    args = []
    for row in manifest:
        job = json.loads((inputs / row["job_json"]).read_text())
        seq = job[0]["sequences"][0]["proteinChain"]["sequence"]
        args.append((row["stem"], seq))

    print(f"dispatching precompute_msa for {len(args)} proteins...")
    with modal_app.app.run():
        results = list(modal_app.precompute_msa.starmap(args))
    n_ran = sum(1 for r in results if not r.get("skipped") and not r.get("failed"))
    n_skipped = sum(1 for r in results if r.get("skipped"))
    n_failed = sum(1 for r in results if r.get("failed"))
    print(f"done: {n_ran} ran, {n_skipped} skipped, {n_failed} failed")
    for r in results:
        status = "skipped" if r.get("skipped") else ("FAILED" if r.get("failed") else "ran")
        print(f"  {r['stem']}: {status}")
        if r.get("failed"):
            print(f"    error: {r.get('error')}")


if __name__ == "__main__":
    main()
