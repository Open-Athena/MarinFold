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
    n_ran = sum(1 for r in results if not r.get("skipped"))
    n_skipped = sum(1 for r in results if r.get("skipped"))
    print(f"done: {n_ran} ran, {n_skipped} skipped")
    for r in results:
        print(f"  {r['stem']}: {'skipped' if r.get('skipped') else 'ran'}")


if __name__ == "__main__":
    main()
