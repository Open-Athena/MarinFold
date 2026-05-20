"""Audit the MSA Volume: for each stem in manifest.csv, verify
<stem>/msa/0/0/non_pairing.a3m exists (i.e. the colabfold extraction
actually completed). Print which ones are missing or have only a
partial layout."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modal_app


def main() -> None:
    manifest = list(csv.DictReader(Path("inputs/manifest.csv").open()))
    stems = [r["stem"] for r in manifest]
    print(f"checking MSA presence for {len(stems)} proteins...")

    with modal_app.app.run():
        results = list(modal_app.audit_msa.starmap([(s,) for s in stems]))

    missing = [r for r in results if not r["non_pairing_exists"]]
    print(f"complete: {len(results) - len(missing)} / {len(results)}")
    for r in missing:
        print(f"  MISSING non_pairing.a3m: {r['stem']} "
              f"(paired={r['paired_exists']}, dir_exists={r['dir_exists']})")


if __name__ == "__main__":
    main()
