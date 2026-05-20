"""Audit MSA presence for a specific stems list (default: 52 missing)."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modal_app  # noqa: E402


def main() -> None:
    stems_file = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/missing_stems.txt")
    stems = [s.strip() for s in stems_file.read_text().splitlines() if s.strip()]
    print(f"auditing {len(stems)} stems from {stems_file}")

    with modal_app.app.run():
        results = list(modal_app.audit_msa.starmap([(s,) for s in stems]))

    missing = [r for r in results if not r["non_pairing_exists"]]
    print(f"complete: {len(results) - len(missing)} / {len(results)}")
    for r in missing:
        print(f"  MISSING: {r['stem']} (paired={r['paired_exists']}, dir_exists={r['dir_exists']})")


if __name__ == "__main__":
    main()
