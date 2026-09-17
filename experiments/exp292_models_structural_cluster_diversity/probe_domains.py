"""Test manually chosen domain boundaries suggested by the ESM PAE-code plots.

These exploratory crops are not validated domain annotations or an automatic
production filter. A low whole-chain TM with high domain TM flags arrangement
differences that need interdomain-confidence review.
"""

import argparse
import json
from pathlib import Path

from structure_audit import Protein, compare, load_protein, write_csv


def crop(protein: Protein, start: int, stop: int) -> Protein:
    """Slice an explicit zero-based half-open residue range."""
    if not 0 <= start < stop <= len(protein.sequence):
        raise ValueError("Invalid domain crop")
    return Protein(
        protein.sequence[start:stop],
        protein.coords[start:stop],
        protein.plddt[start:stop],
    )


def main() -> None:
    """Measure and save each specified domain pair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for pair in json.loads(args.probes.read_text()):
        a = load_protein(args.cache, pair["candidate"])
        b = load_protein(args.cache, pair["anchor"])
        for bounds in pair["crops"]:
            a0, a1, b0, b1 = bounds
            results.append(
                {
                    "candidate": pair["candidate"],
                    "anchor": pair["anchor"],
                    "a_start": a0,
                    "a_stop": a1,
                    "b_start": b0,
                    "b_stop": b1,
                    **compare(crop(a, a0, a1), crop(b, b0, b1)),
                }
            )
    write_csv(args.output, results)


if __name__ == "__main__":
    main()
