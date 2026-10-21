"""Validate the frozen two-chain structures with Helico's own parser."""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    """Check chain identity and sequence-coordinate agreement for every target."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path, default=Path("/data/exp350_foldbench_eval/bundle")
    )
    parser.add_argument("--helico-repo", type=Path, default=Path("/home/bizon/git/helico"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.helico_repo / "src"))
    from helico.bench import structure_to_chains
    from helico.data import parse_mmcif

    rows = [
        json.loads(line)
        for line in (
            args.bundle / "data/foldbench_complex_gt_universe.jsonl"
        ).read_text().splitlines()
    ]
    for row in rows:
        path = (
            args.bundle
            / "examples/ground_truths"
            / f"{row['target_id']}.cif"
        )
        structure = parse_mmcif(path, max_resolution=float("inf"))
        chains = [
            chain
            for chain in structure_to_chains(structure)
            if chain["type"] == "protein"
        ]
        observed = {chain["id"]: chain["sequence"] for chain in chains}
        if len(chains) != 2 or set(observed) != set(row["chain_ids"]):
            raise ValueError(
                f"{row['target_id']}: expected {row['chain_ids']}, "
                f"observed {sorted(observed)}"
            )
        for chain_index, chain_id in enumerate(row["chain_ids"]):
            offset = row["chain_offsets"][chain_index]
            expected = "".join(
                row["chain_sequences"][chain_index][position - offset]
                for position in row["structure_positions_by_chain"][chain_index]
            )
            actual = observed[chain_id]
            compatible = len(actual) == len(expected) and all(
                left == right or left == "X" or right == "X"
                for left, right in zip(actual, expected, strict=True)
            )
            if not compatible:
                raise ValueError(
                    f"{row['target_id']} {chain_id}: Helico sequence differs "
                    "from frozen coordinate map"
                )
    print(f"validated {len(rows)} filtered two-chain Helico references")


if __name__ == "__main__":
    main()
