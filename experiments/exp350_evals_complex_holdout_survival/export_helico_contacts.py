"""Export rollout-ranked contacts into Helico's chain-addressed JSON format."""

import argparse
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq

from score_foldbench_contacts import load_votes

HERE = Path(__file__).resolve().parent
DEFAULT_TARGETS = HERE / "data/foldbench_complex_contact_eval_targets.parquet"
DEFAULT_BUDGETS = {"L5": 0.2, "L2": 0.5, "L": 1.0}


def mapped_rankings(
    target: dict, votes: dict[tuple[int, int], int]
) -> tuple[list[list], list[list]]:
    """Return ranked all-contact and intra-chain-only Helico contact lists."""
    resolved_by_chain = target["structure_positions_by_chain"]
    chain_ids = target["chain_ids"]
    index_map = {}
    chain_of = {}
    for chain_index, positions in enumerate(resolved_by_chain):
        for local_index, full_index in enumerate(positions):
            index_map[int(full_index)] = int(local_index)
            chain_of[int(full_index)] = chain_index
    ranked = []
    for pair, count in votes.items():
        left, right = pair
        if count <= 0 or left not in index_map or right not in index_map:
            continue
        left_chain = chain_of[left]
        right_chain = chain_of[right]
        if left_chain == right_chain and abs(index_map[left] - index_map[right]) < 6:
            continue
        mapped = [
            chain_ids[left_chain],
            index_map[left],
            chain_ids[right_chain],
            index_map[right],
        ]
        ranked.append((count, left, right, mapped, left_chain == right_chain))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    all_contacts = [item[3] for item in ranked]
    intra_contacts = [item[3] for item in ranked if item[4]]
    return all_contacts, intra_contacts


def main() -> None:
    """Write budgeted contact arms without using ground-truth contact counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--scores", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    targets = pq.read_table(args.targets).to_pylist()
    votes = load_votes(args.scores)
    expected = {target["stem"] for target in targets}
    if set(votes) != expected:
        raise ValueError(
            f"Vote/target mismatch: missing={sorted(expected - set(votes))}, "
            f"extra={sorted(set(votes) - expected)}"
        )
    arms: dict[str, dict[str, list]] = {
        f"marinfold_{scope}_{name}": {}
        for scope in ("all", "intra")
        for name in DEFAULT_BUDGETS
    }
    rows = []
    for target in targets:
        all_contacts, intra_contacts = mapped_rankings(
            target, votes[target["stem"]]
        )
        resolved_length = sum(map(len, target["structure_positions_by_chain"]))
        row = {
            "target_id": target["target_id"],
            "split": target["split"],
            "resolved_length": resolved_length,
            "available_all": len(all_contacts),
            "available_intra": len(intra_contacts),
        }
        for name, multiplier in DEFAULT_BUDGETS.items():
            budget = max(1, round(multiplier * resolved_length))
            for scope, contacts in (
                ("all", all_contacts),
                ("intra", intra_contacts),
            ):
                arm = f"marinfold_{scope}_{name}"
                selected = contacts[:budget]
                arms[arm][target["target_id"]] = selected
                row[f"{arm}_contacts"] = len(selected)
        rows.append(row)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for arm, mapping in arms.items():
        (args.out_dir / f"{arm}.json").write_text(
            json.dumps(mapping, indent=2) + "\n"
        )
    with (args.out_dir / "contact_arm_counts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(arms)} contact arms for {len(targets)} targets")


if __name__ == "__main__":
    main()
