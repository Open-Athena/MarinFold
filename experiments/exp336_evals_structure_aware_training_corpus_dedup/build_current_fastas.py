# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Filter #213's FASTAs to the exact decontaminated #232 training universe.

#213 already paid the 146 GB document scan needed to decode every training
sequence. #225 then removed 1,373,423 rows and published the filtered corpora
used by #232. This script applies #225's row-key drop list to #213's pinned
FASTAs, avoiding another public-bucket scan while asserting the exact #232
document counts.
"""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

EXPECTED = {"afdb": 3_963_003, "esm_atlas": 65_553_178}


def parse_header(header: str) -> tuple[str, str]:
    """Return ``(arm, entry_id)`` from #213's stable FASTA header grammar."""
    key = header.removeprefix(">").strip().split(maxsplit=1)[0]
    try:
        arm, local = key.split("|", 1)
        _, _, entry_id = local.split("_", 2)
    except ValueError as error:
        raise ValueError(f"invalid #213 FASTA header {header!r}") from error
    if arm not in EXPECTED or not entry_id:
        raise ValueError(f"invalid #213 FASTA header {header!r}")
    return arm, entry_id


def fasta_records(path: Path) -> Iterator[tuple[str, list[str]]]:
    """Stream FASTA records without joining or rewrapping their sequences."""
    header: str | None = None
    sequence_lines: list[str] = []
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                if header is not None:
                    yield header, sequence_lines
                header = line
                sequence_lines = []
            elif header is None:
                raise ValueError(f"{path}: sequence content precedes the first header")
            else:
                sequence_lines.append(line)
    if header is not None:
        yield header, sequence_lines


def load_drops(path: Path) -> dict[str, set[str]]:
    """Read the final #225 row-key drop list once."""
    table = pq.read_table(path, columns=["arm", "entry_id"])
    drops = {arm: set() for arm in EXPECTED}
    for arm, entry_id in zip(table["arm"].to_pylist(), table["entry_id"].to_pylist(), strict=True):
        if arm not in drops:
            raise ValueError(f"unknown drop-list arm {arm!r}")
        drops[arm].add(entry_id)
    if sum(map(len, drops.values())) != table.num_rows:
        raise ValueError("drop list contains duplicate (arm, entry_id) rows")
    return drops


def filter_fasta(source: Path, destination: Path, arm: str, drops: set[str]) -> dict:
    """Write all and only rows that survive #225's final drop list."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    seen: set[str] = set()
    kept = removed = 0
    with temporary.open("w") as output:
        for header, sequence_lines in fasta_records(source):
            record_arm, entry_id = parse_header(header)
            if record_arm != arm:
                raise ValueError(f"{source}: expected arm {arm!r}, found {record_arm!r}")
            if entry_id in seen:
                raise ValueError(f"{source}: duplicate entry ID {entry_id!r}")
            seen.add(entry_id)
            if entry_id in drops:
                removed += 1
                continue
            output.write(header)
            output.writelines(sequence_lines)
            kept += 1
    missing_drops = drops - seen
    if missing_drops:
        examples = sorted(missing_drops)[:5]
        raise ValueError(f"{source}: {len(missing_drops):,} drop rows absent; examples={examples}")
    if kept != EXPECTED[arm]:
        raise ValueError(f"{arm}: kept {kept:,}, expected #232's {EXPECTED[arm]:,}")
    temporary.replace(destination)
    return {
        "arm": arm,
        "source": str(source),
        "destination": str(destination),
        "source_records": len(seen),
        "removed": removed,
        "kept": kept,
        "expected_kept": EXPECTED[arm],
        "destination_bytes": destination.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp213-work", type=Path, default=Path("/data/exp213_overlap"))
    parser.add_argument(
        "--droplist",
        type=Path,
        default=Path("/data/exp225_decontam/droplist_final.parquet"),
    )
    parser.add_argument("--work", type=Path, default=Path("/data/exp336_dedup"))
    parser.add_argument("--arm", choices=[*EXPECTED, "both"], default="both")
    args = parser.parse_args()
    drops = load_drops(args.droplist)
    arms = EXPECTED if args.arm == "both" else (args.arm,)
    rows = []
    for arm in arms:
        source = args.exp213_work / f"train_{arm}.fasta"
        destination = args.work / f"current_{arm}.fasta"
        rows.append(filter_fasta(source, destination, arm, drops[arm]))
        print(json.dumps(rows[-1], indent=2), flush=True)
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "current_fastas.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
