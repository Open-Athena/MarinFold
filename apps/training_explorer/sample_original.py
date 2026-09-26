"""Uniformly sample the exp199 corpus through its complete local MMseqs index."""

import argparse
import json
import random
from pathlib import Path


POPULATION = 70_889_604
DEFAULT_DB = Path("/data/exp213_overlap/targetDB")
OUTPUT = Path(__file__).resolve().parent / "data/original.json"


def selected_lines(path: Path, wanted: set[int]) -> dict[int, bytes]:
    """Scan one complete index and retain only requested record numbers."""
    found: dict[int, bytes] = {}
    with path.open("rb") as handle:
        for ordinal, line in enumerate(handle):
            if ordinal in wanted:
                found[ordinal] = line.rstrip(b"\n")
    if ordinal + 1 != POPULATION:
        raise ValueError(f"{path}: expected {POPULATION} entries, got {ordinal + 1}")
    if len(found) != len(wanted):
        raise ValueError(f"{path}: missing sampled entries")
    return found


def sample(db: Path, seed: int, count: int) -> dict:
    """Draw document ordinals uniformly without loading all 70.9M records."""
    rng = random.Random(seed)
    ordinals = set(rng.sample(range(POPULATION), count))
    offsets = selected_lines(db.with_suffix(".index"), ordinals)
    names = selected_lines(db.with_suffix(".lookup"), ordinals)
    proteins = []
    with db.open("rb") as source:
        for ordinal in sorted(ordinals):
            index_id, offset, length = offsets[ordinal].split(b"\t")
            lookup_id, header, _ = names[ordinal].split(b"\t", 2)
            if index_id != lookup_id or int(index_id) != ordinal:
                raise ValueError(f"MMseqs index and lookup disagree at {ordinal}")
            source.seek(int(offset))
            sequence = source.read(int(length)).decode("ascii").strip("\x00\n")
            target = header.decode("utf-8")
            arm, local = target.split("|", 1)
            entry_id = local.split("_", 2)[-1]
            proteins.append(
                {
                    "id": f"original:{ordinal}",
                    "label": entry_id,
                    "source": "AFDB" if arm == "afdb" else "ESM-Atlas",
                    "sequence": sequence,
                    "length": len(sequence),
                    "entryId": entry_id,
                    "targetId": target,
                    "corpusOrdinal": ordinal,
                    "structureUrl": None,
                    "structureFormat": "mmcif",
                    "neighbors": [],
                }
            )
    return {
        "title": "Original training set",
        "description": "100 uniform document draws from exp199's AFDB + ESM-Atlas corpus.",
        "population": POPULATION,
        "seed": seed,
        "sampling": "random.sample over the complete 70,889,604-row MMseqs index",
        "provenance": "/data/exp213_overlap/targetDB · exp213",
        "neighborCorpus": "original",
        "neighborsComplete": False,
        "proteins": proteins,
    }


def main() -> None:
    """Write the reproducible original-corpus snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(sample(args.db, args.seed, args.count), separators=(",", ":"))
    )


if __name__ == "__main__":
    main()
