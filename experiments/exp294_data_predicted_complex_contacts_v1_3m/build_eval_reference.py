# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble the eval2 decontamination reference: 577 = #225's 554 + #226's 23.

Both inputs are committed in this repo, so the reference is reproducible without
a bucket read:

* ``exp225/data/reference/eval_queries.fasta`` — the 554-protein v1 reference
  that defined the existing decontaminated corpora (#225, #232).
* ``exp226/data/eval2_manifest.csv`` — the expanded eval set; its 23
  ``foldbench_rest`` rows are the natural proteins #226 added under 40% identity.

#225's reference version ``v1`` means "the 554". This is a different protein
set, so it gets its own version rather than silently redefining v1: a drop list
stamped ``v1`` must keep meaning what the published corpora were filtered with.

    uv run python build_eval_reference.py --out data/eval2_reference.fasta
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
LEGACY_FASTA = (
    EXPERIMENTS
    / "exp225_data_decontaminate_training_corpora/data/reference/eval_queries.fasta"
)
EXPANSION_MANIFEST = (
    EXPERIMENTS / "exp226_evals_expand_foldbench_eval_set/data/eval2_manifest.csv"
)
EXPANSION_DATASET = "foldbench_rest"

REFERENCE_VERSION = "eval2-v1"
N_LEGACY = 554
N_EXPANSION = 23
N_REFERENCE = N_LEGACY + N_EXPANSION


def read_fasta(path: Path) -> dict[str, str]:
    """Parse a plain FASTA into ``{id: sequence}``, rejecting duplicate ids."""
    records: dict[str, str] = {}
    name: str | None = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            name = line[1:].split()[0]
            if name in records:
                raise ValueError(f"{path}: duplicate record id {name}")
            records[name] = ""
        elif name is None:
            raise ValueError(f"{path}: sequence before first header")
        else:
            records[name] += line
    return records


def build(out_path: Path) -> dict[str, Any]:
    """Write the union and assert its size, so a truncated input can't pass."""
    legacy = read_fasta(LEGACY_FASTA)
    if len(legacy) != N_LEGACY:
        raise ValueError(f"{LEGACY_FASTA}: expected {N_LEGACY} records, got {len(legacy)}")

    expansion: dict[str, str] = {}
    with EXPANSION_MANIFEST.open() as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] != EXPANSION_DATASET:
                continue
            key = f"{row['dataset']}__{row['stem']}"
            sequence = row["input_seq"].strip().upper()
            if not sequence:
                raise ValueError(f"{EXPANSION_MANIFEST}: {key} has no input_seq")
            expansion[key] = sequence
    if len(expansion) != N_EXPANSION:
        raise ValueError(
            f"{EXPANSION_MANIFEST}: expected {N_EXPANSION} {EXPANSION_DATASET} rows, "
            f"got {len(expansion)}"
        )

    collisions = sorted(set(legacy) & set(expansion))
    if collisions:
        raise ValueError(f"id collision between the two sources: {collisions}")

    merged = {**legacy, **expansion}
    if len(merged) != N_REFERENCE:
        raise ValueError(f"expected {N_REFERENCE} reference records, got {len(merged)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f">{key}\n{merged[key]}\n" for key in sorted(merged))
    out_path.write_text(body)

    provenance = {
        "reference_version": REFERENCE_VERSION,
        "records": len(merged),
        "legacy_source": str(LEGACY_FASTA.relative_to(EXPERIMENTS)),
        "legacy_records": len(legacy),
        "expansion_source": str(EXPANSION_MANIFEST.relative_to(EXPERIMENTS)),
        "expansion_dataset": EXPANSION_DATASET,
        "expansion_records": len(expansion),
        "residues": sum(len(v) for v in merged.values()),
        "sha256": hashlib.sha256(body.encode()).hexdigest(),
        "output": str(out_path.resolve()),
    }
    out_path.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "data/eval2_reference.fasta")
    return parser


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(build(build_parser().parse_args(argv).out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
