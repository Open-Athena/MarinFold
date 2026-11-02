# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The exact format ceiling: generate a real document from ground truth, decode it.

This does two jobs at once.

**It measures the ceiling exactly.** ``run_baselines.py``'s ``crops-single-doc``
row approximates one document by degrading the ground truth at coverage numbers
read off the SPEC's summary table. This script does the real thing: it runs the
format's **own generator** on the ground-truth structure, producing a genuine
contacts-and-crops-v1 document — real Pass-1 sampling with its σ=2 Å box noise,
real Pass-2 crop selection with the frontier and re-show rules and the
σ=1/(i+1)² refinement schedule, real budget truncation — and then decodes it
back to coordinates with the same code every inference plan uses. Whatever that
scores is what a model that emitted a *perfect* document would score. It is the
number every model result has to be read against.

**It validates the decoder end-to-end.** The document is emitted by code that
knows nothing about ``document_codec``, in a random rotated and translated
frame, so a decode bug cannot cancel out: if the digit arithmetic, the crop
header state machine, the visit-index counting or the position-token inversion
were wrong, the score would collapse rather than land near the expected
ceiling.

Usage::

    uv run python oracle_document.py --gt-dir _scratch/gt \\
        --out-dir _scratch/pred/oracle-doc
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import canonical_pdb
from document_codec import (
    CoordinateEstimate,
    estimate_to_atom_array,
    parse_observations,
)
from marinfold.document_structures.contacts_and_crops_v1 import build_document
from marinfold.document_structures.contacts_v1.parse import (
    RawContact,
    residues_from_sequence,
)


def build_oracle_prediction(record: dict, gt_path: Path, contacts: list):
    """Generate a document from the ground-truth structure and decode it.

    Returns ``(atom_array, stats)``, or ``(None, stats)`` when the chain cannot
    be serialized.
    """
    gt = canonical_pdb.read_structure(gt_path)
    sequence = record["input_seq"]
    residues = residues_from_sequence(sequence)

    # The generator indexes atoms by 0-based sequence index; the canonical GT
    # file numbers residues from 1.
    atoms_by_seq_index = defaultdict(list)
    for res_id, atom_name, coord in zip(
        gt.res_id.tolist(), gt.atom_name.tolist(), gt.coord
    ):
        atoms_by_seq_index[res_id - 1].append(
            (atom_name, float(coord[0]), float(coord[1]), float(coord[2]))
        )

    result = build_document(
        record["record_id"],
        residues,
        [RawContact(seq_i=i, seq_j=j, degree=d) for i, j, d in contacts],
        {k: tuple(v) for k, v in atoms_by_seq_index.items()},
    )
    if result is None:
        return None, {"status": "unserializable"}

    valid_atoms = [frozenset() for _ in residues]
    for seq_index, entries in atoms_by_seq_index.items():
        valid_atoms[seq_index] = frozenset(name for name, *_ in entries)

    estimate = CoordinateEstimate()
    n_pass1 = n_crop = 0
    for observation in parse_observations(
        result.document.split(),
        start=result.start_index,
        length=len(sequence),
        valid_atoms=valid_atoms,
    ):
        estimate.add(observation)
        if observation.source == "pass1":
            n_pass1 += 1
        else:
            n_crop += 1

    array = estimate_to_atom_array(estimate, sequence)
    stats = {
        "status": "ok",
        "num_tokens": result.num_tokens,
        "num_eligible_atoms": result.num_eligible_atoms,
        "num_pass1_mentions": result.num_pass1_mentions,
        "crop_atoms_emitted": result.crop_atoms_emitted,
        "num_crops": result.num_crops,
        "max_box_visits": result.max_box_visits,
        "truncated": result.truncated,
        # Decoded, as opposed to emitted: how many mentions the codec recovered.
        "decoded_pass1_mentions": n_pass1,
        "decoded_crop_statements": n_crop,
        "decoded_atoms": len(estimate),
        "decoded_refined_atoms": len(estimate.refined_keys()),
    }
    return array, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stats", type=Path, default=None, help="per-record stats JSONL")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    records = [json.loads(line) for line in (args.gt_dir / "gt_index.jsonl").open()]
    contacts_by_record = {
        row["record_id"]: row["contacts"]
        for row in (json.loads(line) for line in (args.gt_dir / "gt_contacts.jsonl").open())
    }
    if args.limit is not None:
        records = records[: args.limit]

    stats_path = args.stats or (args.out_dir / "oracle_stats.jsonl")
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    n_ok = 0
    with stats_path.open("w") as stats_fh:
        for i, record in enumerate(records, start=1):
            gt_path = (
                args.gt_dir / "gt_structures" / record["dataset"] / f"{record['stem']}.pdb"
            )
            array, stats = build_oracle_prediction(
                record, gt_path, contacts_by_record[record["record_id"]]
            )
            stats["record_id"] = record["record_id"]
            stats["n_gt_atoms"] = record["n_gt_atoms"]
            stats_fh.write(json.dumps(stats) + "\n")
            if array is None:
                continue
            out_path = args.out_dir / record["dataset"] / f"{record['stem']}.pdb"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            canonical_pdb.write_structure(array, out_path)
            n_ok += 1
            if i % 50 == 0:
                print(f"  ...{i}/{len(records)} ({time.time() - started:.0f}s)", flush=True)

    print(
        f"[oracle] {n_ok}/{len(records)} documents generated + decoded "
        f"in {time.time() - started:.0f}s -> {args.out_dir}"
    )
    print(f"[oracle] stats -> {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
