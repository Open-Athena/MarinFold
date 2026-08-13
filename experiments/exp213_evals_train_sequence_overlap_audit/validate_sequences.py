# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 1b — check the recovered AFDB sequences against an external authority.

``sequence_from_document`` is unit-tested as the exact inverse of
``build_document``, but that only proves the *reader* is right. This checks the
whole chain — published corpus shard → document → one-letter sequence — against
the AlphaFold DB API, which serves the ``uniprotSequence`` each AFDB structure
was folded from. If the two agree, the FASTA the overlap search runs on really
is the training set's sequences.

Only the AFDB arm needs this: the ESM-Atlas arm reads ``sequence`` straight out
of its manifest, with no decoding step to get wrong.

Sampled entries can legitimately 404 — ``afdb-24M`` is a snapshot and AlphaFold
DB has since dropped entries — so those are reported separately and are not
failures. A single *mismatch* is.

    uv run python validate_sequences.py --fasta /data/exp213_overlap/train_afdb.fasta -n 50
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction/{}"


def iter_fasta(path: Path, limit: int | None = None):
    """Yield ``(header, sequence)``; stops after ``limit`` records if given."""
    header, chunks = None, []
    with path.open() as fh:
        for line in fh:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                    if limit is not None:
                        limit -= 1
                        if limit <= 0:
                            return
                header, chunks = line[1:].strip(), []
            else:
                chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks)


def uniprot_accession(header: str) -> str:
    """``afdb|00000_0_AF-A0A7C3LD06-F1`` -> ``A0A7C3LD06``."""
    entry_id = header.split("_", 2)[2]
    parts = entry_id.split("-")
    if len(parts) < 3 or parts[0] != "AF":
        raise ValueError(f"unexpected AFDB entry_id {entry_id!r} in header {header!r}")
    return "-".join(parts[1:-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", type=Path, required=True)
    ap.add_argument("-n", "--num-samples", type=int, default=50)
    ap.add_argument("--pool", type=int, default=20_000,
                    help="read this many records, then sample from them")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/sequence_validation.json"))
    args = ap.parse_args()

    records = list(iter_fasta(args.fasta, limit=args.pool))
    sample = random.Random(args.seed).sample(records, min(args.num_samples, len(records)))
    print(f"checking {len(sample)} of {len(records):,} records from {args.fasta}",
          flush=True)

    matched, mismatched, missing = [], [], []
    for header, sequence in sample:
        accession = uniprot_accession(header)
        try:
            with urllib.request.urlopen(AFDB_API.format(accession), timeout=60) as resp:
                reference = json.load(resp)[0]["uniprotSequence"]
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, IndexError) as e:
            missing.append({"accession": accession, "reason": str(e)})
            continue
        if reference == sequence:
            matched.append(accession)
        else:
            mismatched.append({
                "accession": accession,
                "ours_len": len(sequence), "reference_len": len(reference),
                "first_diff": next((i for i, (a, b) in enumerate(zip(sequence, reference))
                                    if a != b), min(len(sequence), len(reference))),
            })

    summary = {
        "fasta": str(args.fasta),
        "sampled": len(sample),
        "matched": len(matched),
        "mismatched": len(mismatched),
        "not_in_current_afdb_release": len(missing),
        "mismatch_detail": mismatched[:10],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if mismatched:
        print(f"FAIL: {len(mismatched)} sequences disagree with AlphaFold DB",
              file=sys.stderr)
        return 1
    if not matched:
        print("FAIL: no sequence could be checked at all", file=sys.stderr)
        return 1
    print(f"OK: {len(matched)}/{len(matched)} resolvable entries match "
          f"({len(missing)} no longer in the current AFDB release)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
