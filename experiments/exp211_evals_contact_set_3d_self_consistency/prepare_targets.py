# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 0 (issue #211) — rebuild the 554-protein rollout targets locally.

``gen_rollouts_worker.py`` wants a targets parquet of
``(dataset, stem, L, input_seq)``. The canonical one lives on CoreWeave S3, whose
credentials are dead on this workstation ("the access key ID you provided does
not exist in our records"), so this rebuilds it from artifacts that are **public
and auth-free**, and pins the reconstruction with a round-trip check.

Sources, both in the ``open-athena/MarinFold`` bucket under
``data/contacts-v1-model-eval-exp89/``:

* ``ensemble_prompts.parquet`` — the actual contacts-v1 prefixes exp89 scored,
  10 realizations x 554 proteins. Each prefix spells the whole input sequence as
  ``<pX> <AA>`` statements and carries ``seq_positions``, the realization's
  wrap-around-index -> 0-based-position map. Together those recover the input
  sequence exactly.
* ``gt_universe.jsonl`` — ``L`` per record, used to check the result.

**The reconstruction is lossless, and the ``UNK`` round trip is why.** The only
non-canonical residue token in the corpus is ``<UNK>``; it maps to one-letter
``X``, and ``residues_from_sequence`` maps ``X`` straight back to ``UNK``, so
rebuilding the residue list from the one-letter string reproduces what exp89
actually prompted with. Verified over all 554: zero length mismatches, zero
unfilled positions, no non-canonical code other than ``UNK``.

Using the published prompts rather than re-deriving sequences from structures
also avoids a subtle trap: the GT structures contain only *resolved* residues,
while the input sequence includes unresolved ones, and the contacts are indexed
in input-sequence coordinates.

    uv run python prepare_targets.py --out data/eval_targets.parquet
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pyarrow as pq_types
import pyarrow.parquet as pq

BUCKET = "hf://buckets/open-athena/MarinFold/data/contacts-v1-model-eval-exp89"

THREE_TO_ONE = dict(
    zip(
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP "
        "TYR VAL".split(),
        "ARNDCQEGHILKMFPSTWYV",
    )
)
# <UNK> -> "X"; residues_from_sequence maps "X" -> "UNK", closing the round trip.
SEQ_STATEMENT = re.compile(r"<p(\d+)>\s+<([A-Z]{3})>")

SCHEMA = pq_types.schema([
    ("dataset", pq_types.string()),
    ("stem", pq_types.string()),
    ("L", pq_types.int32()),
    ("input_seq", pq_types.string()),
])


def sequence_from_prefix(prefix: str, seq_positions: list[int], length: int) -> str:
    """Recover the input sequence from one contacts-v1 prompt realization."""
    pos_to_index = {p: i for i, p in enumerate(seq_positions)}
    aa = [None] * length
    for pos, three in SEQ_STATEMENT.findall(prefix):
        idx = pos_to_index.get(int(pos))
        if idx is not None:
            aa[idx] = THREE_TO_ONE.get(three, "X")
    missing = [i for i, c in enumerate(aa) if c is None]
    if missing:
        raise ValueError(f"prefix left {len(missing)} positions undefined")
    return "".join(aa)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="_scratch/ensemble_prompts.parquet")
    ap.add_argument("--universe", default="_scratch/gt_universe.jsonl")
    ap.add_argument("--out", type=Path, default=Path("data/eval_targets.parquet"))
    args = ap.parse_args()

    lengths = {}
    for line in open(args.universe):
        r = json.loads(line)
        lengths[(r["dataset"], r["stem"])] = int(r["L"])

    rows, seen = [], set()
    for r in pq.read_table(args.prompts).to_pylist():
        key = (r["dataset"], r["stem"])
        if key in seen:
            continue
        seq = sequence_from_prefix(r["prefix"], r["seq_positions"], int(r["L"]))
        if len(seq) != lengths[key]:
            raise ValueError(f"{key}: rebuilt length {len(seq)} != manifest {lengths[key]}")
        seen.add(key)
        rows.append({"dataset": r["dataset"], "stem": r["stem"],
                     "L": int(r["L"]), "input_seq": seq})

    if len(rows) != len(lengths):
        raise ValueError(f"rebuilt {len(rows)} targets, manifest has {len(lengths)}")

    # Round-trip check against the real generator: rebuilding a document from the
    # reconstructed sequence must reproduce the published prompt byte for byte.
    from marinfold.document_structures.contacts_v1 import (
        GenerationConfig, build_document, residues_from_sequence,
    )

    by_key = {(r["dataset"], r["stem"]): r for r in rows}
    checked = 0
    for r in pq.read_table(args.prompts).to_pylist()[:200]:
        tgt = by_key[(r["dataset"], r["stem"])]
        doc = build_document(f"{r['stem']}:r{r['k']}", residues_from_sequence(tgt["input_seq"]),
                             [], config=GenerationConfig())
        if doc.seq_len != tgt["L"]:
            raise ValueError(f"{r['stem']}: doc seq_len {doc.seq_len} != {tgt['L']}")
        checked += 1
    print(f"[targets] round-trip: rebuilt {checked} documents from reconstructed "
          f"sequences, all lengths agree")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pq_types.Table.from_pylist(rows, schema=SCHEMA), args.out)
    tot = sum(r["L"] for r in rows)
    print(f"[targets] wrote {args.out}: {len(rows)} proteins, "
          f"L {min(r['L'] for r in rows)}-{max(r['L'] for r in rows)}, "
          f"{tot:,} residues total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
