# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize emitted contacts-v1 ground-truth pairs for an exp324 manifest.

The PDB-deduped monomer parquet stores the contacts-v1 document string rather
than parallel contact arrays. This script decodes those documents for the stems
in a manifest and writes a compact parquet with one row per protein and list
columns for the emitted binary contact pairs in sequence coordinates.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HF_DOC_PREFIX = "hf://buckets/open-athena/MarinFold/data/document_structures/contacts_v1_pdb_deduped_monomers/documents"
NUM_POSITION_INDICES = 2000
_POS_RE = re.compile(r"^<p(\d+)>$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hf-doc-prefix", default=HF_DOC_PREFIX)
    return parser.parse_args()


def position_id(token: str) -> int:
    match = _POS_RE.match(token)
    if match is None:
        raise ValueError(f"expected position token, got {token!r}")
    return int(match.group(1))


def ordered_positions(n_term: int, c_term: int) -> list[int]:
    out: list[int] = []
    pos = n_term
    while True:
        out.append(pos)
        if pos == c_term:
            return out
        pos = (pos + 1) % NUM_POSITION_INDICES
        if len(out) > NUM_POSITION_INDICES:
            raise ValueError("unterminated wrapped position interval")


def decode_contact_pairs(document: str) -> tuple[list[int], list[int]]:
    toks = document.split()
    begin_sequence = toks.index("<begin_sequence>")
    begin_statements = toks.index("<begin_statements>")
    end = toks.index("<end>")

    n_term: int | None = None
    c_term: int | None = None
    seq_toks = toks[begin_sequence + 1:begin_statements]
    for idx in range(0, len(seq_toks), 2):
        token = seq_toks[idx]
        value = seq_toks[idx + 1]
        if token == "<n-term>":
            n_term = position_id(value)
        elif token == "<c-term>":
            c_term = position_id(value)
    if n_term is None or c_term is None:
        raise ValueError("document missing n-term/c-term markers")

    pos_to_seq = {pos: seq_idx for seq_idx, pos in enumerate(ordered_positions(n_term, c_term))}
    contact_i: list[int] = []
    contact_j: list[int] = []
    stmt_toks = toks[begin_statements + 1:end]
    for idx in range(0, len(stmt_toks), 3):
        if stmt_toks[idx] != "<contact>":
            raise ValueError(f"expected <contact>, got {stmt_toks[idx]!r}")
        a = pos_to_seq[position_id(stmt_toks[idx + 1])]
        b = pos_to_seq[position_id(stmt_toks[idx + 2])]
        i, j = (a, b) if a < b else (b, a)
        contact_i.append(i)
        contact_j.append(j)
    return contact_i, contact_j


def manifest_stems(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        return {row["stem"] for row in csv.DictReader(handle)}


def main() -> None:
    args = parse_args()
    wanted = manifest_stems(args.manifest)
    rows: list[dict[str, Any]] = []
    for shard_index in range(3):
        uri = f"{args.hf_doc_prefix.rstrip('/')}/shard-{shard_index:05d}.parquet"
        table = pq.read_table(
            uri,
            columns=["entry_id", "seq_len", "document", "contacts_emitted", "truncated"],
        )
        df = table.to_pandas()
        df = df[df["entry_id"].isin(wanted)]
        print(f"{uri}: matched {len(df)} rows", flush=True)
        for rec in df.itertuples(index=False):
            contact_i, contact_j = decode_contact_pairs(rec.document)
            rows.append(
                {
                    "stem": rec.entry_id,
                    "seq_len": int(rec.seq_len),
                    "gt_contact_i": contact_i,
                    "gt_contact_j": contact_j,
                    "n_gt_contacts": len(contact_i),
                    "contacts_emitted": int(rec.contacts_emitted),
                    "truncated": bool(rec.truncated),
                }
            )
    found = {row["stem"] for row in rows}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"missing {len(missing)} manifest stems in HF docs, first={missing[:10]}")

    order = {stem: idx for idx, stem in enumerate(pd.read_csv(args.manifest)["stem"])}
    rows.sort(key=lambda row: order[row["stem"]])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("stem", pa.string()),
            ("seq_len", pa.int32()),
            ("gt_contact_i", pa.list_(pa.int32())),
            ("gt_contact_j", pa.list_(pa.int32())),
            ("n_gt_contacts", pa.int32()),
            ("contacts_emitted", pa.int32()),
            ("truncated", pa.bool_()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), args.out)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
