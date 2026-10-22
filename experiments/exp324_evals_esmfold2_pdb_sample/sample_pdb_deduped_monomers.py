# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample proteins from the public exp222 PDB deduped-monomer contacts-v1 corpus.

Example:
    uv run --no-project --with 'huggingface_hub>=1.5' --with pyarrow --with pandas \
      --with './marinfold' python experiments/exp324_evals_esmfold2_pdb_sample/sample_pdb_deduped_monomers.py \
      --n 100 --seed 324 --max-len 512 --out experiments/exp324_evals_esmfold2_pdb_sample/data/sample_100_manifest.csv
"""

import argparse
import csv
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

CORPUS_PREFIX = "hf://buckets/open-athena/MarinFold/data/document_structures/contacts_v1_pdb_deduped_monomers/documents"
NUM_POSITION_INDICES = 2000
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "UNK": "X",
}


def _pos(token: str) -> int:
    if not (token.startswith("<p") and token.endswith(">")):
        raise ValueError(f"not a position token: {token}")
    return int(token[2:-1])


def sequence_from_document(document: str, seq_len: int) -> str:
    tokens = document.split()
    begin = tokens.index("<begin_sequence>")
    statements = tokens.index("<begin_statements>")
    section = tokens[begin + 1:statements]
    residues_by_pos: dict[int, str] = {}
    n_term = None
    for index in range(0, len(section), 2):
        head = section[index]
        tail = section[index + 1]
        if head == "<n-term>":
            n_term = _pos(tail)
        elif head == "<c-term>":
            continue
        else:
            residues_by_pos[_pos(head)] = tail.strip("<>")
    if n_term is None:
        raise ValueError("missing n-term")
    residues = []
    pos = n_term
    for _ in range(seq_len):
        residues.append(THREE_TO_ONE.get(residues_by_pos.get(pos, "UNK"), "X"))
        pos = (pos + 1) % NUM_POSITION_INDICES
    return "".join(residues)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=324)
    parser.add_argument("--min-len", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument(
        "--order",
        choices=["stem", "random", "length_desc"],
        default="stem",
        help="Output order. Use length_desc before modulo sharding to balance long proteins across shards.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fs = HfFileSystem(token=False)
    files = sorted(item["name"] for item in fs.ls(CORPUS_PREFIX) if item["name"].endswith(".parquet"))
    rows: list[dict[str, object]] = []
    columns = [
        "document",
        "entry_id",
        "pdb_id",
        "seq_len",
        "release_date",
        "resolution",
        "method",
        "cluster_ids",
        "chain_ids",
        "asu_chain_ids",
        "contacts_emitted",
        "highest_contact_degree",
        "lowest_included_contact_degree",
        "resolved_seq_sha1",
    ]
    for path in files:
        with fs.open(path, "rb") as handle:
            table = pq.read_table(handle, columns=columns)
        data = table.to_pylist()
        for row in data:
            seq_len = int(row["seq_len"])
            if not (args.min_len <= seq_len <= args.max_len):
                continue
            sequence = sequence_from_document(row["document"], seq_len=seq_len)
            if "X" in sequence:
                continue
            rows.append(
                {
                    "stem": row["entry_id"],
                    "entry_id": row["entry_id"],
                    "pdb_id": row["pdb_id"],
                    "chain_id": (row["chain_ids"] or [""])[0],
                    "asu_chain_id": (row["asu_chain_ids"] or [""])[0],
                    "sequence": sequence,
                    "seq_len": seq_len,
                    "release_date": row["release_date"],
                    "resolution": row["resolution"],
                    "method": row["method"],
                    "cluster_id": (row["cluster_ids"] or [""])[0],
                    "contacts_emitted": row["contacts_emitted"],
                    "highest_contact_degree": row["highest_contact_degree"],
                    "lowest_included_contact_degree": row["lowest_included_contact_degree"],
                    "resolved_seq_sha1": row["resolved_seq_sha1"],
                }
            )
    if len(rows) < args.n:
        raise SystemExit(f"only {len(rows)} eligible rows for n={args.n}")
    rng = random.Random(args.seed)
    sample = rng.sample(rows, args.n)
    if args.order == "stem":
        sample.sort(key=lambda r: str(r["stem"]))
    elif args.order == "length_desc":
        sample.sort(key=lambda r: (-int(r["seq_len"]), str(r["stem"])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample[0].keys()))
        writer.writeheader()
        writer.writerows(sample)
    lengths = [int(row["seq_len"]) for row in sample]
    print(
        f"wrote {len(sample)} rows to {args.out}; "
        f"eligible={len(rows)} L[min/median/max]={min(lengths)}/{sorted(lengths)[len(lengths)//2]}/{max(lengths)}"
    )


if __name__ == "__main__":
    main()
