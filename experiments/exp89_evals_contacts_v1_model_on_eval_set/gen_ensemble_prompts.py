# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Pre-generate K resampled contacts-v1 prefixes for every eval protein.

Each protein gets K independent sequence definitions (random start position +
random statement order — the format's nuisance symmetries). We materialize the
prefixes here (needs ``marinfold``) so the vLLM/TPU scorer can stay a thin,
``marinfold``-free worker that just reads prompts and scores them.

Output parquet rows: ``(dataset, stem, k, L, prefix, seq_positions)`` where
``prefix`` is the token string up to ``<begin_statements>`` and ``seq_positions``
is the position index per sequence index (so the worker forms ``<p{pos}>``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eval_contact_prediction import BEGIN, NUM_POS
from marinfold.document_structures.contacts_v1 import (
    GenerationConfig, build_document, residues_from_sequence,
)

EXP78 = Path("/home/bizon/git/MarinFold-exp78/experiments/exp78_evals_esmfold_contacts")
MANIFESTS = (EXP78 / "data/eval_manifest_foldbench.csv", EXP78 / "data/eval_manifest_exp65.csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for m in MANIFESTS:
        for _, r in pd.read_csv(m).iterrows():
            res = residues_from_sequence(r["input_seq"])
            for k in range(args.k):
                doc = build_document(f"{r['stem']}#{k}", res, [], config=GenerationConfig())
                if doc is None:
                    continue
                L = doc.seq_len
                prefix = doc.document[: doc.document.index(BEGIN) + len(BEGIN)]
                rows.append(dict(dataset=r["dataset"], stem=r["stem"], k=k, L=L,
                                 prefix=prefix,
                                 seq_positions=[(doc.n_term_index + i) % NUM_POS for i in range(L)]))
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df)} prompts ({df.stem.nunique()} proteins x {args.k} resamples) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
