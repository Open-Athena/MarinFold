# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""B1 extra proteins: the legacy-554 proteins not already used by exp341.

B1 first ran on the 116 eval-val and eval-denovo proteins. This adds every
legacy-554 protein that is

- not in ``foldbench100`` (its natural monomers are eval-val),
- not one of the 200 in ``data/a2_legacy_gt.jsonl.gz`` (A2's legacy set, on
  which the tuned lens is fitted), and
- not the same PDB chain as any exp245 eval protein: 14 ``denovo_pdb``
  chains are also eval-denovo proteins (e.g. ``denovo_pdb__8fjf_A`` and
  ``8fjf_A``), 8 of them in this pool.

#225 decontaminated exp277's corpora against all legacy-554 proteins, so
none is in exp277's training data. Records follow ``a2_select.py``'s
format (exp89 ground truth plus exp213's query sequence) and load with
``harness.load_legacy_proteins``.

Writes ``data/b1_extra_gt.jsonl.gz``.

    uv run python b1_select_extra.py --legacy-gt gt_universe.jsonl
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import pandas as pd

from a2_select import read_fasta

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXPERIMENTS = HERE.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--legacy-gt", type=Path, required=True,
                        help="exp89 gt_universe.jsonl (data/contacts-v1-model-eval-exp89/ on the HF bucket)")
    args = parser.parse_args()

    exp213 = EXPERIMENTS / "exp213_evals_train_sequence_overlap_audit" / "data"
    legacy = pd.read_csv(exp213 / "eval_train_identity.csv")
    legacy["key"] = legacy["dataset"] + "__" + legacy["stem"]
    with gzip.open(DATA / "a2_legacy_gt.jsonl.gz", "rt") as handle:
        used = {json.loads(line)["key"] for line in handle}
    eval_stems = set(pd.read_csv(EXPERIMENTS / "exp245_evals_foldbench_held_out_monomers" / "data" / "eval_sets.csv").stem)

    pool = legacy[legacy["dataset"] != "foldbench100"]
    pool = pool[~pool["key"].isin(used)]
    shared = pool["stem"].isin(eval_stems)
    print(f"{len(pool)} unused non-foldbench100 proteins; dropping {int(shared.sum())} that are exp245 eval chains: "
          f"{sorted(pool.loc[shared, 'key'])}")
    pool = pool[~shared]

    sequences = read_fasta(exp213 / "eval_queries.fasta")
    ground_truth = {}
    with args.legacy_gt.open() as handle:
        for line in handle:
            record = json.loads(line)
            ground_truth[f"{record['dataset']}__{record['stem']}"] = record
    with gzip.open(DATA / "b1_extra_gt.jsonl.gz", "wt") as out:
        for key in pool["key"]:
            record = ground_truth[key]
            out.write(json.dumps({
                "key": key, "dataset": record["dataset"], "stem": record["stem"], "L": record["L"],
                "sequence": sequences[key], "resolved": record["resolved"], "contacts": record["contacts"],
            }) + "\n")
    print(pool.groupby("dataset").agg(n=("key", "size"), median_L=("length", "median"), max_L=("length", "max")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
