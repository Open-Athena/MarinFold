# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Run the retraction diagnostics over a backtracking corpus parquet (#160).

Smoke-tests the diagnostics on authored data before any training, and gives the
corpus's own discrimination numbers as the reference the trained model must
approach. The parquet's ``document`` column is a full contacts-v1 document, so
GT is recovered as the fold's live set at ``<end>`` (correct by construction for
this corpus) and diagnostics run on the edit list.

    uv run python diagnose_corpus.py --parquet ../exp159_data_backtracking_corpus/data/backtracking_corpus.parquet
"""
import argparse

import pandas as pd
from retraction_diagnostics import aggregate, diagnose_document, format_report

from marinfold.document_structures.contacts_v1.read import (
    iter_structure_statements,
    live_contacts,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    args = ap.parse_args()
    df = pd.read_parquet(args.parquet)
    diags = []
    for doc in df["document"]:
        statements = list(iter_structure_statements(doc))
        diags.append(diagnose_document(statements, live_contacts(doc)))
    print(format_report(aggregate(diags)))


if __name__ == "__main__":
    main()
