# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidate + QA the ESM-Atlas backtracking corpus parts (#159).

The fan-out writes one parquet part per 250 documents. This reads them all
back, checks the corpus's hard invariants, reports the retraction statistics
that decide whether it is worth training on, and optionally writes a
consolidated parquet.

The checks are deliberately independent of the generator: correctness is
re-derived from each document by folding it with ``read.live_contacts`` rather
than trusting the metrics columns the worker wrote.

    uv run --with boto3 python consolidate_esm_atlas.py --stats-only
    uv run --with boto3 python consolidate_esm_atlas.py --out data/esm_atlas_corpus.parquet
"""

from __future__ import annotations

import argparse
import io
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_BUCKET = "marin-us-east-02a"
DEFAULT_PREFIX = (
    "protein-structure/MarinFold/exp159_backtracking_esm_atlas/documents/"
)


def _client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url="https://cwobject.com",
        aws_access_key_id=os.environ["CW_KEY_ID"],
        aws_secret_access_key=os.environ["CW_KEY_SECRET"],
        config=Config(s3={"addressing_style": "virtual"}),
        region_name="auto",
    )


def load_parts(bucket: str, prefix: str, limit: int | None = None) -> pd.DataFrame:
    s3 = _client()
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix
    ):
        keys.extend(o["Key"] for o in page.get("Contents", [])
                    if o["Key"].endswith(".parquet"))
    keys.sort()
    if limit:
        keys = keys[:limit]
    print(f"{len(keys)} parts", flush=True)
    frames = []
    for i, key in enumerate(keys):
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        frames.append(pd.read_parquet(io.BytesIO(body)))
        if (i + 1) % 50 == 0:
            print(f"  read {i + 1}/{len(keys)}", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def report(df: pd.DataFrame) -> None:
    """Corpus stats + the invariants that gate training on this data."""
    from marinfold.document_structures.contacts_v1.read import (
        iter_structure_statements,
        live_contacts,
    )

    n = len(df)
    print("\n=== corpus ===")
    print(f"documents:        {n:,}")
    print(f"unique entry_ids: {df.entry_id.nunique():,}")
    print(f"tokens:           {int(df.num_tokens.sum()):,}")
    print(f"mean seq_len:     {df.seq_len.mean():.1f}")
    print(f"truncated:        {int(df.truncated.sum())}")

    # Invariant, re-derived: the fold must recover exactly n_gt contacts.
    folded = df["document"].map(lambda d: len(live_contacts(d)))
    mismatches = int((folded != df["n_gt"]).sum())
    print(f"\nfold == n_gt:     {n - mismatches:,}/{n:,} "
          f"({'OK' if mismatches == 0 else f'{mismatches} MISMATCH'})")

    fp_total = int(df.n_fp_emitted.sum())
    fp_trig = int(df.fp_retracted_by_trigger.sum())
    tp_trig = int(df.tp_retracted_by_trigger.sum())
    print("\n=== retraction ===")
    print(f"docs with a retraction: {(df.n_retract_stmts > 0).mean():.1%}")
    print(f"mean retracts/doc:      {df.n_retract_stmts.mean():.1f}")
    print(f"FP emitted:             {fp_total:,}")
    print(f"caught by trigger:      {fp_trig:,} "
          f"({fp_trig / fp_total:.1%})" if fp_total else "")
    print(f"trigger false alarms:   {tp_trig:,}")

    # The #160 discrimination metric, computed on the corpus itself.
    try:
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "exp160_models_backtracking_training"))
        from retraction_diagnostics import aggregate, diagnose_document, format_report

        sample = df if n <= 20000 else df.sample(20000, random_state=0)
        diags = [
            diagnose_document(list(iter_structure_statements(doc)), live_contacts(doc))
            for doc in sample["document"]
        ]
        print(f"\n=== #160 diagnostics (n={len(sample):,}) ===")
        print(format_report(aggregate(diags)))
    except ImportError as exc:
        print(f"(diagnostics unavailable: {exc})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--limit-parts", type=int, default=None)
    ap.add_argument("--out", default=None, help="write a consolidated parquet")
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    df = load_parts(args.bucket, args.prefix, args.limit_parts)
    if df.empty:
        print("no parts found")
        return
    report(df)
    if args.out and not args.stats_only:
        df.to_parquet(args.out, index=False)
        print(f"\nwrote {args.out} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
