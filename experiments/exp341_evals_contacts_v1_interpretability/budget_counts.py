# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""B2 count: how often the 8192-token contact budget truncates training documents.

Reads the per-document generator metadata (``seq_len``, ``truncated``,
``contacts_passing_min_degree``, ``contacts_emitted``,
``lowest_included_contact_degree``, ``num_tokens``) from the two
decontaminated native corpora exp232 trained on and exp277 reused, as
published on the public MarinFold HF bucket:

- ``data/document_structures/contacts_v1_decontam/train`` (AFDB, 3,963,003 docs)
- ``data/document_structures/contacts_v1_esm_atlas_decontam/train`` (ESM Atlas,
  65,553,178 docs)

Only those six columns are read (``hf_bucket.read_columns``: parquet column
projection over HTTP range requests), so the ~143 GB of document text is
never downloaded. Each shard's columns are cached under
``--cache-dir`` so an interrupted scan resumes.

exp277 also trained on ProteinMPNN redesigns of the same structures (exp266:
31.7M AFDB-derived and 130.9M ESM-derived documents). Those keep each
structure's contact set and length, so their truncation follows the native
document's; they live only on CoreWeave and are not scanned here. Weighting
the native rows by exp277's per-corpus document counts reproduces them if
needed.

Writes ``data/budget_counts.csv`` (one row per corpus x length bin) and
``data/budget_summary.json`` (per-corpus totals).

    uv run python budget_counts.py --cache-dir /path/to/cache
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from hf_bucket import list_parquet, read_columns

CORPORA = {
    "native-afdb": ("data/document_structures/contacts_v1_decontam/train", 3_963_003),
    "native-esm": ("data/document_structures/contacts_v1_esm_atlas_decontam/train", 65_553_178),
}
COLUMNS = [
    "seq_len",
    "truncated",
    "contacts_passing_min_degree",
    "contacts_emitted",
    "lowest_included_contact_degree",
    "num_tokens",
]
LENGTH_BINS = [0, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000]
DATA = Path(__file__).resolve().parent / "data"


def fetch_columns(path: str, cache_dir: Path) -> pa.Table:
    cached = cache_dir / Path(path).name
    if cached.exists():
        return pq.read_table(cached)
    table = read_columns(path, COLUMNS)
    partial = cached.with_suffix(".partial")
    pq.write_table(table, partial)
    partial.replace(cached)
    return table


def scan_corpus(name: str, prefix: str, expected_docs: int, cache_dir: Path, workers: int) -> pd.DataFrame:
    shards = list_parquet(prefix)
    corpus_cache = cache_dir / name
    corpus_cache.mkdir(parents=True, exist_ok=True)
    tables = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_columns, path, corpus_cache): path for path in shards}
        for done, future in enumerate(as_completed(futures), start=1):
            tables.append(future.result())
            if done % 250 == 0 or done == len(shards):
                print(f"{name}: {done}/{len(shards)} shards", flush=True)
    frame = pa.concat_tables(tables).to_pandas()
    if len(frame) != expected_docs:
        raise ValueError(f"{name}: read {len(frame):,} documents from {len(shards)} shards, expected {expected_docs:,}")
    return frame


def summarize(frame: pd.DataFrame) -> dict:
    truncated = frame["truncated"]
    eligible = frame["contacts_passing_min_degree"].sum()
    return {
        "documents": int(len(frame)),
        "tokens": int(frame["num_tokens"].sum()),
        "share_docs_truncated": float(truncated.mean()),
        "share_tokens_in_truncated_docs": float(frame.loc[truncated, "num_tokens"].sum() / frame["num_tokens"].sum()),
        "eligible_contacts": int(eligible),
        "share_eligible_contacts_dropped": float(1 - frame["contacts_emitted"].sum() / eligible) if eligible else None,
        "median_lowest_included_degree_truncated": float(frame.loc[truncated, "lowest_included_contact_degree"].median())
        if truncated.any()
        else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, required=True, help="per-shard column cache (resumable)")
    args = parser.parse_args()

    rows = []
    summary = {}
    for name, (prefix, expected) in CORPORA.items():
        frame = scan_corpus(name, prefix, expected, args.cache_dir, args.workers)
        summary[name] = summarize(frame)
        bins = pd.cut(frame["seq_len"], LENGTH_BINS, right=True)
        for interval, group in frame.groupby(bins, observed=True):
            rows.append({"corpus": name, "length_bin": f"{interval.left + 1}-{interval.right}", **summarize(group)})
        print(json.dumps({name: summary[name]}, indent=2), flush=True)

    DATA.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(DATA / "budget_counts.csv", index=False, float_format="%.6g")
    (DATA / "budget_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
