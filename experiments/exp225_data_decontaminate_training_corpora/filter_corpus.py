# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage 4 — rebuild a corpus with the contaminated rows removed.

A row filter on ``entry_id``, shard by shard. No regeneration: the surviving
rows are the *same rows*, byte-identical in content to what the original corpus
published, so a document in the decontaminated corpus is provably the document
the model would have seen.

**Layout semantics are preserved exactly.** One output shard per input shard,
same index, same name, rows in the same order — only the contaminated ones are
gone. That matters because both corpora encode meaning in their physical
order: AFDB's shards are round-descending (highest-pLDDT data trained on last,
#53), and ESM-Atlas's are 1:1 with their source parts (#139). Re-packing to
even shard sizes would silently destroy that, so shards simply come out
slightly smaller and slightly uneven. A shard that loses every row is still
written, empty, so the numbering never shifts.

**Published as a new prefix, never in place** — ``contacts_v1_decontam/`` and
``contacts_v1_esm_atlas_decontam/`` — so every existing checkpoint stays
reproducible against the corpus it actually saw.

The work splits in two because the two halves have very different costs. The
filter itself is download-bound and fast; the *upload* runs at the workstation's
~2.45 MB/s uplink (measured, and unchanged by parallelism — it is the physical
link), which is ~80 minutes for AFDB and ~15 hours for ESM-Atlas. So this
writes the filtered corpus to a local directory and leaves publishing to
``hf buckets sync``, which can then be run from wherever the bandwidth is.

    uv run python filter_corpus.py --arm afdb --out /data/exp225_decontam/contacts_v1_decontam
    hf buckets sync /data/exp225_decontam/contacts_v1_decontam \\
        hf://buckets/open-athena/MarinFold/data/document_structures/contacts_v1_decontam/train
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from build_corpus_index import BUCKET
from decontam_lib import ARMS, CORPORA, REFERENCE_VERSION, Corpus

HERE = Path(__file__).resolve().parent

def source_compression(fs: HfFileSystem, corpus: Corpus) -> str:
    """The codec shard 0 was written with, lowercased for pyarrow.

    Read rather than assumed, because the two corpora differ — AFDB is SNAPPY,
    ESM-Atlas is ZSTD — and getting it wrong is expensive but silent: writing
    the ESM-Atlas rebuild as SNAPPY inflated it from 133 GB to 191 GB, which at
    this uplink is an extra seven hours of upload for a corpus with *fewer*
    rows than the original.
    """
    with fs.open(f"{BUCKET}/{corpus.remote(0)}", "rb") as fh:
        row_group = pq.ParquetFile(fh).metadata.row_group(0)
    codecs = {row_group.column(i).compression for i in range(row_group.num_columns)}
    if len(codecs) != 1:
        raise SystemExit(f"{corpus.arm}: mixed column codecs {codecs}; cannot round-trip")
    return codecs.pop().lower()


@lru_cache(maxsize=None)
def _filesystem() -> HfFileSystem:
    """One shared anonymous client — constructing one per shard is pure latency."""
    return HfFileSystem(token=False)


def filter_shard(job: tuple[Corpus, int, frozenset, Path, str]) -> dict:
    """Download one shard, drop the contaminated rows, write it locally.

    Returns a manifest row. Resumable: an existing output shard with a
    recorded row count short-circuits the download entirely.
    """
    corpus, shard, dropped, out_dir, compression = job
    name = corpus.shard_name.format(shard)
    out = out_dir / name
    if out.exists():
        return {"shard": shard, "name": name, "rows_in": None,
                "rows_out": pq.ParquetFile(out).metadata.num_rows,
                "bytes": out.stat().st_size, "reused": True}

    # Read the shard into memory in one call, then parse from the buffer.
    # Handing pyarrow the fsspec file object directly makes it pull through
    # many small Python-level reads, each taking the GIL — with a thread pool
    # that serialises the downloads and the whole fan-out runs at roughly
    # single-stream speed (measured: 48 workers gave 40 MB/s where one stream
    # alone gives 17). One big read spends its time inside the HTTP client
    # with the GIL released.
    fs = _filesystem()
    with fs.open(f"{BUCKET}/{corpus.remote(shard)}", "rb") as fh:
        blob = fh.read()
    table = pq.read_table(pa.BufferReader(blob))
    rows_in = table.num_rows
    keep = pc.invert(pc.is_in(table.column("entry_id"), value_set=pa.array(sorted(dropped))))
    filtered = table.filter(keep)

    tmp = out.with_name(out.name + ".partial")
    pq.write_table(filtered, tmp, compression=compression,
                   row_group_size=max(filtered.num_rows, 1))
    tmp.rename(out)
    return {"shard": shard, "name": name, "rows_in": rows_in,
            "rows_out": filtered.num_rows, "bytes": out.stat().st_size, "reused": False}


def dataset_readme(corpus: Corpus, kept: int, dropped: int, rule: str) -> str:
    """The README that travels with the corpus — what was removed, and by what."""
    return f"""# `{corpus.decontam_prefix.split("/")[-2]}` — eval-decontaminated {corpus.label}

A row-filtered rebuild of
[`{corpus.prefix}`](https://huggingface.co/buckets/open-athena/MarinFold/tree/main/{corpus.prefix}),
produced by [issue #225](https://github.com/Open-Athena/MarinFold/issues/225).
The surviving rows are unchanged — same documents, same order, same shard
numbering — with contaminated rows removed.

| field | value |
| --- | --- |
| `decontam_reference` | `contacts_v1_eval_reference/{REFERENCE_VERSION}` |
| `decontam_rule` | {rule} |
| documents before | {corpus.n_documents:,} |
| documents removed | {dropped:,} ({100 * dropped / corpus.n_documents:.3f} %) |
| **documents after** | **{kept:,}** |
| shards | {corpus.n_shards} (unchanged; sizes are now uneven) |

## What the reference is

The union of two query sets, both published under
`data/decontamination/contacts_v1_eval_reference/{REFERENCE_VERSION}/`:

- **`eval_queries.fasta`** — the 554-protein contact benchmark (#89) that #180
  tracks: FoldBench-100 + exp65's 454 low-MSA / novel-fold candidates.
- **`foldbench_all_queries.fasta`** — every protein chain in *all* of FoldBench
  (1,940 chains / 1,449 unique sequences across the monomer, protein-protein,
  antibody-antigen, protein-peptide, protein-ligand, protein-DNA and
  protein-RNA tasks), so the corpus is clean against FoldBench tasks we do not
  currently score as well as the ones we do.

`droplist_final.parquet` alongside them is the exact per-row list this rebuild
applied, with the identity, coverage and nearest reference protein for every
removed row.

## Caveats

- **Sequence axis only.** No structural decontamination is applied here. #225
  measured that a fold-level purge (TM ≥ 0.5) would cost 37 % of the AFDB
  corpus and declined it; the cheap structural tier (TM ≥ 0.9, 0.54 % of AFDB)
  is measured in that experiment but is *not* applied to this build.
- **The rule uses no E-value threshold**, so it has no significance floor of
  its own and its size depends mildly on how deep MMseqs2 was asked to report
  (`-e 1000`, `-s 7.5`, `--max-seqs 1000000`, uncensored).
- The originals are untouched, so any existing checkpoint stays reproducible
  against the corpus it actually saw.
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--droplist", type=Path,
                    default=Path("/data/exp225_decontam/droplist_final.parquet"))
    ap.add_argument("--out", type=Path, default=None,
                    help="default: /data/exp225_decontam/<decontam prefix leaf>")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit-shards", type=int, default=None,
                    help="smoke test: filter only this many shards (skips the row-count "
                         "assertion)")
    ap.add_argument("--rule", default="identity >= 30% over >= 50% of the shorter sequence, "
                                      "no E-value threshold",
                    help="recorded verbatim in the dataset README")
    args = ap.parse_args()

    corpus = CORPORA[args.arm]
    out_dir = args.out or Path("/data/exp225_decontam") / corpus.decontam_prefix.split("/")[-2]
    out_dir.mkdir(parents=True, exist_ok=True)

    droplist = pd.read_parquet(args.droplist, columns=["arm", "entry_id"])
    dropped = frozenset(droplist.loc[droplist["arm"] == args.arm, "entry_id"])
    print(f"[{args.arm}] {len(dropped):,} entry_ids to drop of {corpus.n_documents:,}",
          flush=True)

    compression = source_compression(_filesystem(), corpus)
    print(f"[{args.arm}] source codec: {compression}", flush=True)

    n_shards = min(args.limit_shards or corpus.n_shards, corpus.n_shards)
    jobs = [(corpus, shard, dropped, out_dir, compression) for shard in range(n_shards)]
    rows: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, row in enumerate(pool.map(filter_shard, jobs), 1):
            rows.append(row)
            if done % 100 == 0 or done == n_shards:
                rate = done / max(time.time() - t0, 1e-9)
                print(f"[{args.arm}] {done}/{n_shards} shards, "
                      f"{sum(r['rows_out'] for r in rows):,} rows kept, "
                      f"{time.time() - t0:.0f}s (eta {(n_shards - done) / rate / 60:.1f} min)",
                      flush=True)

    rows.sort(key=lambda r: r["shard"])
    kept = sum(r["rows_out"] for r in rows)
    total_bytes = sum(r["bytes"] for r in rows)
    manifest = out_dir / "shard_manifest.csv"
    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if args.limit_shards is None:
        expected = corpus.n_documents - len(dropped)
        if kept != expected:
            raise SystemExit(
                f"{args.arm}: kept {kept:,} rows but the drop list implies {expected:,}. "
                "Either the filter missed rows or an entry_id is not unique — do not "
                "publish this build."
            )
        print(f"[{args.arm}] row count checks out: {kept:,} = {corpus.n_documents:,} - "
              f"{len(dropped):,}", flush=True)
        (out_dir / "README.md").write_text(
            dataset_readme(corpus, kept, len(dropped), args.rule)
        )

    (out_dir / "build.provenance.json").write_text(
        json.dumps(
            {
                "arm": args.arm,
                "source_prefix": corpus.prefix,
                "target_prefix": corpus.decontam_prefix,
                "reference_version": REFERENCE_VERSION,
                "rule": args.rule,
                "droplist": str(args.droplist),
                "n_documents_before": corpus.n_documents,
                "n_dropped": len(dropped),
                "n_documents_after": kept,
                "n_shards": n_shards,
                "bytes": total_bytes,
                "compression": compression,
                "seconds": round(time.time() - t0, 1),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[{args.arm}] {kept:,} rows in {n_shards} shards, "
          f"{total_bytes / 1e9:.1f} GB -> {out_dir}", flush=True)
    print(f"[{args.arm}] publish with:\n"
          f"  hf buckets sync {out_dir} "
          f"hf://buckets/open-athena/MarinFold/{corpus.decontam_prefix}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
