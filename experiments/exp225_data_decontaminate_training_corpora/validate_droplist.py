# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Prove the drop list names real corpus rows before anything is filtered with it.

The drop list is produced by a chain of four independent steps — a contacts-v1
document decoded back to a sequence (#213), written into a FASTA header, keyed
by MMseqs2, and parsed back out here. Every step is a place where the list could
end up naming rows that do not exist, or worse, rows that exist but are not the
ones the alignment was against. Either failure is silent: the Stage 4 filter
would simply drop fewer rows than intended, and the resulting corpus would look
decontaminated without being so.

Two checks, and the second is the one that matters:

* **Existence** — every ``entry_id`` in the drop list is an ``entry_id`` in the
  corpus. Catches a mangled id.
* **Coordinate agreement** — the ``(shard, row)`` the header carries, resolved
  against the corpus index, yields *the same* ``entry_id``. This is the strong
  check: the coordinates and the id travel through the pipeline independently,
  so agreement means the alignment really was against the row we are about to
  drop. Catches an off-by-one or a mis-ordered shard, which existence cannot.

Also asserted: ``entry_id`` is unique within the corpus, without which a filter
keyed on it would remove more than the list names.

    uv run python validate_droplist.py --arm afdb
    uv run python validate_droplist.py --arm esm_atlas --limit-shards 60
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from build_corpus_index import read_shard, shard_columns
from decontam_lib import ARMS, CORPORA
from huggingface_hub import HfFileSystem

HERE = Path(__file__).resolve().parent


def corpus_index(arm: str, work: Path, limit_shards: int | None) -> pd.DataFrame:
    """The full index if it has been built, else a freshly sampled prefix."""
    full = work / f"index_{arm}.parquet"
    if limit_shards is None:
        if not full.exists():
            raise SystemExit(f"{full} does not exist; run build_corpus_index.py --arm {arm}")
        return pd.read_parquet(full, columns=["entry_id", "shard", "row"])

    corpus = CORPORA[arm]
    fs = HfFileSystem(token=False)
    columns = shard_columns(fs, corpus)
    print(f"[{arm}] sampling shards 0..{limit_shards - 1}", flush=True)
    jobs = [(corpus, shard, columns) for shard in range(limit_shards)]
    # Threaded: each shard read is a network round trip, and the ESM-Atlas
    # shards are ~40 MB apiece, so serial sampling is minutes of pure waiting.
    with ThreadPoolExecutor(max_workers=16) as pool:
        tables = list(pool.map(read_shard, jobs))
    frame = pd.concat([t.to_pandas() for t in tables], ignore_index=True)
    return frame[["entry_id", "shard", "row"]]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--work", type=Path, default=Path("/data/exp225_decontam"))
    ap.add_argument("--droplist", type=Path, default=None,
                    help="default: <work>/droplist_sequence.parquet")
    ap.add_argument("--limit-shards", type=int, default=None,
                    help="validate against a sampled shard prefix instead of a full index; "
                         "only drop-list rows inside that prefix are checked")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: data/droplist_validation_<arm>.json")
    args = ap.parse_args()

    out = args.out or HERE / f"data/droplist_validation_{args.arm}.json"
    index = corpus_index(args.arm, args.work, args.limit_shards)
    if not index["entry_id"].is_unique:
        raise SystemExit(
            f"{args.arm}: entry_id is not unique in the corpus — a filter keyed on it "
            "would drop more rows than the list names"
        )

    droplist = pd.read_parquet(args.droplist or args.work / "droplist_sequence.parquet")
    dropped = droplist[droplist["arm"] == args.arm]
    if args.limit_shards is not None:
        dropped = dropped[dropped["shard"] < args.limit_shards]

    missing = sorted(set(dropped["entry_id"]) - set(index["entry_id"]))
    joined = dropped.merge(index, on=["shard", "row"], suffixes=("_drop", "_corpus"))
    agree = int((joined["entry_id_drop"] == joined["entry_id_corpus"]).sum())

    report = {
        "arm": args.arm,
        "corpus_rows_checked": len(index),
        "droplist_rows_checked": len(dropped),
        "entry_ids_missing_from_corpus": len(missing),
        "missing_examples": missing[:10],
        "coordinates_resolved": len(joined),
        "coordinates_agreeing_with_entry_id": agree,
        "entry_id_unique_in_corpus": True,
        "limit_shards": args.limit_shards,
        "droplist": str(args.droplist or args.work / "droplist_sequence.parquet"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)

    if missing:
        raise SystemExit(f"{len(missing)} drop-list entry_ids are not in the corpus")
    if len(joined) != len(dropped) or agree != len(dropped):
        raise SystemExit(
            f"coordinate check failed: {len(joined)}/{len(dropped)} resolved, "
            f"{agree} agreed — the drop list does not name the rows it was computed from"
        )
    print(f"[validate] {args.arm}: all {len(dropped):,} drop-list rows verified", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
