# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the 50:50 backtracking mix to the open-athena/MarinFold HF bucket (#160).

The mix the training run actually consumed lives in ``gs://marin-us-east5``
(and, from the abandoned GPU route, CoreWeave S3) — both private working
copies. This puts it on the public bucket, beside the corpora it was built
from, so the run is reproducible without GCS credentials.

Reads the shards from GCS rather than rebuilding them with ``build_mix.py``:
the mix is a *sample* (a shuffled draw of clean documents from exp139's 66.76M),
so re-running the builder would produce a different corpus even at the same
seed if either source changed. Publishing the exact bytes that were trained on
is the point.

The tokenizer travels with the data (the repo's convention for corpora): the
3849-token crops/ccoord **superset**, whose final token is ``<retract>``. It is
copied from the bucket directory the training job read, not rebuilt, for the
same reason.

    uv run python publish_mix_to_hf.py --stage     # GCS -> local + stats
    uv run python publish_mix_to_hf.py --upload    # local -> HF bucket
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

GCS = ("gs://marin-us-east5/protein-structure/MarinFold/"
       "exp160_backtracking_training/corpus")
BUCKET = ("hf://buckets/open-athena/MarinFold/data/document_structures/"
          "contacts_v1_backtracking_mix50")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def stage(out_dir: Path) -> dict:
    """Copy the shards down and summarise them. Returns the stats for the README."""
    import fsspec

    train = out_dir / "train"
    train.mkdir(parents=True, exist_ok=True)
    fs, _ = fsspec.core.url_to_fs(GCS)
    shards = sorted(fs.glob(f"{GCS}/train/*.parquet"))
    if not shards:
        raise SystemExit(f"no shards under {GCS}/train")
    print(f"{len(shards)} shards to stage", flush=True)

    totals = {"documents": 0, "tokens": 0, "backtracking": 0, "clean": 0, "shards": len(shards)}
    bt_ids: set[str] = set()
    clean_ids: set[str] = set()
    for i, key in enumerate(shards):
        dest = train / Path(key).name
        if not dest.exists():
            fs.get_file(key, str(dest))
        df = pd.read_parquet(dest)
        totals["documents"] += len(df)
        totals["tokens"] += int(df.num_tokens.sum())
        counts = df.kind.value_counts()
        totals["backtracking"] += int(counts.get("backtracking", 0))
        totals["clean"] += int(counts.get("clean", 0))
        bt_ids |= set(df.entry_id[df.kind == "backtracking"])
        clean_ids |= set(df.entry_id[df.kind == "clean"])
        print(f"  {dest.name}: {len(df):,} docs", flush=True)

    # The disjointness claim is the mix's one real invariant — a protein seen in
    # both forms would confound "learns to retract" with "memorises this
    # protein". Re-checked on the published bytes, not just at build time.
    totals["protein_overlap"] = len(bt_ids & clean_ids)
    totals["proteins"] = len(bt_ids | clean_ids)
    if totals["protein_overlap"]:
        raise SystemExit(f"halves share {totals['protein_overlap']} proteins — refusing to publish")

    tok = out_dir / "tokenizer"
    tok.mkdir(parents=True, exist_ok=True)
    for name in TOKENIZER_FILES:
        fs.get_file(f"{GCS}/tokenizer/{name}", str(tok / name))
    vocab = json.loads((tok / "tokenizer.json").read_text())["model"]["vocab"]
    totals["vocab_size"] = len(vocab)
    totals["retract_id"] = vocab["<retract>"]
    print(f"tokenizer: {totals['vocab_size']} tokens, <retract>={totals['retract_id']}")
    return totals


def render_readme(s: dict) -> str:
    d = s["documents"]
    return f"""# contacts-v1 backtracking 50:50 mix (exp160)

The exact training corpus of issue
[#160](https://github.com/Open-Athena/MarinFold/issues/160): {d:,} contacts-v1
documents, half of which **retract their own mistakes** and half of which are
ordinary clean documents.

| | |
|---|---|
| documents | {d:,} |
| backtracking half | {s['backtracking']:,} ({s['backtracking'] / d:.1%}) |
| clean half | {s['clean']:,} ({s['clean'] / d:.1%}) |
| tokens | {s['tokens']:,} |
| distinct proteins | {s['proteins']:,} |
| **proteins shared between halves** | **{s['protein_overlap']}** |
| shards | {s['shards']} |

## Why a mix

The backtracking half ([#159](https://github.com/Open-Athena/MarinFold/issues/159))
teaches the model to take a wrong contact back. On its own it would also teach
it that *every* document contains retractions. The clean half — ordinary
contacts-v1 documents sampled from the ESM-Atlas corpus
([#139](https://github.com/Open-Athena/MarinFold/issues/139)) — keeps the model
able to answer without retracting, so backtracking stays a move it can choose
rather than a format it must fill.

The two halves are drawn from **disjoint proteins** ({s['protein_overlap']}
shared, asserted above on the published shards). exp139 has 66.76M documents
and #159 used ~1M of them, so there is no reason to show the model the same
protein in both forms.

## Contents

```
train/shard-{{00000..{s['shards'] - 1:05d}}}.parquet
tokenizer/          # the {s['vocab_size']}-token crops/ccoord superset vocab
README.md
```

Columns: `entry_id`, `document`, `num_tokens`, `kind` (`backtracking` | `clean`).

## Tokenizer

The **superset** vocab ({s['vocab_size']} tokens), not the contacts-v1 one:
`<retract>` is its final token, id **{s['retract_id']}**. Against the
contacts-v1 tokenizer the exp120 base model was trained under, this is an
append — every one of that model's 2,845 ids keeps its meaning, verified with
0 mismatches — so warm-starting needs an embedding **resize**, not a remap.
Levanter does not resize on warm start, so #160 does it offline
(`resize_init_vocab.py`) and asserts rows 0..2844 come out bit-identical.

Use one tokenizer throughout a run: `<retract>` has a different id here than in
the plain contacts-v1 vocab, and both are correct in their own corpus.

## Reading it

```python
import pandas as pd
from marinfold.document_structures.contacts_v1.read import live_contacts

df = pd.read_parquet("hf://buckets/open-athena/MarinFold/data/"
                     "document_structures/contacts_v1_backtracking_mix50/"
                     "train/shard-00000.parquet")
live_contacts(df.document[0])   # the contacts still asserted at <end>
```
"""


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("data/publish_mix"))
    ap.add_argument("--stage", action="store_true", help="GCS -> local + stats")
    ap.add_argument("--upload", action="store_true", help="local -> HF bucket")
    ap.add_argument("--hf", default="hf")
    args = ap.parse_args()

    stats_path = args.dir / "stats.json"
    if args.stage:
        args.dir.mkdir(parents=True, exist_ok=True)
        stats = stage(args.dir)
        stats_path.write_text(json.dumps(stats, indent=2))
        print(f"\nstaged: {stats}")

    if args.upload:
        stats = json.loads(stats_path.read_text())
        (args.dir / "README.md").write_text(render_readme(stats))
        _run([args.hf, "buckets", "sync", str(args.dir / "train"), f"{BUCKET}/train"])
        _run([args.hf, "buckets", "sync", str(args.dir / "tokenizer"), f"{BUCKET}/tokenizer"])
        _run([args.hf, "buckets", "cp", str(args.dir / "README.md"), f"{BUCKET}/README.md"])
        print(f"\npublished -> {BUCKET}")


if __name__ == "__main__":
    main()
