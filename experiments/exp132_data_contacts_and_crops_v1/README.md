---
marinfold_experiment:
  issue: 132
  title: "exp: generate contacts-and-crops-v1 training set"
  kind: data
  branch: worktree-exp130-contacts-and-crops
---

# exp132 — generate the contacts-and-crops-v1 dataset (local)

**Issue:** [#132](https://github.com/Open-Athena/MarinFold/issues/132) · **Kind:** `data`

## Question

Generate the
[`contacts-and-crops-v1`](../../marinfold/marinfold/document_structures/contacts_and_crops_v1)
(#130 / PR #131) training corpus at scale — **for the same proteins as the
contacts-v1 training set** — and publish it (sharded parquet + tokenizer +
README) to the
[`open-athena/MarinFold`](https://huggingface.co/buckets/open-athena/MarinFold)
HF bucket, **calling into the existing generator rather than
re-implementing it**.

## Approach

Mirrors [exp105](../exp105_data_contacts_and_coordinates_v1_zephyr) (ccoord
corpus) and exp126 (think corpus).

**Same proteins as contacts-v1.** Reuse exp53's *selection manifest*
verbatim — one row per selected `(entry, round)` record — so there is no new
selection stage: identical proteins, rounds, and splits.

```
gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/selection_manifest/{train,val,test}/shard_*.parquet
```

**One stage (generation), run locally.** For each manifest row,
`generate_rows.py` fetches the AFDB structure from its `gcs_uri` — the
**public** `gs://public-datasets-deepmind-alphafold-v4` bucket, so no
requester-pays / on-cluster dependency — parses it with gemmi, and calls
`contacts_and_crops_v1.generate_document(...)` with the manifest `entry_id`
as the deterministic seed. The output row is the format's `metadata_row()`
plus the manifest provenance columns. `generate_local.py` fans this out
one-process-per-shard (`ProcessPoolExecutor`), resumable (present output
shards are skipped; writes are atomic via `.tmp` + `os.replace`).

## Run

```bash
# from the marinfold package dir (so the crops package is importable), with gcsfs:
SCRATCH=/data/exp132_contacts_and_crops_v1_scratch
gcloud storage cp -r \
  gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/selection_manifest/{train,val,test} \
  "$SCRATCH/manifest/"

PYTHONPATH=$(pwd)/../experiments/exp132_data_contacts_and_crops_v1 \
  uv run --with gcsfs python ../experiments/exp132_data_contacts_and_crops_v1/generate_local.py \
    --split all --manifest "$SCRATCH/manifest" --out "$SCRATCH/documents" --procs 48
```

## Forecast (measured on a 40-doc local pilot)

- ~4.21M docs (2067 train / 22 val / 22 test shards, ~2000 rows each).
- ~1.19 docs/s single-thread (pyconfind-bound) → **~24 h at 48 procs**.
- ~8190 tokens/doc → ~34.5 B tokens; **~60 GB** ZSTD parquet (14.3 KB/doc).

## Publish

```bash
export HF_TOKEN=...   # open-athena-scoped
uv run python publish_to_hf.py --documents "$SCRATCH/documents" \
    --hf-bin /home/bizon/anaconda3/bin/hf
```

→ `hf://buckets/open-athena/MarinFold/data/document_structures/contacts_and_crops_v1/{train,val,test}/` + `tokenizer/` + `README.md`.

## Files

- `generate_rows.py` — per-row worker: fetch + gemmi + `generate_document` → output row (adapted from exp105).
- `generate_local.py` — shard-parallel local driver (adapted from exp126).
- `publish_to_hf.py` — scan → render dataset README + tokenizer → sync to the bucket.
