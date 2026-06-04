---
marinfold_experiment:
  issue: 53
  title: "exp: generate contacts_v1 dataset using zephyr"
  kind: data
  branch: exp/53-contacts-v1-zephyr
---

# exp53 — generate the contacts-v1 dataset on zephyr

**Issue:** [#53](https://github.com/Open-Athena/MarinFold/issues/53) · **Kind:** `data` · **Branch:** `exp/53-contacts-v1-zephyr`

## Question

Can we generate the [`contacts-v1`](../../marinfold/marinfold/document_structures/contacts_v1)
training-document corpus at scale from [`afdb-24M`](https://huggingface.co/datasets/timodonnell/afdb-24M)
— **up to 5 examples per structural cluster, organised into pLDDT "rounds"**
— on the marin Iris cluster via Zephyr, and publish it (sharded parquet +
README) to the [`open-athena/MarinFold`](https://huggingface.co/buckets/open-athena/MarinFold)
HF bucket and to GCS, in under ~an hour, **calling into the existing
contacts-v1 generator rather than re-implementing it**?

## Background

- `contacts-v1` document format + generator:
  [`marinfold/.../document_structures/contacts_v1`](../../marinfold/marinfold/document_structures/contacts_v1)
  (`generate_document(...) -> GenerationResult`). exp53 imports it directly.
- Prior at-scale Zephyr runner for a *different* doc type:
  [exp5 / PR #38](https://github.com/Open-Athena/MarinFold/pull/38)
  (`contacts-and-distances-v2`). exp53 reuses its Zephyr/Iris pattern
  (`Dataset.from_files(...).load_parquet(...).map_shard(...).write_parquet(...)`,
  per-row cif fetch from a `gcs_uri` column, `iris job run`).
- Precedent dataset: [`timodonnell/protein-docs`](https://huggingface.co/datasets/timodonnell/protein-docs)
  `contacts-and-distances-v1-5x` — a `round` column, `train`/`val`/`test`
  splits, 2000-row shards, physically laid out **round-0 first**. We mirror
  the schema and **reverse** the physical order (see below).

## Approach

Two stages, split along the natural seam — a cheap structure-free
**selection** (a cluster groupby/shuffle) and an expensive embarrassingly
parallel **generation**.

### Stage A — selection (`selection.py`, local DuckDB)

afdb-24M already carries everything selection needs in small columns
(`struct_cluster_id`, `global_plddt`, `seq_len`, `split`, `gcs_uri`) — **no
structure is parsed here.** We:

1. read those columns for all ~24M rows (12,005 shards),
2. pre-filter `seq_len ∈ [2, 2000]` (contacts-v1's serializable range),
3. group by `struct_cluster_id`, rank members by `global_plddt` descending
   (deterministic `entry_id` tie-break), assign `round = 0..4`
   (`round 0` = highest pLDDT), and
4. emit up to 5 rows per cluster as a **selection manifest**.

Two changes from protein-docs (per the issue):

- **Drop tiny clusters.** Clusters with `< 3` usable members are discarded,
  so `round-0 == round-1 == round-2` in size (the largest rounds) instead of
  round-0 being inflated by singletons.
- **Reverse the order.** The manifest is written as *single-round* shards
  numbered **descending** per split (round-4 shards first … round-0 last), so
  the published corpus trains on the highest-pLDDT data **last**. Every row
  also keeps an explicit `round` column.

> afdb-24M's `split` is cluster-consistent (verified: 0 of 110k sampled
> clusters mixed), so carrying per-entry `split` introduces no train/test
> leakage across a structural cluster.

Reading 12,005 shards over `hf://` directly trips HuggingFace's 3,000-req /
5-min **API** quota (one `paths-info` call per shard). `fetch_manifest_columns.py`
sidesteps it: list shard paths once, then range-read each shard's small
columns over the public `/resolve/` CDN URL (parquet footer + small-column
chunks only) into one local parquet that `selection.py` runs on.

### Stage B — generation (`cli.py` + `generate_rows.py`, Zephyr on Iris)

For each manifest row: fetch the structure from `gcs_uri`, parse with gemmi,
and call `contacts_v1.generate_document(structure, entry_id=...)`. The output
row is contacts-v1's `metadata_row()` (document text + all metadata) plus the
manifest provenance (`round`, `struct_cluster_id`, `seq_cluster_id`, `split`,
`uniprot_accession`, `tax_id`, `organism_name`). `generate_rows.py` holds the
per-row worker (no zephyr import, unit-tested locally + **byte-identical** to
calling `generate_document` directly); `cli.py` wraps it in a Zephyr
`map_shard`. Run **once per split** so the output keeps the `train`/`val`/`test`
layout and inherits Stage A's round-descending shard order. Rows whose
structure fails to generate (multi-chain / parse error / out of range) are
**dropped** — round-0 is not back-filled (decision below).

### Outputs

- **GCS** (working copy): `gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/`
  — `selection_manifest/<split>/...` and `documents/<split>/contacts_v1-{shard:05d}-of-{total:05d}.parquet`.
- **HF bucket** (published): `open-athena/MarinFold` →
  `data/document_structures/contacts_v1/<split>/shard_*.parquet` (round-desc) + `README.md`.
- **Tokenizer**: pushed via `contacts-v1 tokenizer` and referenced from the dataset README.

### Decisions (resolved with @timodonnell)

1. **GCS region** = `marin-us-east5` per AGENTS.md (exp5 used us-central1; not followed).
2. **Round-0 fill** = drop reps that fail to generate; no next-best-pLDDT backfill.
3. **Launch** = build + local/iris smoke + PR, then **pause for go-ahead** before the full run.

### Layout

```
exp53_data_contacts_v1_zephyr/
├── selection.py              # Stage A: cluster -> rounds manifest (DuckDB)
├── fetch_manifest_columns.py # rate-limit-safe afdb-24M small-column fetch
├── cli.py                    # Stage B: Zephyr/Iris generate driver
├── generate_rows.py          # Stage B per-row worker (no zephyr import)
├── pyproject.toml            # marinfold[contacts-v1] + duckdb + (extra: marin-zephyr)
└── tests/                    # Stage A unit tests + Stage B byte-identity tests
```

### Running it

```bash
uv sync --extra test               # Stage A + worker tests
uv run python -m pytest tests/ -q

# Stage A (local): consolidate small columns, then select.
uv run python fetch_manifest_columns.py --out ~/exp53_scratch/afdb24m_small.parquet
uv run python selection.py --input ~/exp53_scratch/afdb24m_small.parquet \
    --out gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/selection_manifest

# Stage B (Iris), once per split (train shown):
uv run --extra zephyr iris --cluster=marin job run --cpu 1 --memory 2GB -- \
  python cli.py generate \
    --input "gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/selection_manifest/train/shard_*.parquet" \
    --out   "gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/documents/train/contacts_v1-{shard:05d}-of-{total:05d}.parquet" \
    --worker-cpu 1 --worker-memory 4g --max-workers 512 --fetch-concurrency 32
```

## Success criteria

1. Documents come from `contacts_v1.generate_document` (no fork) — byte-identical
   to a direct call (pinned by `tests/test_generate_rows.py`).
2. Rounds are correct: ≤5 per cluster, `round-0 == round-1 == round-2` size,
   clusters with `<3` members dropped, physical shard order round-descending.
3. Full generation completes on Iris within ~an hour, written to GCS and
   published to the HF bucket with a README.

## Results

### Stage A — selection (done)

afdb-24M: **24,009,002** structures across **1,679,067** structural clusters.
After the `seq_len ∈ [2, 2000]` pre-filter and the `<3`-member drop:

- **960,054 clusters kept** (718,997 dropped as too small).
- **4,213,203 documents selected** → 2,111 manifest shards.

| split | round 0 | round 1 | round 2 | round 3 | round 4 |
|---|--:|--:|--:|--:|--:|
| train | 941,028 | 941,028 | 941,028 | 719,519 | 587,079 |
| val | 9,558 | 9,558 | 9,558 | 7,316 | 5,964 |
| test | 9,468 | 9,468 | 9,468 | 7,248 | 5,915 |

`round-0 == round-1 == round-2` holds in every split (the issue's
constraint), and the manifest is physically ordered round-4 → round-0.
Counts: [`data/selection_counts.csv`](data/selection_counts.csv) /
[`data/selection_stats.json`](data/selection_stats.json). Per-core
generation throughput measured at **4.45 docs/s/core** (~225 ms/structure),
so ~4.2 M docs ≈ **~31 min on 512 Iris workers**.

### Stage B — generation

_(Fill in after the Iris run: wall-clock + worker count, output paths,
W&B/Iris job ids, any dropped-row counts.)_

## Conclusion

_(Fill in once results are in.)_
