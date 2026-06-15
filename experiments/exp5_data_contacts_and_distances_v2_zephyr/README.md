---
marinfold_experiment:
  issue: 5
  title: "implement data generation on zephyr"
  kind: data
  branch: u/alxmrs/exp5
---

# implement data generation on zephyr

**Issue:** [#5](https://github.com/Open-Athena/MarinFold/issues/5) · **Kind:** `data` · **Branch:** `u/alxmrs/exp5`

## Question

Can we run the v2 training-doc generator (`contacts-and-distances-v2`)
at scale on the marin Iris cluster via Zephyr, ingesting the
[`timodonnell/afdb-1.6M`](https://huggingface.co/datasets/timodonnell/afdb-1.6M)
manifest and producing per-shard parquet outputs to GCS?

## Hypothesis

Yes — with the right tuning. Prior runs in exp34 surfaced three
specific levers that, together, should land 1.6M structures in
single-digit minutes:

1. Don't ship the bulky `cif_content` cross-cloud. The manifest has a
   `gcs_uri` column pointing at the public AFDB GCS bucket — parquet
   is columnar, so reading just `[entry_id, gcs_uri]` is ~160 KB/shard
   instead of ~70 MB.
2. Per-shard I/O is dominated by ~2,000 small in-region GCS GETs.
   Concurrent fetches inside a shard (a thread pool around the per-row
   parse) overlap the GETs with gemmi's parse, which releases the GIL
   during the C++ tokenize.
3. The per-shard CPU is *single-threaded* in Python — gemmi's
   per-attribute access dominates. Honor that by sizing
   `--worker-cpu 1` and scaling out via `--max-workers` rather than
   over-allocating per-worker cores.

## Background

- Issue [#5](https://github.com/Open-Athena/MarinFold/issues/5) is the
  Zephyr-runner request. The dataset target was originally v1, but the
  v2 endorsement is in [this comment](https://github.com/Open-Athena/MarinFold/issues/5#issuecomment-4557947163):
  *"We have a new document type, `contacts-and-distances-v2` that we
  can use as a test for running document generation on zephyr."*
- Document spec + the canonical v2 generation algorithm:
  [`exp34_document_structures_contacts_and_distances_v2`](../exp34_document_structures_contacts_and_distances_v2/).
  exp34 stays untouched (per the PR #19 review); exp5 is the at-scale
  Zephyr runner that re-implements the spec from scratch for
  performance, with a byte-identity test pinning the contract.
- The "no bulk-numpy extraction in gemmi's Python binding" finding —
  and the maintainer's `pos.tolist()` + `Model.all()` workaround —
  comes from [project-gemmi/gemmi#314](https://github.com/project-gemmi/gemmi/issues/314).
  Both optimizations are folded into `parse.py`.

## Approach

**exp5 is a Zephyr wrapper around exp34's local impl.** The v2 doc
algorithm — parse + generate + vocab — comes from
[`exp34`](../exp34_document_structures_contacts_and_distances_v2/),
imported directly via a `sys.path` shim in `cli.py`. Two copies of an
algorithm always drift; one source of truth doesn't (see Tim's review on
PR [#38](https://github.com/Open-Athena/MarinFold/pull/38)). The
trade-off is that we accept exp34's per-row CPU cost as-is rather than
re-implementing for speed — the prior production runs landed at ~8 min
end-to-end on the cluster, well within budget for a one-time job.

Performance-oriented re-engineering of the parse + generate inner
loop — and the columnar `ParsedStructure` + CSR substrate that comes
with it — lives in
[`exp42`](../exp42_data_shared_protein_data_substrate_csr_parquet/), where
training-time on-the-fly doc generation actually needs the speedup.
exp5 stays small.

### Design choices

- **One source of truth.** `cli.py` does `import parse, generate, vocab`
  — all three modules resolve to exp34 via a `sys.path` shim at module
  load (exp34 is `package = false` so we can't path-dep it as an
  installable package; the long-term fix is to graduate v2 to
  `marinfold/marinfold/document_structures/contacts_and_distances_v2/`,
  at which point the shim disappears).
- **Per-row I/O adapter.** exp34's `parse.parse_structure(path)` takes
  a local filesystem path; the manifest gives us a `gs://` URI. The
  worker fetches via `fsspec` into a `NamedTemporaryFile`, then calls
  the parser. exp34's parser derives `entry_id` from the cif's
  `_entry.id` field, so the random tempfile name is harmless. Per-row
  cost: one tempfile open/write/unlink, ~negligible.
- **URI-mode input is the default.** `--cif-uri-column` defaults to
  `gcs_uri`. The bulky `cif_content` path is still available via
  `--cif-text-column` for testing or datasets without URI columns.
- **In-shard concurrent fetches** via `Dataset.map_shard` +
  `ThreadPoolExecutor` (`--fetch-concurrency`, default 32). The
  prior production run landed at ~128 s/shard because the 2,000 per-row
  GCS GETs ran sequentially. Threading them overlaps GETs with gemmi
  parse (gemmi releases the GIL during the C++ parse).
- **Resource defaults bake in the lesson:** `--worker-cpu 1`,
  `--worker-memory 4g`, `--max-workers` left to Zephyr's default of
  128 on a cluster (raise via the flag or `ZEPHYR_MAX_WORKERS`).
- **`--num-docs N` is a global cap on emitted documents**, collapses
  to a single output file. For sharded output use a `{shard}`
  placeholder in `--out` and drop `--num-docs`.

### Output schema

Every output row carries provenance so a generated doc can be traced
back to its source structure and audited against the published
reference dataset. Always present:

| column | type | source |
|---|---|---|
| `entry_id` | string | passthrough from input manifest |
| `structure` | string | constant — the doc-structure name `"contacts-and-distances-v2"` |
| `document` | string | the v2 doc text |
| `sha1` | string | `sha1(document.encode())` — matches the `sha1` column in `timodonnell/protein-docs/contacts-and-distances-v1-5x`, so byte-equality with a published reference is one column compare |
| `seq_len` | int | number of residues the doc actually serialized (parsed-structure value, not the manifest's) |
| `global_plddt` | float | mean pLDDT used to pick the doc's pLDDT-bin token (parsed-structure value) |
| `contacts_emitted` | int | count of `<long-range-contact>` + `<medium-range-contact>` + `<short-range-contact>` tokens in the doc |

Opportunistic passthrough — any of the following manifest columns
that exist in the input parquet schema are copied verbatim onto
every row (afdb-1.6M carries them all; minimal test manifests
typically don't):

`split`, `seq_cluster_id`, `struct_cluster_id`, `uniprot_accession`,
`tax_id`, `organism_name`, `gcs_uri`.

`gcs_uri` is special: when `--cif-uri-column='gcs_uri'` (the default)
it's *both* the data source and a passthrough column, so the manifest
read deduplicates it before calling `load_parquet(columns=...)`.

Schema detection happens once at submission time (peek the first
matching parquet file), so the output schema is stable across shards.

### See also: on-the-fly training-time generation (#42)

For training-time doc generation directly from the parsed substrate —
with per-epoch RNG variation as free data augmentation, ~12× smaller
artifact than raw CIFs, and a doc-format-agnostic dataloader callback
API — see
[`exp42`](../exp42_data_shared_protein_data_substrate_csr_parquet/) (the
"shared protein-data substrate" experiment). exp5 publishes the static
v2 corpus to HF; exp42 publishes a CSR-parquet substrate that any
doc-format experiment can consume on the fly.

### Layout

```
experiments/exp5_data_contacts_and_distances_v2_zephyr/
├── cli.py          # `generate` (Zephyr pipeline) + `tokenizer` subcommands.
│                   # Imports parse/generate/vocab from exp34 via sys.path shim.
├── pyproject.toml  # marinfold + gemmi/numpy/pyarrow + marin-zephyr/fsspec/gcsfs/hf
└── tests/
    └── test_cli.py            # end-to-end cli.cmd_generate smoke
```

Notably absent: `parse.py`, `generate.py`, `vocab.py` — all three live
in [`exp34`](../exp34_document_structures_contacts_and_distances_v2/)
and are imported at runtime. exp5's job is the Zephyr pipeline around
those modules, nothing more.

### Running it locally

```bash
uv sync --extra test

# Build / save / push the v2 tokenizer.
uv run python cli.py tokenizer --save-local /tmp/tok-v2/

# Run the full test suite (incl. byte-identity vs exp34).
uv run pytest tests/ -v
```

### Generating on Iris

In a dedicated terminal, connect to the cluster:

```bash
uv run iris --cluster=marin cluster dashboard
```

**Smoke test** — a single parquet shard from the manifest, capped to 100
docs, single output file:

```bash
uv run iris --cluster=marin job run --cpu 1 --memory 2GB -- python cli.py generate --input "hf://datasets/timodonnell/afdb-1.6M/shard_000-999/shard_000000.parquet" --num-docs 100 --out "gs://marin-tmp-us-central1/marin-fold-tests/exp5-smoke.parquet" --worker-cpu 1 --worker-memory 4g --fetch-concurrency 32
```

**Full run** — all 1.6M structures, one output parquet per input shard:

```bash
uv run iris --cluster=marin job run --cpu 1 --memory 2GB -- python cli.py generate --input "hf://datasets/timodonnell/afdb-1.6M/**/*.parquet" --out "gs://marin-us-central1/protein-structure/MarinFold/exp5/corpus_v2-{shard:05d}-of-{total:05d}.parquet" --worker-cpu 1 --worker-memory 4g --worker-disk 64g --max-workers 512 --fetch-concurrency 32
```

Each command must stay on **one line** — a trailing space after a
backslash silently truncates the command and the rest leaks to your
shell. Cancel a running job with
`uv run iris --cluster=marin job stop <JOB_ID>` (find IDs via
`iris --cluster=marin job list`).

## Success criteria

1. **One source of truth for v2.** Generated docs come from exp34's
   `parse.parse_structure` + `generate._generate_one` (no fork, no
   re-implementation). Verifiable by `git grep -rn 'def parse_structure\|def _generate_one'`
   returning hits only under `exp34_*/`.
2. **Full 1.6M-structure generation completes** on the marin Iris
   cluster, writing one parquet per input shard to GCS, with no shard
   left unwritten and no `ZephyrWorkerError` propagating.
3. **Wall-clock within budget** — the v2 production runs from the
   exp34 session landed at ~14 min (sequential `cif_content`) and ~8
   min (`gcs_uri` mode without intra-shard concurrency). The
   thread-pool fetches should keep the per-shard time well under the
   prior ~128 s. We're explicitly *not* targeting the 5-min number
   from the earlier perf-focused branch — that speedup required the
   forked parse/generate, which we've intentionally dropped per
   @timodonnell's PR [#38](https://github.com/Open-Athena/MarinFold/pull/38)
   review.

## Results

_(Fill in after the at-scale run completes.)_

## Conclusion

_(Fill in once results are in.)_
