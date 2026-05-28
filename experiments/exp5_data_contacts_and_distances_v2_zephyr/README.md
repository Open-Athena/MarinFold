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

This is a fresh implementation. It calls out to nothing in exp34 at
runtime; the byte-identity contract is enforced by a test that
imports exp34's reference via `sys.path` and SHA-1-compares the
generated docs.

### Design choices

- **Columnar in-memory structure** (`parse.ParsedStructure`).
  Per-residue arrays at the *structure* level: `plddt_per_residue:
  float64[N]`, `cb_or_ca_xyz: float64[N, 3]`, plus a flat CSR atom
  table (`atom_offsets: int32[N+1]`, `atom_name_id: uint8[T]`,
  `atom_xyz: float64[T, 3]`). No backward compat with marinfold's
  tuple-of-tuples `Residue.atoms` — designed to round-trip zero-copy
  to `pyarrow.RecordBatch` if a precomputed-store experiment ever
  needs it.
- **Single-pass parser** with the gemmi fast path: `Model.all()` flat
  CRA iteration + `atom.pos.tolist()` (one nanobind call vs three
  attribute accesses). Polymer-iteration fallback for non-AFDB
  inputs. See `parse.py` for the determinism contract that keeps docs
  byte-identical to exp34.
- **Vectorized contact eligibility.** `np.triu_indices(k=1)` over the
  `cb_or_ca_xyz` array replaces a ~1.7 M-call Python `euclidean`
  loop. Row-major order is preserved so `rng.sample` over each
  contacts-by-mode list sees identical inputs to exp34.
- **URI-mode input is the default.** `--cif-uri-column` defaults to
  `gcs_uri`. The bulky `cif_content` path is still available via
  `--cif-text-column` for testing or datasets without URI columns.
- **In-shard concurrent fetches** via `Dataset.map_shard` +
  `ThreadPoolExecutor` (`--fetch-concurrency`, default 32). This is
  the new lever — the prior production run landed at ~128 s/shard
  because the 2,000 per-row GCS GETs ran sequentially. The fetch
  backend is `fsspec` + `gcsfs`; see "On obstore" below for why we
  didn't switch.
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

### On obstore (and why we didn't switch)

We evaluated [obstore](https://developmentseed.org/obstore/) (Rust
`object_store`-backed, advertised ~9× throughput vs `fsspec` on
parallel small-object GETs) as the per-row fetch backend. It works on
single-region GCS buckets like our output bucket
`gs://marin-us-central1/...`, but it **cannot read AFDB** as shipped
(obstore 0.9.5). The root cause is upstream: `object_store`'s HTTP
handler hard-requires a `Content-Length` header on every GCS GET
response, and AFDB cifs are stored with `Content-Encoding: gzip` so
GCS transparently decompresses on the wire (visible in the response
as `x-guploader-response-body-transformations: gunzipped`) and omits
`Content-Length`. The documented client-side workaround
(`Accept-Encoding: gzip`) can only be set via
`client_options.default_headers` today, and those headers leak into
the OAuth2 token fetch and break it.

Variants we tried (all fail identically on AFDB):

* `GCSStore.from_url` / `GCSStore(bucket=…)` / `.get_range` /
  `.stream`,
* `client_options={"http1_only"/"http2_only"/"allow_http"}`,
* `client_options={"default_headers": {"Accept-Encoding": "gzip"}}`
  — breaks the token fetch,
* static `credential_provider` + `default_headers` — Accept-Encoding
  doesn't reach the GCS GET,
* `obstore.fsspec.register("gs")` + `fsspec.open(...)`,
* `FsspecStore("gs", ...)`.

Until either `object_store` relaxes the `Content-Length` invariant or
obstore exposes per-request / per-bucket header overrides that don't
touch the auth flow, we stay on `fsspec/gcsfs` for the fetch.
Worth filing upstream the next time someone touches obstore.

### Layout

```
experiments/exp5_data_contacts_and_distances_v2_zephyr/
├── vocab.py        # re-exports marinfold v1 vocab + appends V2_NEW_TOKENS
├── parse.py        # columnar ParsedStructure + gemmi fast-path extractor
├── generate.py     # vectorized v2 generation; byte-identical RNG stream vs exp34
├── cli.py          # `generate` (Zephyr pipeline) + `tokenizer` subcommands
├── pyproject.toml  # marinfold + gemmi/numpy/pyarrow + marin-zephyr/fsspec/gcsfs/hf
└── tests/
    ├── test_parse.py          # columnar shape + per-residue invariants
    ├── test_byte_identity.py  # SHA-1 match against exp34 reference
    └── test_cli.py            # end-to-end cli.cmd_generate smoke
```

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

1. **Byte-identity vs exp34.** Generated v2 docs are SHA-1-equal to
   exp34's reference implementation for the same structure + entry_id,
   across multiple seeds — covered by
   `tests/test_byte_identity.py::test_doc_byte_identical_to_exp34`.
2. **Full 1.6M-structure generation completes** on the marin Iris
   cluster, writing one parquet per input shard to GCS, with no shard
   left unwritten and no `ZephyrWorkerError` propagating.
3. **Wall-clock improves over the prior runs** — the v2 production
   runs from the exp34 session landed at ~14 min (sequential
   `cif_content`) and ~8 min (`gcs_uri` mode without intra-shard
   concurrency). With the thread-pool fetches the per-shard time
   should drop below the prior ~128 s, and end-to-end wall should
   come in well under ~5 min compute.

## Results

_(Fill in after the at-scale run completes.)_

## Conclusion

_(Fill in once results are in.)_
