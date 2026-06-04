# exp53 — handoff / where to pick up

Status snapshot for resuming in a fresh session. Branch:
`exp/53-contacts-v1-zephyr`. Everything through the **local** validation is
done and committed; the remaining work is the **on-cluster** run + publish.

## Done

- **Stage A (selection)** — `selection.py` + `fetch_manifest_columns.py`, unit
  tested (6 tests). Ran on the full afdb-24M: **24,009,002 structures /
  1,679,067 clusters -> 960,054 clusters kept (>=3 members) -> 4,213,203
  documents** across 2,111 manifest shards. `round-0 == round-1 == round-2`
  per split; shards physically ordered round-4 -> round-0. Counts committed in
  `data/selection_{counts.csv,stats.json}`.
- **Stage B (generation)** — `generate_rows.py` (worker) + `cli.py` (Zephyr
  driver), tested (2 tests, incl. **byte-identity** vs a direct
  `generate_document` call). Validated end-to-end locally under
  `ZephyrContext` via the inline-cif path (`records_out: 20`, valid docs).
  Measured **4.45 docs/s/core** (~225 ms/structure).
- **Cluster reachability** — `iris --cluster=marin` connects (after pinning
  `httpx<1`; the 1.0 prerelease breaks marin-iris). Local GCS write to
  `gs://marin-us-east5` works with `GOOGLE_CLOUD_PROJECT=hai-gcp-models`.

## Local scratch artifacts (NOT in git — on this machine)

Both live under `~/exp53_scratch/` and are regenerable:

- `~/exp53_scratch/afdb24m_small/` — 12,005 per-shard parquet of afdb-24M's
  small columns. Regenerate (~24 min, resumable):
  `uv run python fetch_manifest_columns.py --out ~/exp53_scratch/afdb24m_small`
- `~/exp53_scratch/selection_manifest/` — the 2,111-shard round manifest
  (`train/`, `val/`, `test/`), i.e. **Stage B's input**. Regenerate (~13 s):
  `uv run python selection.py --input ~/exp53_scratch/afdb24m_small --out ~/exp53_scratch/selection_manifest`

## Resume here (next steps)

Prereqs: `cd experiments/exp53_data_contacts_v1_zephyr && uv sync --extra test`
and `export GOOGLE_CLOUD_PROJECT=hai-gcp-models` for local GCS writes.

**1. Upload the manifest to GCS** (Iris workers read it from there):
```python
# uv run python - <<'PY'
import os, gcsfs, glob
fs = gcsfs.GCSFileSystem(project="hai-gcp-models")
src = os.path.expanduser("~/exp53_scratch/selection_manifest")
dst = "marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/selection_manifest"
for p in glob.glob(src + "/**/*.parquet", recursive=True):
    fs.put(p, f"{dst}/{os.path.relpath(p, src)}")
PY
```

**2. Iris smoke (100 docs)** — confirms the on-cluster gcs_uri fetch (can't be
tested locally: the AFDB bucket is requester-pays and local user creds give
truncated reads; Iris workers' service account reads fine — exp5 proved this):
```bash
uv run iris --cluster=marin job run --cpu 1 --memory 2GB -- \
  python cli.py generate \
    --input "gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/selection_manifest/train/shard_00000.parquet" \
    --num-docs 100 \
    --out "gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/_smoke/out-{shard:05d}.parquet" \
    --worker-cpu 1 --worker-memory 4g --fetch-concurrency 32
```
Verify the output is non-empty + docs valid; note the on-cluster per-doc time.

**3. GATE** — report measured rate + worker count + projected wall-clock, then
get @timodonnell's go-ahead before the full run (decision #3).

**4. Full run** — once per split (`train`, then `val`, `test`):
```bash
uv run iris --cluster=marin job run --cpu 1 --memory 2GB -- \
  python cli.py generate \
    --input "gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/selection_manifest/train/shard_*.parquet" \
    --out "gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/documents/train/contacts_v1-{shard:05d}-of-{total:05d}.parquet" \
    --worker-cpu 1 --worker-memory 4g --max-workers 512 --fetch-concurrency 32
```
~4.2 M docs -> ~31 min @ 512 workers (resize from the smoke rate).

**5. Publish** — copy GCS `documents/<split>/` -> HF bucket
`open-athena/MarinFold` at `data/document_structures/contacts_v1/<split>/`
(round-descending order preserved) + a dataset README; push the tokenizer
(`uv run contacts-v1 tokenizer --push open-athena/contacts-v1-tokenizer`);
add a `history/runs/*` entry + `data/timings.csv`; open the PR; comment on the
issue with a robot emoji prefix (do NOT close the issue).

## Gotchas / facts (so you don't rediscover them)

- **HF API rate limit** (3,000 req / 5 min): never read afdb-24M via per-file
  `hf://` (one `paths-info` call/shard). `fetch_manifest_columns.py` uses an
  authenticated `HfFileSystem` + pre-warmed dircache + `/resolve/` CDN reads.
- **httpx must be `<1`** — marin-iris's GCP provider calls
  `httpx.Client(timeout=)`, removed in the 1.0 prerelease. Pinned in
  `pyproject.toml` (base deps).
- **duckdb `hf://`** uses `con.register_filesystem(fsspec.filesystem("hf"))`
  (the bundled `httpfs` extension 404s for this duckdb build).
- **AFDB GCS bucket is requester-pays** -> local reads truncate without a
  billing project; Iris workers read fine. So Stage B's gcs_uri path is only
  validatable on-cluster.
- **`iris job list`** unfiltered errors (`offset>5000`); use state/name filters.
- **HF bucket** `open-athena/MarinFold` is an HF *bucket* (not a normal
  dataset/model repo — both 404 as those types). Work out the upload path at
  publish time (see the `hf-cli` skill).
- **marin-zephyr** is in **base** deps (not an extra) so Iris workers get the
  runtime however the job ships its env (matches exp5). Pinned to ref
  `alxmrs/stamp-iris-build-date`; revisit against the live cluster if it moved.

## Decisions (resolved with @timodonnell)

1. GCS canonical copy -> `gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/` (AGENTS.md; not exp5's us-central1).
2. Round-0: **drop** reps that fail to generate; no next-best-pLDDT backfill.
3. Build + smoke + PR, then **pause for go-ahead** before the full run.
