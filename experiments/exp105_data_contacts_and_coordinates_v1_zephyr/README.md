---
marinfold_experiment:
  issue: 105
  title: "exp: generate contacts-and-coordinates-v1 training set"
  kind: data
  branch: claude/affectionate-montalcini-544d89
---

# exp105 — generate the contacts-and-coordinates-v1 dataset on zephyr

**Issue:** [#105](https://github.com/Open-Athena/MarinFold/issues/105) · **Kind:** `data`

## Question

Can we generate the
[`contacts-and-coordinates-v1`](../../marinfold/marinfold/document_structures/contacts_and_coordinates_v1)
training corpus at scale — **for the same proteins as the contacts-v1
training set** — on the marin Iris cluster via Zephyr, and publish it
(sharded parquet + tokenizer + README) to the
[`open-athena/MarinFold`](https://huggingface.co/buckets/open-athena/MarinFold)
HF bucket, **calling into the existing generator rather than
re-implementing it**?

## Approach

The format and its generator land in a sibling PR (the
`contacts_and_coordinates_v1` package). This experiment is the data run.

**Same proteins as contacts-v1.** contacts-v1's corpus (exp53) was selected
from `afdb-24M` into a *selection manifest* — one row per selected
`(entry, round)` record (≤5 per structural cluster, `seq_len ∈ [2, 2000]`,
round-labeled by pLDDT), with a `gcs_uri` and provenance columns. We reuse
that manifest **verbatim**, so there is no new selection stage: identical
proteins, identical rounds/splits.

```
gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/selection_manifest/{train,val,test}/shard_*.parquet
```

**One stage (generation).** For each manifest row, `generate_rows.py`
fetches the structure from its `gcs_uri` (gzip-safe `read_object_bytes`),
parses it with gemmi, and calls
`contacts_and_coordinates_v1.generate_document(...)` with the manifest
`entry_id` as the deterministic seed. The output row is the format's
`metadata_row()` (document + coordinate metadata) plus the manifest
provenance columns (`round`, `struct_cluster_id`, `split`, …). `cli.py`
wraps this in a Zephyr `map_shard`, reusing exp53's performance decisions
(see the [`zephyr-pipeline-performance`](../../.agents/skills/zephyr-pipeline-performance/SKILL.md)
skill): `thread_per_row_in_shard` overlaps the GCS GETs with pyconfind,
1 CPU/worker scaled via `--max-workers`, region-pinned + preemptible, and
fail-loud with per-row lenient drops only for designed-in filters
(multi-chain, out-of-range, too large for the coordinate cube).

## Local pilot (cost + size estimate)

`local_pilot.py` generates documents off-cluster (fetching the same
AlphaFold models from EBI's public HTTPS endpoint, since the manifest's
`gcs_uri` is requester-pays / on-cluster only) so we can eyeball real
documents and project the run before committing.

The key finding is **structure-independent**: coordinate mention events are
sampled *with replacement until the token budget fills*, so **every
document reaches ~32,766 of the 32,768-token budget** regardless of chain
length (measured 32,764–32,768 for L = 5 … 2000; 1QYS / 92 residues =
32,767 tokens, 7,415 events). That makes the corpus projection robust and
independent of the protein size distribution:

| quantity | per doc | × 4,213,203 docs (exp53 selection) |
|---|---|---|
| tokens | ~32,766 | **~138 B tokens** |
| parquet (zstd) | ~54 KB | **~229 GB** |
| raw document text | ~265 KB | ~1.1 TB |

For comparison, contacts-v1 (exp53) was ~4.67 B tokens — this corpus is
**~30× larger in tokens** because the 32768-token budget (4× contacts-v1's
8192) is filled by coordinates rather than left mostly empty. Decision
(issue #105): proceed to the **full ~4.2M-protein run**; the numbers above
are the storage/compute budget to expect.

_(The pilot's per-doc wall-clock and the requester-pays fetch are validated
on-cluster at the iris-smoke step below, not locally — the sandbox this was
drafted in can't reach the AFDB bucket.)_

## Running it

> **The submitting client must be fresh (<14 days).** The live controller
> rejects *root* `LaunchJob` submissions whose `marin-iris` client is older
> than 14 days (`_check_client_freshness`, marin PR #5108). The frozen
> `marin-*-latest` wheels this project pins (2026-05-29) are past that floor,
> so **do not** submit with `uv run iris` from this project's venv — it fails
> at submit with "marin-iris client is too old (build 2026-05-29; minimum …)".
> Submit instead with a fresh **editable** marin-iris (an editable install
> reports its git-commit date as the client revision). On this workstation a
> recent editable checkout lives at `/home/bizon/git/marin`, so set
> `IRIS=/home/bizon/git/marin/.venv/bin/iris` and confirm freshness first:
> `"$IRIS" ...` where
> `python -c "from iris.version import client_revision_date; print(client_revision_date())"`
> must be ≥ today−14 (if stale, `git -C /home/bizon/git/marin pull` a branch
> whose `lib/iris` was touched recently, or make a fresh editable checkout).
> Worker/child tasks are **exempt** from the gate, so the worker env keeps the
> frozen wheels (see `pyproject.toml`). One cosmetic consequence: zephyr's
> dashboard status-text push 404s against the current controller and logs
> non-fatal "Failed to report task status text" warnings — the pipeline still
> runs and writes correct output (smoke-verified 2026-07-08).

Below, `$IRIS` is that fresh editable client, run from this experiment
directory (so the 0.1 MB workspace bundle = `cli.py` + `generate_rows.py` +
`pyproject.toml`).

**0. Tokenizer** (once; needed by trainers, not by generation). Build locally;
it is published **into the HF bucket** next to the data in step 3. (The
`open-athena/…-tokenizer` *model repo* pattern is superseded — corpus
tokenizers live in the bucket, and `open-athena` model-repo creation 403s.)

```bash
python -m marinfold.document_structures.contacts_and_coordinates_v1.cli \
    tokenizer --save-local ./tokenizer
```

**1. Iris smoke** — 100 docs from one train shard, single output file.
Validates the requester-pays fetch and measures per-doc latency:

```bash
"$IRIS" --cluster=marin job run --cpu 1 --memory 2GB -- \
  python cli.py generate \
    --input "gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/selection_manifest/train/shard_00000.parquet" \
    --out   "gs://marin-us-central1/protein-structure/MarinFold/exp105_ccoord_v1/_smoke/out.parquet" \
    --num-docs 100 --region us-central1 --preemptible
```

**2. Full run** — once per split. `{shard}` in `--out` writes one output
file per input shard (preserving the round-descending order):

```bash
for split in train val test; do
  "$IRIS" --cluster=marin job run --cpu 1 --memory 2GB -- \
    python cli.py generate \
      --input "gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/selection_manifest/${split}/shard_*.parquet" \
      --out   "gs://marin-us-central1/protein-structure/MarinFold/exp105_ccoord_v1/documents/${split}/ccoord_v1-{shard:05d}-of-{total:05d}.parquet" \
      --worker-cpu 1 --worker-memory 4g --max-workers 512 --fetch-concurrency 32 \
      --region us-central1 --preemptible
done
```

**3. Publish** the `documents/` tree + `tokenizer/` + a dataset README to
`buckets/open-athena/MarinFold/data/document_structures/contacts_and_coordinates_v1/{train,val,test}/`
(mirrors the contacts-v1 layout from exp53).

> Output bucket is `gs://marin-us-central1` (region-local to the Iris
> cluster, controller zone `us-central1-a`) so per-row PUTs aren't
> cross-region; the input manifest stays where exp53 wrote it (`us-east5`).

## Success criteria

A new contacts-and-coordinates-v1 training set on HuggingFace for the same
~4.2M proteins as contacts-v1 (train/val/test), byte-identical to calling
`generate_document` directly, with 0 generation drops beyond the designed-in
filters.

## Results

_(Fill in after the run: doc counts per split, drop counts, total tokens,
wall-clock, published HF path.)_

## Conclusion

_(Fill in after results are in.)_
