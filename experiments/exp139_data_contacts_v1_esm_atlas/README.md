---
marinfold_experiment:
  issue: 139
  title: "exp: generate contacts-v1 corpus from the ESMFold2 Atlas distillation set (67M) + save reusable pyconfind contacts"
  kind: data
  branch: claude/esm-atlas-contacts-v1-gen-db9624
---

# exp: generate contacts-v1 corpus from the ESMFold2 Atlas distillation set (67M) + save reusable pyconfind contacts

**Issue:** [#139](https://github.com/Open-Athena/MarinFold/issues/139) · **Kind:** `data` · **Branch:** `claude/esm-atlas-contacts-v1-gen-db9624`

## Question

Generate the [`contacts-v1`](https://github.com/Open-Athena/MarinFold/blob/main/marinfold/marinfold/document_structures/contacts_v1/SPEC.md) training corpus **at scale from the ESMFold2 Atlas distillation set** ([`open-athena/esm-atlas-esmfold2-distill`](https://huggingface.co/buckets/open-athena/esm-atlas-esmfold2-distill), the #91 curation — 66.76M single-chain monomers, ~2.08 TB) and publish it to the [`open-athena/MarinFold`](https://huggingface.co/buckets/open-athena/MarinFold) HF bucket — calling straight into the format's generator, no re-implementation. In the **same pyconfind pass**, also publish the **raw pyconfind contacts** as a reusable, doc-format-agnostic intermediate so future document types (e.g. `contacts-and-crops-v1`) can be generated from this source **without re-running pyconfind**.

This is the "67M proteins instead of 4M" training-corpus expansion flagged in `UPDATES.md` (#91) — ~16× the AFDB `contacts-v1` corpus (exp53).

## Hypothesis

_(Copy from the issue.)_

## Background

- **Source** ([`open-athena/esm-atlas-esmfold2-distill`](https://huggingface.co/buckets/open-athena/esm-atlas-esmfold2-distill)): an HF **bucket** (repoType `bucket`), 3,338 `structures/parts/part_*.parquet` (~20k structures each) + `selected_manifest.parquet` + README; **2.08 TB**. Per-row schema has **inline `cif_content`** (mmCIF text, pLDDT in B-factor), `entry_id`, `sequence`, `seq_len`, `global_plddt`, `ptm`, `plddt_std`, `seq_cluster_id`, `cluster_size`, `split` (always `train`), `source`. These are already the deduplicated cluster representatives (40% identity linclust) — no cluster/round selection needed.
- **Two structural differences from AFDB (exp53/exp105):** (1) the cif is **inline** (`cif_content`), not a per-row `gcs_uri` pointer — so exp53's "source from a URI, not inline" perf rule is reversed here; (2) the source lives on **HF storage (CloudFront/S3), not GCS** — so any read into GCP is cross-cloud. `bucket` metadata reports `cdnRegions: []`.
- **Generator seam** (`marinfold/document_structures/contacts_v1/`): `parse.analyze_structure(structure)` runs pyconfind (the expensive step) → `AnalyzedStructure(entry_id, residues, contacts, global_plddt)`; `generate.build_document(entry_id, residues, contacts, …)` is a pure, pyconfind-free serializer. `contacts` are the **raw** pyconfind side-chain contacts (`seq_i < seq_j`, `degree`), before the per-doc `min_seq_separation` / `min_contact_degree` filters — i.e. exactly the reusable primitive. This split lets us run pyconfind **once** and emit both the document and the saved contacts.
- **Fan-out template:** exp53 (`_zephyr`) + exp105 (`_zephyr`, the better region-handling reference) — `Dataset.from_files(input).load_parquet(cols).map_shard(worker).write_parquet(out)` under `ZephyrContext(ResourceConfig(cpu, ram, disk, regions, preemptible))`, launched via `iris --cluster=marin job run --cpu 1 --memory 2GB -- python cli.py …`. See `.agents/skills/zephyr-pipeline-performance/SKILL.md`.
- **Region facts** (`AGENTS.md` §Cross-region): Iris CPU workers land in **us-central1** (controller in `us-central1-a`); co-locate GCS output with the zone workers land in; the marin cluster has **no preemptible CPU scale group** (exp105 lesson → use `--no-preemptible`); a >10 GB cross-region copy needs explicit sign-off.

## Approach

Everything runs in **us-central1** so there are no GCP inter-region hops; the only cross-cloud transfer is the one-time HF→GCS mirror.

**Stage 0 — mirror the source into the compute region (one-time).**
Cloud-side copy `structures/parts/part_*.parquet` (+ `selected_manifest.parquet`) from the HF bucket into `gs://marin-us-central1/protein-structure/MarinFold/exp<N>_esm_atlas_contacts_v1/source/` on an **iris pod** (per the cloud-side-mirror recipe: `uv run --with`, `download_bucket_files`, `gcsfs`, org token + ~6 workers for 429s) so the ~2.08 TB moves at cloud bandwidth, not the ~2.5 MB/s workstation uplink. This mirror is **reused** by future doc types (crops) that need coordinates from the cif.

**Stage 1 — generate on iris CPUs (Zephyr, us-central1).**
Adapt exp105's `cli.py` + a `generate_rows.py` worker (exp53 already ships an **inline-cif path** — `cif_text_column`). Per structure, in **one pyconfind pass**:
- `analyze_structure(gemmi.read_structure_string(cif_content))` → `AnalyzedStructure`;
- serialize the raw contacts intermediate (residues `resname/resnum/chain` arrays + contacts `seq_i/seq_j/degree` arrays + `global_plddt`), **and**
- `build_document(...)` → the contacts-v1 document + `metadata_row()`.

Emit **one combined "analyzed" row** per structure = `document` + doc metadata + serialized residues/contacts + provenance (`entry_id`, `seq_cluster_id`, `ptm`, `plddt_std`, `cluster_size`, `source`). `map_shard` over parts, **one output parquet per input part**, `ZephyrContext(resources=ResourceConfig(cpu=1, ram="4g", disk="32g", regions=["us-central1"], preemptible=False))`, scale via `--max-workers`. **Scope:** flat 1:1 — one doc per structure, all `train`; the only drops are the generator's designed filters (`seq_len ∉ [2, 2000]`, multi-chain, parse failure), counted and reported by reason (lenient `None`, no backfill). Output → `gs://marin-us-central1/.../exp<N>_esm_atlas_contacts_v1/analyzed/part_*.parquet`.

**Stage 2 — library helper + publish.**
- **(Landed)** A small reusable `AnalyzedStructure` (de)serializer in `marinfold` (`analyzed_to_row` / `analyzed_from_row`, row-of-arrays ↔ `AnalyzedStructure`) so `contacts-and-crops-v1` (and others) can reconstruct residues + contacts and call `build_document`/its own builder **without pyconfind**. Round-trip unit test proves `analyzed_from_row(analyzed_to_row(x))` rebuilds a byte-identical document vs a direct build.
- Publish to `open-athena/MarinFold`:
  - **Reusable contacts:** the full `analyzed/` set → `data/contacts/esm_atlas_esmfold2_distill/` (raw pyconfind contacts + residues + provenance) with a README recording the exact pyconfind geometry (`native_only=True`, `contact_distance=3.0`, `dcut=25.0`, `clash_distance=2.0`, Dunbrack-2010 rotamer lib) and rotamer-library version.
  - **Documents corpus:** a documents-only column projection → `data/document_structures/contacts_v1_esm_atlas/train/` + `tokenizer/` + dataset README (mirrors the `contacts_v1/` / `contacts_v1_think/` layout).
- Add a `history/runs/*` entry (W&B if the run logs) + `data/timings.csv`; open the PR; comment the outcome on this issue with a robot-emoji prefix (do **not** close it).

**Pre-flight checks (smoke, before the full run):** confirm ESM Atlas cif parses with gemmi and is single-chain (contacts-v1 hard-skips multi-chain); confirm B-factor pLDDT scale; measure on-cluster docs/s/core and the seq_len>2000 drop rate on ~1–2 parts; then GATE for go-ahead before the full fan-out.

## Success criteria

- Source mirrored to `gs://marin-us-central1/.../source/` (3,338 parts; count/size verified).
- Generation complete with **0 unexpected drops** beyond the documented filters (drop counts reported by reason).
- Reusable contacts published to `data/contacts/esm_atlas_esmfold2_distill/` + documents to `data/document_structures/contacts_v1_esm_atlas/train/` + tokenizer + READMEs; a shard of each reads back **anonymously**.
- `AnalyzedStructure` (de)serializer landed with a round-trip byte-identity test, so a future crops corpus can be built from the saved contacts without pyconfind.
- All I/O stayed in us-central1 (no GCP inter-region hops); the one HF→GCS egress is the only cross-cloud transfer.

## Code & how to run

The pyconfind seam lives in `marinfold` (this branch adds the reusable serializer);
the experiment is three scripts + a worker.

| File | Role |
| --- | --- |
| `mirror_source.py` | **Stage 0** — mirror the HF bucket → `gs://marin-us-central1/…/source/` on an iris pod (per-file dl→GCS-put→rm, resumable, 429 backoff). |
| `generate_rows.py` | Per-row worker — one `analyze_structure` pass → one combined row (`analyzed_to_row` contacts **+** `build_document` document + provenance). No zephyr import; unit-tested. |
| `cli.py` | **Stage 1** — Zephyr `map_shard` driver; `iris job run` wraps it. Single-region us-central1, on-demand, one output parquet per input part. |
| `publish.py` | **Stage 2** — split the combined `analyzed/` set into the documents corpus + reusable contacts views and push to the HF bucket. |
| `marinfold …/contacts_v1/parse.py` | `analyzed_to_row` / `analyzed_from_row` + `ANALYZED_ROW_COLUMNS` — the reusable pyconfind-contacts record (byte-identity round-trip test in `marinfold/tests/.../contacts_v1/test_parse.py`). |

```bash
# Stage 0 — mirror (on an iris pod; the workstation uplink is far too slow for 2 TB).
#   DRY_RUN=1 first to print the plan; LIMIT=N to mirror only the first N parts (smoke).
/home/bizon/git/marin/.venv/bin/iris --cluster marin job run --no-wait \
    --enable-extra-resources --cpu=8 --memory=16GB --disk=32GB --zone=us-central1-a \
    -e MIRROR_WORKERS 6 -e HF_TOKEN "$HF_TOKEN" \
    -- uv run --with 'huggingface_hub>=1.5,<2' --with hf_xet --with gcsfs \
       python mirror_source.py

# Stage 1 — smoke (100 docs, one merged file) then GATE before the full run.
uv run iris --cluster=marin job run --cpu 1 --memory 2GB -- \
  python cli.py generate \
    --input "gs://marin-us-central1/protein-structure/MarinFold/exp139_esm_atlas_contacts_v1/source/structures/parts/part_00000.parquet" \
    --num-docs 100 \
    --out "gs://marin-us-central1/protein-structure/MarinFold/exp139_esm_atlas_contacts_v1/_smoke/out-{shard:05d}.parquet"

# Stage 1 — full fan-out (one output parquet per input part; single-region on-demand).
uv run iris --cluster=marin job run --cpu 1 --memory 2GB -- \
  python cli.py generate \
    --input "gs://marin-us-central1/protein-structure/MarinFold/exp139_esm_atlas_contacts_v1/source/structures/parts/part_*.parquet" \
    --out "gs://marin-us-central1/protein-structure/MarinFold/exp139_esm_atlas_contacts_v1/analyzed/analyzed-{shard:05d}-of-{total:05d}.parquet" \
    --max-workers 512
```

Submit Stage 1 with a **fresh editable iris** (`/home/bizon/git/marin/.venv/bin/iris`) if
the pinned PyPI `marin-iris` dev wheel has aged past the 14-day client-freshness floor.

## Status

**DONE** — corpus generated and published (2026-07-23).

## Results

Generated on **CoreWeave Genoa** (`cw-us-east-02a`, 4× `cd-gp-a192-genoa` =
384 physical EPYC 9454 cores, reserved/pinned) rather than the GCP marin pool —
see "Where it ran" below.

| | |
| --- | --- |
| Documents | **66,759,922** (1:1 with the source; **41 drops**, 0.00006 %) |
| Tokens | **71,383,345,402** (~71.4 B), mean **1,069**/doc |
| Raw pyconfind contacts | **31,884,900,670**, mean **478**/structure |
| Shards | 3,338, verified **3,338/3,338 coverage, 0 missing, 0 duplicates** |
| Mean protein length | 237.3 residues (source range 60–1000) |
| Compute | ~2,850 core-hours (Genoa 1-core: 3,074 s per 20 k-structure part) |

**Published** to [`open-athena/MarinFold`](https://huggingface.co/buckets/open-athena/MarinFold)
(both verified by **anonymous** read-back):

| View | Path | Size |
| --- | --- | --- |
| Documents corpus | `data/document_structures/contacts_v1_esm_atlas/train/` (+ `tokenizer/`, `README.md`) | 133.3 GB |
| Reusable contacts | `data/contacts/esm_atlas_esmfold2_distill/` (+ `README.md`) | 316.7 GB |

This is **~16× the AFDB `contacts_v1` corpus** (4.21 M docs / 4.67 B tokens) —
the "67 M proteins instead of 4 M" expansion from #91.

### Validation
- Reusable-contacts serializer (`analyzed_to_row` / `analyzed_from_row`) landed in
  `marinfold` with a round-trip **byte-identity** test: a document built from the
  saved contacts equals one built directly, so future formats skip pyconfind.
- Worker validated on real ESM-Atlas cif before launch (single-chain, 0 drops, raw
  contacts ⊃ document contacts). The published `shard-00000` row 0 matches that
  local validation exactly (`0000052aa00ab212061f7c6987fd87ae`, 60 residues,
  91 raw contacts) — deterministic from local laptop to published corpus.

### Where it ran (and why)
The GCP marin CPU pool is a small `n2-highmem-2` **fallback** scale group, not a
fan-out pool: a 256-worker request was granted **~6 workers** and never scaled,
projecting **~37 days**. CoreWeave's `cpu-genoa` pool is *reserved and pinned
always-warm*, so 512 workers materialized in ~10 minutes. Same job: 37 days → ~7 h.

### Operational notes (cost us time; worth knowing)
- **pyconfind `[fast]` (numba) auto-parallelizes to ~26 cores per worker.** Pin
  `NUMBA_NUM_THREADS=1` (+`OMP`/`OPENBLAS`) and scale by worker count, else workers
  oversubscribe the node.
- **CoreWeave pods default to 5 Gi ephemeral storage** — pass `--disk` for anything
  that stages files locally (the mirror was evicted twice before this was found).
- **HF rate-limits the bucket `xet-write-token` endpoint.** 16 upload workers →
  sustained 429s and failure; **4 workers → zero 429s and *faster* overall**.
- **marin API drift**: `Dataset`/`ZephyrContext`/`ResourceConfig` now live in
  `zephyr.execution`; pin `httpx<1` (the 1.0.dev prerelease breaks `iris` on import).
- A **cluster-wide control-plane restart** killed the main run at 88 % and the first
  resume batch. Recovery was by *verifying coverage against object storage* rather
  than trusting job status — which also caught duplicate parts flushed by zombie
  workers after the kill. Any rerun of this shape should verify coverage + dedupe
  before publishing.

## Conclusion

The corpus exists, is complete (3,338/3,338 parts, 0 gaps, 0 duplicates), and is
published with its tokenizer and dataset READMEs. Two artifacts, not one: the
**documents corpus** is ready for training (#124-style follow-ups), and the
**reusable pyconfind contacts** mean the next document format over this source
(e.g. `contacts-and-crops-v1`) costs a cheap serialization pass instead of
~2,850 core-hours of pyconfind.
