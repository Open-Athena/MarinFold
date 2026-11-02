# exp91 distillation run — status (2026-07-08)

Status of the ESM Atlas (ESMFold2 Atlas, 1.1B predicted monomers) distillation
run that produces a quality-filtered, novel, de-duplicated subset to augment the
afdb-24M training set. For the *why* (characterization, funnel design) see
[`README.md`](README.md); for the *how* (runbook) see [`PIPELINE.md`](PIPELINE.md).

## TL;DR

**COMPLETE.** The full funnel completed 2026-07-06 and the **materialize** step
(decode each rep's `structure_blob` → mmCIF parquet) completed **2026-07-08**. We
now have, for all **66,759,963 representative proteins**, their **3D structures +
sequences + selection metadata** as parquet in S3
(`s3://marinfold-exp91-usw2/exp91/structures/parts/`, 3,338 parts, 66,759,963 rows
exactly). Schema is aligned to `timodonnell/afdb-24M`. **Ready for
`contacts-v1 generate`.**

## The run

- Instance: `i-073c8a0b3fa1a592f` (r7i.24xlarge, 96 vCPU / 768 GB, us-west-2),
  launched 2026-07-05 19:50 UTC, pipeline `_DONE` 2026-07-06 10:57 UTC,
  **self-terminated ~12:00 UTC** after the finish upload (billing stopped, fleet empty).
- Funnel wall-time ~15.1h; run cost ~$95.
- Third attempt: the first two runs OOM-wedged (see "Failures & fixes"). This one
  used the readahead-capped scan + chunked novelty + per-chunk S3 checkpointing.

### Final funnel (1,095,530,880 Atlas rows in)

| Stage | Result | Time |
|---|---|---|
| scan | 383,988,806 survivors (pLDDT ≥ 0.70, pTM ≥ 0.50, len 60-1000) | 21 min |
| novelty | −220,802,158 (57.5% redundant vs afdb-24M @ 40% id, `-s 4.0`) | ~13.0h (77 chunks) |
| leakage | −41,517 (vs eval set; negligible, as predicted) → kept **163,144,153** | 42 min |
| cluster | 163M → **66,759,963 clusters** (mmseqs linclust @ 40% id) | 34 min |
| select | **66,759,963 reps** (1/cluster: longest, then lowest pLDDT std) | 23 min |

## What we have now (`s3://marinfold-exp91-usw2/exp91/out/`)

Durable — no bucket lifecycle/expiry policy, versioning off; persists until
explicitly deleted. Final total **185 objects / 312 GB** (~$7/month in S3 Standard;
most of it the intermediate `survivors_0_*.fasta` shards — the deliverables, manifest
+ `clu_rep_seq.fasta`, are ~27 GB). The materialized **structures live in a separate
prefix** (`structures/parts/`, see "Materialize" below), not here.

| Object | Size | Contents |
|---|---|---|
| `selected_manifest.csv` | 8.64 GB | **The deliverable.** 66,759,963 rows; columns `cluster_id, protein_hash, seq_len, mean_plddt, ptm, plddt_std, cluster_size`. Authoritative list of selected reps + quality scores. Joins to the FASTA on `protein_hash`. |
| `clu_rep_seq.fasta` | 18.2 GB | Amino-acid **sequence** of every rep, keyed by `protein_hash`. This is the training sequence set. |
| `clu_cluster.tsv` | 10.8 GB | Full cluster membership (rep → all members), if the non-rep neighbors are ever wanted. |
| `cluster_map.csv` | 10.8 GB | Same membership, CSV form. |
| `survivors_0_*.fasta` | ~3.8 GB × 32 | Intermediate: the 384M pre-clustering survivor sequences (per scan shard). Not needed for training; kept for debugging. |
| `pipeline.log` | 4.9 MB | Full run log (per-stage/per-chunk timings, RAM/disk). |
| `_progress_novelty/`, `_progress_leakage/` | — | Per-chunk drop-ID checkpoints (resume state). |
| `_DONE`, `_pipeline_ok` | — | Success markers. |

## Materialize (structures) — DONE 2026-07-08 (`s3://marinfold-exp91-usw2/exp91/structures/parts/`)

Decoded each selected rep's `structure_blob` (Atlas Lance) → mmCIF → parquet.
Ran as **2 shards** (`chunk_id % 2`) across 2× c7i.24xlarge in us-west-2, each
self-terminating with a `_DONE_shard_<id>` marker; both finished ~01:5x UTC.

- **3,338 parts** (`part_00000.parquet` … `part_03337.parquet`), chunk_ids
  0-3337 contiguous, 0 missing / 0 dupes.
- **66,759,963 rows total** (Parquet-footer tally) — matches
  `selected_manifest.csv` exactly. Last part is the partial final chunk (19,963 rows).
- `seq_ok` (blob-decoded sequence == Atlas `sequence`) essentially all True
  (~1 mismatch per 20k spot-checked — trace rate, non-aborting integrity check).
- Cost: probe + diagnostic + materialize ~$40 (total exp91 ~$135).

### Schema (per part) — aligned to `timodonnell/afdb-24M`

`entry_id, cif_content, sequence, seq_ok, split, source, seq_len, global_plddt,
ptm, plddt_std, seq_cluster_id, cluster_size`

- Renamed to match afdb-24M: `mean_plddt` → `global_plddt`, `cluster_id` →
  `seq_cluster_id`. `entry_id` / `cif_content` / `seq_len` already matched.
- `split = "train"` (constant); real train/val/test holdout **deferred** to a later
  decision coordinated with the (unbuilt) eval set + its leakage dedup.
- `source = "esm-atlas-v1"`.
- Atlas-specific extras kept: `sequence`, `seq_ok`, `ptm`, `plddt_std`,
  `cluster_size`. afdb-only cols (uniprot_accession/tax_id/organism_name/
  struct_cluster_id/gcs_uri) are **not** fabricated.
- **CAVEAT:** afdb `global_plddt` may be 0-100 while Atlas values are 0-1 (same
  column name) — rescale before training on the union.

### How the OOM was beaten (the hard part of this step)

Repeated "wedges" (2- and 4-box, and a 1-box diagnostic) all froze ~8-10 min in
with CPU → 1%, NetworkIn → 0. `--log-io` proved the cause was **per-worker RAM
creep → OOM cascade**, not S3: Arrow's memory pool retained freed `take()` buffers
and glibc retained decoded-CIF string arenas; × many synchronized workers → the
192 GB box OOM-killed mid-first-chunk. Fixes that held: `_release_memory()`
(`pa.default_memory_pool().release_unused()` + glibc `malloc_trim(0)`) after each
sub-batch (intra-chunk creep) + `maxtasksperchild=1` on the decode Pool (fresh
process per chunk → resets cross-chunk creep) + `DECODE_WORKERS=64`. See the
memory note `exp91-full-run-in-flight` for the full diagnostic trail.

## Validated facts (useful for the next run)

- The scan readahead cap held: RAM avail bottomed at **360 GB** (the wedged 64-worker
  run hit ~0). See `pipeline.py` `_scan_range` (`batch_readahead=4, fragment_readahead=2`).
- Clustering 163M sequences peaked at only ~45 GB used (~723 GB avail), so **768 GB
  was overkill**. A future rerun can drop to **m7i.24xlarge (96 vCPU / 384 GB)** —
  same speed (novelty is core-bound, not RAM-bound), ~20% cheaper.
- The `_progress_*` → S3 checkpoint round-trip works: a failed run resumes via
  `./create_dataset.sh run` (re-scans ~21 min, rebuilds the target DB, skips done chunks).
- Leakage vs the eval set is ~nil (41.5K of 384M), consistent with the
  metagenomic-vs-recent-PDB reasoning.

## What remains

1. **`contacts-v1 generate`** on the `structures/parts/` prefix (all in-region
   us-west-2). This is the immediate next step — the dataset is ready.

2. **Publish design (gated).** Any HF publish is cross-cloud, multi-TB (needs
   sign-off), and the Atlas is **CC BY-SA 4.0** (ShareAlike) — verify exact license
   terms before republishing a derived dataset.

3. **Land the branch.** `exp/91-evals-esmfold2-atlas` is still local: push +
   open a PR (`/ultrareview` against `origin/main`), merge, delete.

4. **Housekeeping (optional).** Prune the run logs left in the `structures/` prefix
   (`pipeline_shard_*.log(.live)`, `probe.log.live`) before any HF publish. The
   funnel `out/` prefix still carries ~122 GB of intermediate `survivors_0_*.fasta`
   (kept for debugging; deletable).

## Failures & fixes (for reproducibility)

- **Run 1** (`i-060bb48dd327f9dce`): novelty stage `mmseqs easy-search` over ~384M
  queries fell into split-prefilter mode, generated 2.3 TB+ scratch, thrashed. Fixed
  by chunking the query set (`--query-chunk-seqs`) with per-chunk scratch deletion.
- **Run 2** (`i-0394e34f2b6368f68`): scan OOM-wedged — 64 workers × uncapped Lance S3
  readahead × the heavy `per_residue_plddt` column marched RAM 772 → 0 GB. Fixed by
  capping scanner readahead + dropping `SCAN_WORKERS` 64 → 32 (commit `b44dc0d`).
- **Run 3** (`i-073c8a0b3fa1a592f`): succeeded.
