# Proteina monomers for contacts-v1: running experiment and initial results

**Results through September 10, 2026, 13:20 UTC / 9:20 a.m. EDT.**
[Experiment #278](https://github.com/Open-Athena/MarinFold/issues/278) ·
[Implementation PR #282](https://github.com/Open-Athena/MarinFold/pull/282) ·
[Scale W&B run](https://wandb.ai/open-athena/MarinFold/runs/exp278-proteina-scale-20260909)

We are generating synthetic, sequence-paired protein monomers to test whether they
can add structural diversity to MarinFold's contacts-v1 training data. About
15 hours 49 minutes after launch, we have saved **476,976 generated backbones**
and produced **263,735 quality-passing provisional documents**. The run is active.
These documents still require evaluation-reference decontamination and global
structural diversity selection before training use. The experiment has not yet
established a final retained yield or a training benefit.

## What is running

The pipeline is:

**Proteina Cα trace → one ProteinMPNN sequence → ESMFold structure → quality
checks → contacts-v1 document using the refolded geometry.**

Proteina supplies Cα coordinates. ProteinMPNN assigns a sequence to that trace;
ESMFold then predicts a full structure from the sequence. We compare its Cα trace
against the generated trace for self-consistency. The contact labels come from
the **ESMFold-refolded backbone**, using the existing native-amino-acid rotamer
contact calculation. Both geometries remain available for later analysis.

### Generation recipe

| Component | Actual scale-run setting |
|---|---|
| Protein lengths | Every integer length from 60 through 500 residues; one chain |
| Proteina source | [`a44b407daf6a5358e43cd68907f3e3f1cbc65fdc`](https://github.com/NVIDIA-BioNeMo/proteina/tree/a44b407daf6a5358e43cd68907f3e3f1cbc65fdc) |
| Short checkpoint, 60–250 aa | `proteina_v1.2_DFS_200M_notri.ckpt` |
| Long checkpoint, 251–500 aa | `proteina_v1.6_DFS_200M_notri_long_chain_generation.ckpt` |
| Sampling | 400 steps (`dt=0.0025`), self-conditioning, `sampling_mode=sc`, log schedule with parameter 2; noise scale 0.45 short / 0.35 long; score scale and guidance weight 1; autoguidance ratio 0 |
| Sampling numerics | Compiled network with static shapes; FP32 Proteina sampling with reference float32 matmuls, without the pilot's optional TF32 sampling speedup |
| Backbone batch size | 32 at 60–200 aa; 16 at 201–500 aa |
| Conditioning | Unconditional plus CATH class requests `1.x.x.x` (mainly alpha), `2.x.x.x` (mainly beta), `3.x.x.x` (alpha/beta) |
| Sequence design | Cα-only ProteinMPNN `v_48_020`; revision `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`; temperature 0.1; X excluded; **one sequence attempt per backbone** |
| Refolding | `facebook/esmfold_v1` revision `75a3841ee059df2bf4d56688166c8fb459ddd97a`; `num_recycles=4`; trunk chunk size 128; ESM submodel in half precision; the separate design/refolding process permits TF32 matmuls |
| Runtime | H100 80GB; PyTorch 2.4.1 / CUDA 12.4 image pinned by digest; Transformers 4.48.3; pyconfind 0.5.0 |

The full sampling arguments are in [sample_worker.py](sample_worker.py), model
setup and refolding in [fold_worker.py](fold_worker.py), dependency versions in
[bootstrap.sh](bootstrap.sh), and image digest in [launch.py](launch.py).
[prepare_assets.py](prepare_assets.py) identifies the checkpoint downloads;
workers record the loaded asset checksums. Each submission records its exact
content-addressed code bundle.

The current conditioning is at broad **class** level. Requested labels are not
independent measurements of realized architecture/topology or proof of new folds.
The scale run uses the two 200M no-triangle checkpoints and one design attempt;
the optional 400M, architecture/topology conditioning and additional sequence
attempts from the original proposal are not part of this run.

### Quality gate and document construction

Each candidate must have a complete, sequence-matched single-chain backbone and
finite Cα coordinates, then pass all of the following:

- Full-chain, aligned Cα self-consistency RMSD **≤2 Å**.
- Mean Cα ESMFold pLDDT **≥70**. The pinned API returns normalized confidence;
  the wrapper converts it to the 0–100 scale.
- No nonlocal Cα clash below 2.5 Å for pairs separated by at least three positions.
- Adjacent Cα distances between 3.4 and 4.2 Å, with an explicit cis-proline
  exception: Cα distance 2.7–3.4 Å, cis omega within 20 degrees, and C–N distance
  1.15–1.60 Å.

These are the implemented computational checks in [quality.py](quality.py), not
experimental evidence of folding or monomeric state in solution.

Passing structures use `GenerationConfig()` from the existing contacts-v1
generator: native-only rotamers, `contact_distance=3.0`, `dcut=25.0`,
`clash_distance=2.0`, minimum sequence separation 6 and minimum contact degree
0.001. The worker verifies that each serialized document reconstructs the
designed sequence exactly. Candidate scores and geometry failures are preserved;
serializer inconsistencies raise errors rather than silently dropping examples.

## Workload, quotas and compute

The [frozen manifest](data/scale-20260909/manifest.json) contains **4,820,608 raw
candidates**, divided into **13,923 cases** and **768 independent worker queues**.
Its SHA256 is
`a7812d1e4173454eba2185f688cde8b60d4d8b55c97481b32acce56bdb364f80`.

The target is approximately **one million retained documents**, with uniform
accepted counts across integer lengths and equal accepted contributions from the
four requested arms. Raw quotas interpolate inverse pilot retention separately
by length and arm, then round to whole batches. Thus raw generation is deliberately
unequal: harder length/arm combinations receive more attempts. The manifest is
fixed during this run; there is no online adaptation of these quotas.

| Requested arm | Planned raw backbones |
|---|---:|
| Unconditional | 1,288,160 |
| Mainly alpha (`1.x.x.x`) | 1,466,304 |
| Mainly beta (`2.x.x.x`) | 725,744 |
| Alpha/beta (`3.x.x.x`) | 1,340,400 |
| **Total** | **4,820,608** |

At interpolated pilot yields this manifest projects approximately **1,005,827
retained candidates** and **12,728 pure H100-hours**, or **15,273 H100-hours with
20% overhead**. These are planning estimates, not measured scale-run expenditure
or guaranteed yield. Global duplicate growth, changed retention, CPU selection,
storage and transfers can change the final cost. The finite manifest is bounded
below six million raw candidates; the user may choose a smaller final corpus
after reviewing the evidence.

All 768 jobs are Iris **root jobs on `cw-us-east-02a`**, each requesting one H100,
one replica/process, eight CPUs and 64 GiB each of RAM and scratch. They use batch
priority (`Fray priority=3`, Kubernetes `iris-batch`). There is no multi-GPU gang
or parent driver whose exit kills the fleet. Each job can be admitted, evicted
and retried independently. **768 queued jobs does not mean 768 allocated GPUs.**
East02 has 256 schedulable H100s shared with other work. Our observed running
allocation reached 112 during startup and was **88 at the latest check**, with
680 pods waiting. No production job was terminal-failed at that check.
See the [fleet snapshot](data/scale-20260909/fleet-20260910T1319Z.json).

Each worker owns a deterministic list of cases, balanced by estimated compute
and shuffled to mix lengths/classes. For each case it runs backbone generation,
then design/refolding and document construction, in separate subprocesses.
Models remain loaded within a stage/case; downloaded assets stay cached on the
worker between cases. It then moves to the next case. This is a single-GPU
pipeline per worker, rather than separate global generation and folding fleets.

Native Iris retries allow 100 worker/preemption failures and two application
failures. Startup recorded 32 retries, including deleted pods; an inspected job
returned independently to the admission queue. Saved backbone batches and
sequences are reused on restart. Incomplete work can be repeated after eviction.

Compute and durable storage are co-located in East02. RNO2A remains unused while
explicit approval for its large cross-region model/output transfers is pending.

## Initial results

Production began **September 9 at 21:32:01 UTC / 5:32 p.m. EDT**. Counts below come
from committed artifacts and batch markers. These are live, nontransactional
snapshots; in-flight work can lag the displayed totals.

| Stage | Sept 9, 23:30 UTC (~2 h) | Sept 10, 13:20 UTC (~15 h 49 m) |
|---|---:|---:|
| Backbones saved | 68,624 | **476,976** |
| Designed sequences saved | 53,904 | **428,992** |
| Refolds committed | 52,960 | **427,152** |
| Quality-pass provisional documents | 34,355 | **263,735** |
| Quality-pass / completed refolds | 64.9% | **61.7%** |

At the latest snapshot, 1,155 cases had completed the full GPU pipeline. The
49,824-backbone gap between generation and committed refolding is unfinished
downstream work, not a count of quality rejections.

Quality retention by length at the latest snapshot:

| Length | Refolds committed | Quality-pass documents | Quality retention |
|---|---:|---:|---:|
| 60–100 | 20,832 | 17,219 | 82.7% |
| 101–200 | 51,328 | 37,043 | 72.2% |
| 201–300 | 84,464 | 55,928 | 66.2% |
| 301–400 | 102,608 | 64,381 | 62.7% |
| 401–500 | 167,920 | 89,164 | 53.1% |
| **All** | **427,152** | **263,735** | **61.7%** |

Longer proteins show lower quality retention. The pooled percentage also depends
on which lengths and arms have finished; its change from 64.9% to 61.7% alone is
not evidence that the generator's behavior deteriorated over time.

Sources: [two-hour snapshot](data/scale-20260909/reports/snapshot-20260909T233035Z.json),
[latest reported snapshot](data/scale-20260909/reports/snapshot-20260910T132043Z.json),
and [per-case counts](data/scale-20260909/reports/cases-20260910T132043Z.csv).

### What these counts establish—and what remains to be measured

The pipeline is producing sequence-paired, serializable monomer documents at
scale, and retaining raw artifacts through worker turnover. **263,735 is a
quality-pass count, not a final training-corpus count.** Full-corpus sequence and
evaluation-structure exclusion, structural clustering and diversity capping have
not yet been applied. Final retained counts can be substantially smaller.

The earlier 1,536-backbone screen produced 1,044 quality-pass candidates, 523 after
decontamination and 494 after the fine-cluster cap, in 453 clusters: **32.2% final
screen retention**. Its six-length/equal-raw-count mixture differs from this scale
manifest, so that pooled percentage is not a direct scale forecast. Class
conditioning changed secondary-structure composition, but the small matched
comparison did not establish the proposed diversity gain. See the
[pilot results and limitations](README.md#results-initial-screen-completed-on-2026-09-09).

Before scale launch, two crossover canaries at 250/251 aa saved all 64 backbones
and sequences; 47 passed quality. Twenty-one CPU tests passed, and a separate
16-candidate audit exercised sequence/structure screening, clustering and timing
export end to end. These checks validate integration, not production-scale
diversity or the statistical reliability of the yield projection.

## What is saved and where

The durable root is:

```text
s3://marin-us-east-02a/MarinFold/exp278-proteina/scale-20260909/
```

| Under `cases/<case-id>/` | Contents |
|---|---|
| `generated/batch-*.npz` | Original Cα coordinates and original per-input generation timings in the same atomic object |
| `folded/sequences/*.parquet` | Every designed sequence, saved before refolding, with source identity, seed, model revision and timing/worker metadata |
| `folded/candidates/*.parquet` | All committed refolds, including rejects: original Cα coordinates, sequence, full refolded PDB, quality scores and geometry checks |
| `folded/documents-provisional/*.parquet` | Quality-passing contacts-v1 documents |
| `generated/timings.csv`, `folded/timings/*.csv` | Predictor timings and execution metadata |
| `complete.json`, `completed-parts/`, `progress.json` | Stage/batch completion and recovery state |

Manifests, submission records, worker status and reports live under the same root.
Stable source-batch and sample identities connect sequences, both geometries,
documents and timing records. **Rejects are retained**, so later analyses can
change the quality/selection criteria without losing those examples. Small
summaries and manifests are also committed here; the bulk artifacts require
access to the CoreWeave bucket. [SCALE_OPS.md](SCALE_OPS.md) contains operational
commands and recovery details.

## Next review

The first detailed review is scheduled to start **September 10 at 15:32 UTC /
11:32 a.m. EDT**, 18 hours after production submission. **Generation continues
while we review.** The workstation timer remains enabled; it requires the
workstation and user session to be available, and catches a missed run at the
next login.

[scale_review.py](scale_review.py) will capture progress and resource time and
run [scale_audit.py](scale_audit.py): up to 64 candidates per 40-aa length bin and
requested arm, sampled from committed refolds. The audit applies the frozen
sequence/evaluation-structure screens and measures pre/post-refolding structural
redundancy, diversity accumulation and nearest detected training-reference
matches. The sequence exclusion rule is `(identity ≥30% and shorter-sequence
coverage ≥50%) OR E≤0.001`; the fine structural-neighbor rule requires TM-score
≥0.8 in both normalizations and ≥80% coverage of both chains, with a cap of five
per connected fine cluster. Reference scope and search limitations are documented
in [README.md](README.md#filtering-and-corrections).

The audit weights quality/reference-pass estimates to the sampled committed-refold
population. Its cluster cap applies only within the sample; it cannot establish
the final global diversity retention. Reports and metrics will be saved, logged
to W&B and linked from a new comment on the experiment issue. The subsequent
decision is whether to keep the recipe and target, adjust the experiment, or
finish with fewer than one million documents.
