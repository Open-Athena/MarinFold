# Exp199 rollout-v2 checkpoint evaluation

This directory evaluates five selected exp199 exports with the exp82 rollout+resample recipe. It also reproduces the historical exp75 E8 checkpoint as an acceptance test for the evaluation path. The driver and every GPU worker run in CoreWeave. Four exp199 snapshots were downloaded from pinned Hugging Face revisions inside CoreWeave and verified before upload to CoreWeave S3. The cooldown export already existed in CoreWeave and was verified and evaluated in place without a checkpoint copy. The E8 validation reuses an existing CoreWeave S3 copy only after hashing all seven files in CoreWeave against its pinned Hugging Face manifest. Eval inputs stream from Hugging Face into the same bucket. This CoreWeave evaluation path contains no GCS source or destination. The one-time continuation publisher is separate and copies only its final Levanter and HF checkpoints from TRC GCS into durable Hugging Face storage.

## Fixed recipe

- 577 `(dataset, stem)` units / 575 unique stems, with the unchanged legacy 554 reported separately
- 100 fresh contacts-v1 realizations per unit
- temperature 1.0, top-p 0.95, top-k disabled (`-1`)
- token budget `min(8192 - prompt_tokens, 6L + 128)`
- occurrence-frequency voting, no pairwise tie-break
- 12 independent single-H100 shards per checkpoint, batch priority
- one real-protein, 100-rollout smoke gate per checkpoint before full fanout

The model source identities, individual file sizes/digests, source losses, eval input digests, MarinFold revision, and CoreWeave S3 root are fixed in `checkpoint_specs.py`. Exported tensors are float32 and vLLM evaluates them as bfloat16, matching the established CoreWeave path.

## Run

From this directory, with the normal Marin Iris credentials loaded:

```bash
uv run python submit_coreweave.py --run-id v2-YYYYMMDD-NN --seed 1
```

Run the historical E8 acceptance suite separately:

```bash
uv run python submit_coreweave.py \
  --run-id e8ref-v2-YYYYMMDD-NN \
  --suite e8-reference \
  --seed 1
```

Run the continuation checkpoint without rerunning the original three models:

```bash
uv run python submit_coreweave.py \
  --run-id contbase-v2-YYYYMMDD-NN \
  --suite continuation \
  --seed 1
```

Run the CoreWeave cooldown directly from its existing final HF-format export:

```bash
uv run python submit_coreweave.py \
  --run-id cooldown-v2-YYYYMMDD-NN \
  --suite cooldown \
  --seed 1
```

To resume the same S3 run from a distinct Iris root job, add a unique
`--job-suffix`. Workers use atomic completion markers and skip finished units:

```bash
uv run python submit_coreweave.py \
  --run-id v2-YYYYMMDD-NN \
  --seed 1 \
  --job-suffix resume1
```

The root CPU driver is federated to `cw-us-east-02a`. It verifies or stages the
pinned HF artifacts, submits and waits for smoke jobs, submits and waits for 12
full H100 jobs per selected checkpoint, validates completeness, runs the exact
checked-in exp89 metric source, and publishes results below:

```text
s3://marin-us-east-02a/marin/protein-structure/MarinFold/
  exp199_optimize_contacts_v1_afdb_esm/evals/rollout_v2/<run-id>/
```

`results/run_manifest.json` is the completion authority. New runs score 577 units and write it only after each checkpoint has 577 matrices, 57,700 finished rollouts, complete timing rows, and 11,540 metric rows. The same run reports the legacy 554, eval2 pooled, eval2-natural, and <30% cuts. Partial sparse parts and completion markers remain resumable after preemption or failure.

## Results

Run `v2-20260812-06` evaluated the original three checkpoints and run
`contbase-v2-20260812-01` evaluated the continuation checkpoint on 2026-08-12.
Both used sampling seed 1 and have zero unfinished rollouts. The earlier seed-0
attempt was rejected rather than scored because one p06 rollout hit the fixed
token cap.

| Model | All R | Long R | All AUC | Long AUC |
|---|---:|---:|---:|---:|
| TRC p03-aug, step 72,599 | 0.5736 | 0.5264 | 0.9407 | 0.9249 |
| TRC p03-base, step 72,599 | 0.5792 | 0.5327 | 0.9427 | 0.9272 |
| CoreWeave p06-aug, step 145,199 | **0.6088** | **0.5633** | **0.9480** | **0.9334** |
| TRC continuation srcbase-aug100, step 145,199 | 0.6033 | 0.5551 | 0.9472 | **0.9335** |
| CoreWeave p06 cooldown, step 290,400 | **0.6307** | **0.5837** | **0.9511** | **0.9383** |

The cooldown improves over its earlier p06-aug checkpoint by 0.0218 all-range and 0.0204 long-range R-precision on the exact legacy 554. Its validation loss falls from 2.9712 to 2.9397. At p03, base is ahead of aug by 0.0056 all-range and 0.0063 long-range R-precision. The continuation remains 0.0056 and 0.0082 behind p06-aug on those cuts.

### CoreWeave p06 cooldown

Run `cooldown-v2-20260815-01` evaluated only W&B run `prot-exp199-cw-cv1-p06-cool-s01` at final step 290,400. It used the existing export at `s3://marin-us-east-02a/marin/protein-structure/MarinFold/exp199_continue_contacts_v1_cw/checkpoints/protein/prot-exp199-cw-cv1-p06-cool-s01/2026.08.14.1/hf/step-290400`; no checkpoint was copied and no HF publication was created.

| Reporting cut | n | All R | Long R | All AUC | Long AUC |
|---|---:|---:|---:|---:|---:|
| **eval2 natural (<40% identity)** | 78 | **0.3579** | **0.2998** | **0.8702** | **0.8426** |
| eval2 pooled (<40% identity) | 307 | 0.5539 | 0.4955 | 0.9328 | 0.9152 |
| eval2 natural (<30% identity) | 61 | 0.3202 | 0.2538 | 0.8568 | 0.8252 |
| eval2 pooled (<30% identity) | 275 | 0.5503 | 0.4896 | 0.9322 | 0.9140 |
| Legacy exp89 | 554 | 0.6307 | 0.5837 | 0.9511 | 0.9383 |

Completeness passed with 577 `(dataset, stem)` units, 575 unique stems, 57,700/57,700 finished rollouts, zero unfinished rollouts, 577 dense matrices, and 11,540 metric rows. The worker SHA-256 is `f28829f9826a7089f082cb55de45582c7cbc389ea3d700e5c5869ff29bb6bc82`, exactly the worker accepted by the E8 reproduction below. `data/cooldown_run_manifest.json`, `data/cooldown_checkpoint_verification.json`, and `data/cooldown_subset_aggregate_metrics.csv` are the checked-in evidence. Full outputs remain under `s3://marin-us-east-02a/marin/protein-structure/MarinFold/exp199_optimize_contacts_v1_afdb_esm/evals/rollout_v2/cooldown-v2-20260815-01/`.

The full aggregate table is checked in at `data/aggregate_metrics.csv`. Complete
per-unit metrics, matrices, and provenance manifests are under:

```text
s3://marin-us-east-02a/marin/protein-structure/MarinFold/
  exp199_optimize_contacts_v1_afdb_esm/evals/rollout_v2/v2-20260812-06/
s3://marin-us-east-02a/marin/protein-structure/MarinFold/
  exp199_optimize_contacts_v1_afdb_esm/evals/rollout_v2/contbase-v2-20260812-01/
```

Each model passed validation with 554 `(dataset, stem)` units, 552 unique stems,
55,400/55,400 finished rollouts, zero unfinished rollouts, 554 dense matrices,
and 11,080 metric rows. Long-range R and AUC have 553 valid values out of 554;
all-range R and AUC have 554/554.

The continuation export is pinned at Hugging Face revision
`00eddb761fd028f07ce7bc088930271516da9866`. CoreWeave downloaded and verified
only its six-file, 5,885,614,712-byte HF export, then stored it at the eval-local
`models/` path. `data/continuation_checkpoint_verification.json` records the
source, destination, sizes, and digests.

`data/timings.csv` contains the required per-input timing records for all five exp199 checkpoints and the E8 validation checkpoint (3,347 rows total).

## Historical E8 validation

Run `e8ref-v2-20260812-01` completed entirely in CoreWeave on 2026-08-12 with
sampling seed 1. It evaluated only
`prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084/hf/step-35679` from pinned HF
revision `4c9e7779635b585730180823e0ab4b3319b82f67`.

The model already existed at
`s3://marin-us-east-02a/MarinFold/exp163/model/step-35679`. The path name was not
treated as proof of identity: the CoreWeave driver streamed and hashed every
object before evaluation. Both weight-shard SHA-256 values and all five small-file
Git blob hashes matched the pinned HF manifest, so no checkpoint copy was made.
The full evidence is in `data/e8_checkpoint_verification.json`.

| Metric | Historical exp82 | New | Absolute difference | Pass (≤0.005) |
|---|---:|---:|---:|:---:|
| All R | 0.424529 | 0.424697 | 0.000167 | yes |
| Long R | 0.365615 | 0.366053 | 0.000438 | yes |
| All AUC | 0.900963 | 0.900503 | 0.000460 | yes |
| Long AUC | 0.873780 | 0.873036 | 0.000745 | yes |

All four historical values reproduce within tolerance. Completeness also passed:
554 units, 552 unique stems, 55,400/55,400 finished rollouts, zero unfinished
rollouts, 554 dense matrices, and 11,080 metric rows. The checked-in evidence is
`data/e8_reference_validation.json` and
`data/e8_reference_aggregate_metrics.csv`; complete results and the authoritative
run manifest are under:

```text
s3://marin-us-east-02a/marin/protein-structure/MarinFold/
  exp199_optimize_contacts_v1_afdb_esm/evals/rollout_v2/e8ref-v2-20260812-01/
```

For clarity, the exp199 result run `v2-20260812-06` evaluated exactly the three
requested checkpoints: TRC p03-aug step 72,599, TRC p03-base step 72,599, and
CoreWeave p06-aug step 145,199. The E8 checkpoint was evaluated only in the
separate reference run above. The continuation checkpoint was evaluated only in
`contbase-v2-20260812-01`.

## PR comparison

`data/pr_comparison.csv` records the corrected exp199 results, both exp75 E8
measurements, and the historical exp146 and exp166 context with explicit
evaluation provenance, W&B run IDs, steps, HF paths, and verified CoreWeave paths
where available. `plot_pr_comparison.py` renders
`plots/rprecision_ranges_vs_loss.png` from that table and the checked-in exp89
Protenix-v2 rows. `build_pr_all_r_rows.py` derives the compact, checked-in
per-protein all-range table from the exact historical and CoreWeave result files;
`plot_primary_comparison.py` uses it to render the primary boxplot-and-scatterplot
figure at `plots/final_checkpoint_rprecision.png`.

The plot shows the two exp75 evaluations separately but averages them before fitting, so one checkpoint contributes one fit observation. Only unique 1.5B checkpoints enter the descriptive curve; the historical exp146 3B checkpoint is shown but excluded. Circles identify the six results computed through the `/eval-checkpoint` workflow from PR #214, while squares identify values taken from previous evaluations. The cooldown box and point use its legacy-554 slice so they remain directly comparable to PR #219.
