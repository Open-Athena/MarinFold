# Exp199 rollout-v2 checkpoint evaluation

This directory evaluates the three selected exp199 Hugging Face exports with the
fixed exp89 universe and exp82 rollout+resample recipe. It also reproduces the
historical exp75 E8 checkpoint as an acceptance test for the evaluation path. The
driver and every GPU worker run in CoreWeave. Exp199 snapshots are downloaded
from a pinned Hugging Face revision inside CoreWeave and verified before upload
to CoreWeave S3. The E8 validation reuses an existing CoreWeave S3 copy only
after hashing all seven files in CoreWeave against its pinned Hugging Face
manifest. Eval inputs stream from Hugging Face into the same bucket. This
workflow contains no GCS source or destination.

## Fixed recipe

- 554 `(dataset, stem)` units / 552 unique stems
- 100 fresh contacts-v1 realizations per unit
- temperature 1.0, top-p 0.95, top-k disabled (`-1`)
- token budget `min(8192 - prompt_tokens, 6L + 128)`
- occurrence-frequency voting, no pairwise tie-break
- 12 independent single-H100 shards per checkpoint, batch priority
- one real-protein, 100-rollout smoke gate per checkpoint before full fanout

The public HF model revision, individual file sizes/digests, source losses, eval
input digests, MarinFold revision, and CoreWeave S3 root are fixed in
`checkpoint_specs.py`. Exported tensors are float32 and vLLM evaluates them as
bfloat16, matching the established CoreWeave path.

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

`results/run_manifest.json` is the completion authority. It is written only after
all checkpoints have 554 matrices, 55,400 finished rollouts, complete timing rows,
and 11,080 metric rows each. Partial sparse parts and completion markers remain
resumable after preemption or failure.

## Results

Run `v2-20260812-06` completed on 2026-08-12 with sampling seed 1. The seed-0
attempt was rejected rather than scored because one p06 rollout hit the fixed
token cap. The completed run has zero unfinished rollouts for every model.

| Model | All R | Long R | All AUC | Long AUC |
|---|---:|---:|---:|---:|
| TRC p03-aug, step 72,599 | 0.5736 | 0.5264 | 0.9407 | 0.9249 |
| TRC p03-base, step 72,599 | 0.5792 | 0.5327 | 0.9427 | 0.9272 |
| CoreWeave p06-aug, step 145,199 | **0.6088** | **0.5633** | **0.9480** | **0.9334** |

The p06 checkpoint improves over p03-base by 0.0296 all-range R-precision and
0.0306 long-range R-precision. At p03, base is ahead of aug by 0.0056 and
0.0063 respectively.

The full aggregate table is checked in at `data/aggregate_metrics.csv`. Complete
per-unit metrics, matrices, and the provenance manifest are under:

```text
s3://marin-us-east-02a/marin/protein-structure/MarinFold/
  exp199_optimize_contacts_v1_afdb_esm/evals/rollout_v2/v2-20260812-06/
```

Validation: 554 `(dataset, stem)` units, 552 unique stems, 55,400/55,400
finished rollouts, zero unfinished rollouts, 554 dense matrices, and 11,080
metric rows per model. Long-range R and AUC have 553 valid values out of 554;
all-range R and AUC have 554/554.

`data/timings.csv` contains the required per-input timing records for all three
exp199 checkpoints and the E8 validation checkpoint (2,216 rows total).

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
separate reference run above.
