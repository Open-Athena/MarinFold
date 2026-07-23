---
marinfold_experiment:
  issue: 147
  title: "exp: pilot on-the-fly contacts-v1 training from ESM Atlas contacts"
  kind: models
  branch: codex/on-the-fly-contacts
---

# exp: pilot on-the-fly contacts-v1 training from ESM Atlas contacts

**Issue:** [#147](https://github.com/Open-Athena/MarinFold/issues/147) · **Kind:** `models` · **Branch:** `codex/on-the-fly-contacts`

## Question

Can we train the exp117 Qwen3 1.47B recipe directly from the reusable ESMFold2 Atlas pyconfind-contact rows, constructing the canonical `contacts-v1` documents and token sequences at read time rather than materializing a text corpus?

## Hypothesis

A shard-local direct Levanter dataset can reconstruct `AnalyzedStructure`, call the existing deterministic `contacts-v1` builder, encode with the pinned contacts tokenizer vocabulary, and greedily pack the same document stream quickly enough to feed a TPU training pilot. The ordinary full `contacts-v1` validation loss should remain directly comparable to exp117 (interesting below 2.8).

## Approach

- Read premade raw contacts from `hf://buckets/open-athena/MarinFold/data/contacts/esm_atlas_esmfold2_distill/`.
- Stage a sub-10 GB pilot subset in the TPU region. No text, token, or
  per-document length index is materialized.
- Map global example indices to deterministically shuffled shards and fixed
  shard-local output slots. Shuffle all rows within each shard per epoch.
- Reconstruct canonical causal serialized documents on demand with
  `analyzed_from_row` + `build_document`, then encode with the immutable
  `timodonnell/contacts-v1-tokenizer@5d68a24a899f` vocabulary and EOS.
- Pack each complete shard with best-fit decreasing and emit exactly 2,650
  examples. Uniformly sample packed bins if the shard is overfull; insert
  zero-loss examples if it is underfull. Cross-document attention remains
  blocked through segment IDs.
- Train from scratch with the best finished exp117 configuration: Qwen3
  1.47B, sequence length 8192, global batch 256, LR 3.1623e-3, WD 0.2,
  cosine schedule, 10% warmup, AdamW betas 0.9/0.95, data seed 0. The
  default v6e-8 pilot uses per-device parallelism 16 and two gradient
  accumulation steps.
- Evaluate the regular full `contacts_v1` validation set normally under
  `eval/tokenized/contacts-v1-val/loss`.

### Prototype contract and consequences

`FixedQuotaPremadeContactsDataset` is a contacts-specific adapter over the
reusable
`marinfold_models.shard_documents.FixedQuotaShardDocumentDataset`. The generic
layer owns index mapping, shuffling, whole-shard packing, quota selection, and a
reconstructable LRU cache; the experiment supplies the premade-contact columns
and deterministic row-to-`Document` function.

For `S` shards and quota `N`, an example index maps arithmetically to
`(epoch, shuffled_shard, slot)` within an epoch of `S * N` examples. Repeated or
out-of-order reads are identical. Cache contents affect performance only, so
ordinary Levanter checkpoint resume needs no loader state or integration hook.
The training configuration keeps outer shuffling disabled and uses a
single-component mixture block size of one because shuffling already lives in
the indexed dataset.

The 2,650-example quota is provisional. From the sampled 20,000-document shard
statistics, it should retain roughly 98.25% raw-token utilization while making
overflow rare, before accounting for actual between-shard variation and
packing fragmentation. If a shard is overfull, packed bins are selected
uniformly without replacement. Every document in that shard consequently has
the same conditional inclusion probability because it belongs to exactly one
bin.

### Pilot staging

The staging helper is preview-only unless `--execute` is supplied. The preview
reads public Hugging Face and destination GCS metadata to report only the
missing shards; it does not transfer data:

```bash
uv run stage_pilot.py --num-shards 16
```

After reviewing the reported source size and destination, apply that exact
plan explicitly:

```bash
uv run stage_pilot.py --num-shards 16 --execute
```

An approved full-corpus mirror uses the same resume-safe destination. Transfers
above 10 GB require the explicit acknowledgement flag:

```bash
uv run stage_pilot.py \
  --num-shards 3338 --workers 32 \
  --execute --allow-large-transfer
```

Once staged, construct and consume one complete fixed-quota shard locally on
the GCS data path:

```bash
PYTHONPATH=.:../../marinfold:../../models \
  uv run smoke_dataset.py --num-shards 1
```

This reports wall time, examples per second, generator drops, truncations,
quota discards or padding packs, and real-token packing utilization. It reads
the staged shard but does not launch training.

### Launch

Inspect the fully lowered Marin graph without submitting anything:

```bash
PYTHONPATH=.:../../marinfold:../../models uv run train.py
```

The graph contains a CPU dependency that materializes the ordinary
`contacts-v1` validation cache and one v6e-8 training artifact that consumes
the direct on-the-fly dataset. After the staged-data smoke test passes, submit
that exact graph through a small CPU coordinator. The coordinator dispatches
the validation-tokenization and v6e-8 worker jobs:

```bash
uv run iris --cluster=marin job run --no-wait --enable-extra-resources \
  --cpu=1 --memory=2G --extra=cpu --zone=us-east5-b \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python train.py --run
```

The schedule-matched sanity run keeps exp117's 35,680-step optimizer horizon
and 1,115-step evaluation cadence. It first tries a same-region v6e-32; the
first two validation points are the decision points:

```bash
uv run iris --cluster=marin job run --no-wait --enable-extra-resources \
  --cpu=1 --memory=2G --extra=cpu --zone=us-east5-b \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e EXP147_STEPS 35680 \
  -e EXP147_STEPS_PER_EVAL 1115 \
  -e EXP147_NUM_SHARDS 3338 \
  -e EXP147_TPU v6e-32 \
  -e EXP147_ZONE us-east5-b \
  -e EXP147_PER_DEVICE_PARALLELISM 8 \
  -e EXP147_NAME exp147-otf-contacts-v1-1_5b-steps35680-bs256-v6e32 \
  -e EXP147_VERSION steps35680-v6e32-dev \
  -- python train.py --run
```

Before submission, the current branch must be pushed and this experiment's
lockfile refreshed to that pushed commit so Iris workers install the same
document and loader code. GCP application-default credentials are also needed
by the launcher. Running `train.py --run` directly outside Iris is rejected.

### Reference run

The parameter reference is Eric’s best finished 1.47B exp117 run:
[`prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs256-europe-west4`](https://wandb.ai/eric-czech/marin/runs/prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs256-europe-west4).
It reached `eval/tokenized/contacts-v1-val/loss = 2.703709`. The exp147
pilot defaults to 200 steps rather than copying its full 16-epoch token
budget; the architecture, batch, and optimizer settings match.

## Success criteria

1. A local one-shard data-loader smoke test proves on-the-fly documents are byte/token identical to the published companion documents.
2. Fixed-quota packing reaches at least 98% real-token utilization on a representative
   premade-contacts shard without random row reads across shards.
3. A short TPU pilot starts, sustains training without input starvation, and
   reports the standard `contacts-v1` validation metric.
4. The run is recorded in `history/runs/` and linked here.

## Results

The stateless prototype is implemented. Synthetic Parquet tests cover best-fit
packing, exact quota selection, zero-loss padding, overflow, deterministic
random and repeated access, cache reconstruction, remote URI preservation, and
the single-component Levanter mixture.

On 2026-07-23, 16 source shards (1.52 GB) were staged to
`gs://marin-us-east5/protein-structure/MarinFold/exp147_on_the_fly_contacts_v1_pilot/pilot_data/contacts/`.
A complete shard was then read from GCS and converted on the local CPU:

- 20,000 source documents produced 2,608 packed examples.
- No documents were dropped or truncated, and no packs were discarded by the
  fixed quota.
- The 2,650-example quota added 42 zero-loss padding examples.
- Real-token packing utilization was 98.2469%.
- Construction and consumption took 55.88 seconds: 47.42 packed examples per
  second.

The 200-step pilot then completed successfully on one preemptible v6e-8 in
`us-east5-b`, with zero failures or preemptions:

- [W&B run](https://wandb.ai/open-athena/MarinFold/runs/exp147-otf-contacts-v1-1_5b-pilot-200s-bs256-v6e8)
  and [Iris job](https://iris.oa.dev/#/job/%2Fjder%2Firis-run-train-20260723-195327).
- Validation `contacts-v1-val/loss` improved from `5.510117` at step 100 to
  `4.693140` at the final step 199. Final train loss was `4.541664`.
- Across steps 3–198, median step time was 27.792 seconds, median throughput was
  9.211 examples/second and 75,460 tokens/second, and median MFU was 14.057%.
  Median foreground loading time was 1.1 ms. The background 8,192-example
  producer emitted noisy slow-loading warnings, but it did not sustain a
  foreground training stall.
- The Levanter-native checkpoint is at
  `gs://marin-us-east5/protein-structure/MarinFold/exp147_on_the_fly_contacts_v1_pilot/users/jder/checkpoints/exp147-otf-contacts-v1-1_5b-pilot-200s-bs256-v6e8/exp147-dev/checkpoints/step-199`.
  The HF-compatible export, including the tokenizer, is next to it under
  `exp147-dev/hf/step-199`.

This run used git SHA `b317807`, before the no-rendering optimization in
`7260479`; its throughput is therefore the baseline for that change.

## Conclusion

The stateless fixed-quota loader successfully fed a v6e-8 throughout training,
and the ordinary contacts-v1 validation metric and both checkpoint formats work
end to end. This pilot deliberately compressed the optimizer schedule into 200
steps, so its loss is not a fair same-step comparison with the 35,680-step
exp117 reference. The next sanity check should retain the exp117 step count and
evaluation cadence and compare the first one or two validation points.
