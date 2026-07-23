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
reads public Hugging Face metadata but does not contact GCS or transfer data:

```bash
uv run stage_pilot.py --num-shards 16
```

After reviewing the reported source size and destination, apply that exact
plan explicitly:

```bash
uv run stage_pilot.py --num-shards 16 --execute
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
uv run iris --cluster=marin job run --no-wait \
  --cpu=1 --memory=2G --extra=cpu --zone=us-east5-a \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
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

The code-only stateless prototype is implemented. Synthetic Parquet tests cover
best-fit packing, exact quota selection, zero-loss padding, overflow,
deterministic random and repeated access, cache reconstruction, and the
single-component Levanter mixture. No real data has been transferred and no
training job has been launched.

## Conclusion

_(Fill in after results are in.)_
