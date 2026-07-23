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
- Shuffle shards per epoch and partition them disjointly across JAX processes.
  Within each shard, shuffle bounded row blocks so reads retain parquet
  locality.
- Reconstruct canonical causal serialized documents on demand with
  `analyzed_from_row` + `build_document`, then encode with the existing
  contacts vocabulary and EOS.
- Feed generated documents into a bounded online best-fit packer. Partially
  filled bins persist across loader calls; bins are emitted when they reach
  the target fill fraction or when the bounded reservoir must make progress.
  Cross-document attention remains blocked through segment IDs.
- Train from scratch with the best exp117 configuration: Qwen3 1.47B, sequence length 8192, global batch 128, LR 3.1623e-3, WD 0.2, cosine schedule, 10% warmup, AdamW betas 0.9/0.95, data seed 0.
- Evaluate the regular full `contacts_v1` validation set normally under
  `eval/tokenized/contacts-v1-val/loss`.

### Prototype contract and consequences

`StreamingPremadeContactsDataset` is a deliberately stateful adapter behind
Levanter's random-access `AsyncDataset` interface. Its `get_batch(indices)`
uses the number of requested examples but does not interpret the index values.
The training configuration therefore sets `shuffle=False` and
`mixture_block_size=1`; shard and row shuffling live inside the stream.

This is sufficient to inspect the data path and run a non-resuming pilot, but
it is not yet safe for a long preemptible run:

- Levanter prefetches many batches ahead of the optimizer.
- The shard cursor, row-block cursor, partial packing bins, and prefetched
  examples are not included in training checkpoints.
- Resuming therefore restarts the data stream rather than reproducing the
  exact next batch.
- The current loader assumes every JAX process receives at least one staged
  shard and that the process count remains fixed for a run.

Before a long run, either the loader/prefetch state must become checkpointed at
an optimizer-step boundary or packing must become a deterministic function of
a batch identifier.

## Success criteria

1. A local one-shard data-loader smoke test proves on-the-fly documents are byte/token identical to the published companion documents.
2. Online packing reaches at least 98% real-token utilization on a representative
   premade-contacts shard without random row reads across shards.
3. A short TPU pilot starts, sustains training without input starvation, and
   reports the standard `contacts-v1` validation metric.
4. The run is recorded in `history/runs/` and linked here.

## Results

The code-only prototype is implemented. Synthetic parquet tests cover
deterministic construction, carry-over of partial bins, process-disjoint shard
assignment, the special non-consuming `getitem_async` peek Levanter performs
during loader initialization, and packed utilization. No real data has been
transferred and no training job has been launched.

## Conclusion

_(Fill in after results are in.)_
