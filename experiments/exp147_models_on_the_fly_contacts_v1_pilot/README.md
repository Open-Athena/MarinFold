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

`StreamingPremadeContactsDataset` is now a small contacts-specific adapter over
the reusable `marinfold_models.streaming_documents.StreamingDocumentDataset`.
The generic layer owns shuffling, process partitioning, packing, and loader
state; the experiment supplies the premade-contact columns and a deterministic
row-to-`Document` function. Its `get_batch(indices)` uses index values only to
retain optimizer-step checkpoint boundaries, not as random-access document
addresses. The training configuration therefore sets `shuffle=False` and
`mixture_block_size=1`; shard and row shuffling live inside the stream.

The checkpoint state covers the shard and row-block cursors, partial and ready
packing bins, the non-consuming shape-inference peek, and exact snapshots from
before each prefetched optimizer step. Open-bin documents are represented by
source-row references and deterministically reconstructed on restore, so scorer
callbacks never need to be serialized. Configuration fingerprints include a
versioned generator ID and process topology.

`save_checkpoint(path, step=N)` and `load_checkpoint(path)` are implemented and
round-trip tested, including restoring an earlier optimizer boundary after
later batches were prefetched. Each JAX process needs a separate loader-state
sidecar, and the process count must remain fixed. The remaining integration
before a preemptible run is a small hook in the training entrypoint that calls
`save_model_checkpoint_sidecar(...)` whenever Levanter commits model checkpoint
`N`, and calls its load counterpart before constructing the resumed
`DataLoader`; stock `levanter.main.train_lm` has no user callback surface for
this yet.

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
during loader initialization, packed utilization, exact loader-state restore,
and recovery of an optimizer-step state from behind Levanter-style prefetch.
No real data has been transferred and no training job has been launched.

## Conclusion

_(Fill in after results are in.)_
