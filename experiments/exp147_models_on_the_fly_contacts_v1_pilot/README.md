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
  `analyzed_from_row` + `build_document`, then encode with the existing
  contacts vocabulary and EOS.
- Pack each complete shard with best-fit decreasing and emit exactly 2,650
  examples. Uniformly sample packed bins if the shard is overfull; insert
  zero-loss examples if it is underfull. Cross-document attention remains
  blocked through segment IDs.
- Train from scratch with the best exp117 configuration: Qwen3 1.47B, sequence length 8192, global batch 128, LR 3.1623e-3, WD 0.2, cosine schedule, 10% warmup, AdamW betas 0.9/0.95, data seed 0.
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
