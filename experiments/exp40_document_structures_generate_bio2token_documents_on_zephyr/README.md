---
marinfold_experiment:
  issue: 40
  title: "exp: Generate Bio2Token documents on Zephyr"
  kind: document_structures
  branch: main
---

# exp: Generate Bio2Token documents on Zephyr

**Issue:** [#40](https://github.com/Open-Athena/MarinFold/issues/40) · **Kind:** `document_structures` · **Branch:** `main`

## Question

Does a commonly used neural tokenizer, like bio2token, make useful documents for protein prediction? Can we make use of the compressed representation to supplement context to predict other useful information (like secondary structures)? Can we do this at scale on Marin infrastructure (iris, zephyr, via a data loader)?

## Hypothesis

- Documents should be smaller
- They should do about as well as our custom document format
- Zephyr should be equipped to perform neural inference en masse.

## Background

See [parent issue](https://github.com/Open-Athena/MarinFold/issues/2).

## Approach

- Add [bio2token](https://github.com/flagshippioneering/bio2token/tree/main) as a `uv` dep in this experiment's pyproject.
- Based on our model for PDBs, write an adapter for our structure of data to their [expected pdb dict format](https://github.com/flagshippioneering/bio2token/blob/e3139ba655aa71e2afd0904ef46679b2796815d9/src/bio2token/data/utils/utils.py#L300).
- Create an efficient Zephyr pipeline (see the [Zephyr agent skill](.agents/skills/zephyr-pipeline-performance/SKILL.md)) that adapts our data source to inference batches (via the previous step), and make it perform inference via the [bio2token encoder](https://github.com/flagshippioneering/bio2token/blob/main/src/bio2token/models/encoder.py). The encoder outputs a 1d tensor of integers (tokens).
- Hydrate the model with the official [bio2token checkpoint](https://github.com/flagshippioneering/bio2token/blob/main/checkpoints/bio2token/bio2token_pretrained/epoch%3D0243-val_loss_epoch%3D0.71-best-checkpoint.ckpt).
- Update the torch backend to [target XLA/TPUs](https://docs.pytorch.org/xla/master/learn/migration-to-xla-on-tpus.html).
- Write the token documents in a similar document format (i.e. parquet) with a similar chunking/shard structure as the [standard token documents](experiments/exp1_document_structures_contacts_and_distances_v1/README.md). 
  - See this parquet store for reference: `gs://marin-us-central1/protein-structure/MarinFold/exp5/corpus_v2-{shard:05d}-of-{total:05d}.parquet`

## Success criteria

- Documents can be generated on GCS that are ready for experiments.
  - "Ready for experiments" means is reviewed and approved by Tim.
- (reach): a Dataloader can efficiently apply bio2token on a source data store to generate documents on the fly.

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
