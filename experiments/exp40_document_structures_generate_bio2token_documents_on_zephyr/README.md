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

See parent issue.

## Approach

Adapt previous zephyr scripts in this repo (in a new experiment) to perform inference with bio2token.

## Success criteria

- Documents can be generated on GCS that are ready for experiments.
  - "Ready for experiments" means is reviewed and approved by Tim.
- (reach): a Dataloader can efficiently apply bio2token on a source data store to generate documents on the fly.

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
