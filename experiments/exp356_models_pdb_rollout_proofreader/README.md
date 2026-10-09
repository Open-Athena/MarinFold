---
marinfold_experiment:
  issue: 356
  title: 'exp: bidirectional proofreading of MarinFold rollouts on experimental PDB structures'
  kind: models
  branch: codex/exp356-proofreader
---

# exp: bidirectional proofreading of MarinFold rollouts on experimental PDB structures

**Issue:** [#356](https://github.com/Open-Athena/MarinFold/issues/356) · **Kind:** `models` · **Branch:** `codex/exp356-proofreader`

## Question

Can an exp277-initialized bidirectional verifier identify incorrect contacts in actual MarinFold rollouts, using all supplied contacts, from one-contact prefixes through complete rollouts?

## Hypothesis

Fine-tuning a pretrained contact-language backbone on experimentally determined structures will teach contextual error detection and calibrated contact probabilities. Their mean estimates precision; a pooled readout estimates recall.

## Approach

Select 50,000 experimental PDB protein-chain structures from the existing curated PDB corpus, audit reference completeness, remove benchmark homologs, and split related structures together. Generate eight real exp277 step-266344 rollouts per training protein on CoreWeave. Preserve full prompts, emission order, end status and per-input timings. Crop before bidirectional encoding, oversampling one-contact and short prefixes. Train a separate exp277 backbone with full attention, a shared per-contact sigmoid classifier and recall head. Run correctness tests and a real GPU smoke before production, checkpoint resumably, and monitor/recover all jobs. Evaluate contact calibration, prefix precision/recall, contextual improvement and selection/aggregation against baselines. Publish the checkpoint with tokenizer and inference code, provenance, timings, reports and a PR. Eval-test remains unread.

## Success criteria

A completed production training run on 50,000 experimental chain structures, reproducible dataset/splits, validated attention and label alignment, resumable checkpoints, and held-out protein-group results at one-contact through full-rollout lengths. Report negative results honestly; do not promote an unvalidated model. Preserve the existing generator default.

## Results

The frozen target dataset contains 50,000 training chain structures from 46,933 PDB entries, 1,552 validation structures and 1,473 test structures. Training spans 34,894 distinct supplied sequences and 9,074 connected homology groups. The largest training group contains 409 structures. Validation and test contain 287 and 295 groups respectively. The public manifest records every target and split; related entries, exact sequences and connected groups are disjoint across splits.

Inputs contain only observed residues. A supplied sequence must match the deposited canonical sequence or one unique contiguous segment covering at least 90% of it, with at most 20 omitted residues at either terminus. Internal gaps are rejected. Median terminal omission is three residues across the selected targets. This admits unresolved terminal tags without labeling contacts to unobserved residues as negatives. Reference contacts are the complete exp222 native-only pyconfind contact set (3 Å side-chain threshold, minimum contact degree 0.001, sequence separation at least six). Recall refers to that supplied sequence's contact set.

The first generation smoke produced 384 rollouts across 48 structures: 381 passed strict parsing, with no empty rollouts. The one-H100 training smoke saved at step 6, reloaded model and optimizer in a separate job, reproduced its validation metrics exactly, and continued to step 12. This checks execution and recovery; it does not establish model quality. [W&B recovery smoke](https://wandb.ai/open-athena/MarinFold/runs/debug-exp356-recovery-v1).

Production inputs are frozen at `s3://marin-us-east-02a/MarinFold/exp356/data/v1/targets.parquet`; checksums and selection counts are in `data/targets_provenance.json`. Production rollout generation and eight-H100 training validation are in progress. The new held-out splits measure proofreading generalization; they do not establish absence from the backbone's pretraining corpus.

## Conclusion

_(Fill in after results are in.)_
