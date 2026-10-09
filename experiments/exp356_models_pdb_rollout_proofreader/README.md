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

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
