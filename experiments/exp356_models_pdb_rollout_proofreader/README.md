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

The eight-H100 smoke completed four optimizer steps with global batch 64, saving a reloadable checkpoint and reducing validation loss from 0.913 to 0.684. Its checkpoint also passed the complete evaluation/reporting pipeline. These short smoke runs verify execution, not production quality. [W&B eight-H100 smoke](https://wandb.ai/open-athena/MarinFold/runs/debug-exp356-eight-h100-v1).

All 32 production generation shards completed, writing 3,328 batches (about 0.99 GB) for 53,025 structures and eight requested rollouts per structure. The final audit passed independent token-to-contact decoding, reference-label checks, duplicate accounting and target coverage. It retained 423,371 of 424,200 trajectories, excluding 703 malformed and 126 empty outputs; every structure retains at least one trajectory. The retained corpus contains 109,251,347 contact labels, including incorrect proposals.

Production training is running as [exp356-exp277-bidir-pdb50k-v1](https://wandb.ai/open-athena/MarinFold/runs/exp356-exp277-bidir-pdb50k-v1), on eight H100s in CoreWeave US-East-02A. Three epochs over 399,213 retained training rollouts require 18,714 optimizer steps at global batch 64. Validation runs every 500 steps; model, tokenizer and optimizer checkpoints are committed to co-located storage before recovery pointers advance. Final held-out quality is not yet measured.

The early step-500 checkpoint is not release quality. The complete validation pass covers all 1,552 structures and 287 related groups: mean within-protein full-rollout AUROC is 0.538, Brier score 0.241, precision MAE 0.183 and recall MAE 0.189. Probability-weighted aggregation gives R-precision 0.519 versus 0.517 for frequency alone; additional context worsens first-contact Brier by 0.0064. Detailed tables are under `data/validation_step500_*`. A preceding narrow preview on the first 128 sorted IDs is retained separately under `data/preview_step500_*`.

Training continues. On the fixed training-time validation sample, step 1,000 improves contact Brier from 0.235 at step 500 to 0.215, precision MAE from 0.273 to 0.239, and recall MAE from 0.077 to 0.051. These sample metrics are separate from the full evaluation above.

By step 2,000, the full validation pass shows useful contextual proofreading: mean within-protein AUROC is 0.728, full-rollout Brier 0.188, precision MAE 0.109 and recall MAE 0.114. Additional contacts improve first-contact Brier by 0.038 (group-bootstrap 95% interval 0.030–0.046). Keeping the highest-scored half gives precision 0.536 versus 0.407 for the first half in emission order. Selecting a rollout by estimated precision improves mean precision from 0.409 to 0.443. Probability-weighted aggregation still barely changes R-precision (0.519 versus 0.517). Tables are in `data/validation_step2000_*`; this is validation evidence, not the final test result.

A comparison preserves exp277's causal backbone as a frozen feature extractor and trains a four-layer, 512-wide bidirectional encoder over contact triples. Each triple contributes the backbone states of its three tokens; a global readout also receives the sequence-prompt boundary, the last supplied token and prefix-size features. This keeps pretrained causal features intact while still allowing later contacts to revise earlier judgments. Both architectures pass the 18-test suite, including attention behavior, frozen gradients, padding and save/load checks. The comparison passed an eight-H100 optimizer-recovery smoke and is training as [exp356-exp277-frozen-contact-pdb50k-v2](https://wandb.ai/open-athena/MarinFold/runs/exp356-exp277-frozen-contact-pdb50k-v2), with an initial pause at step 2,000 for comparison on the complete validation split. Its 22.18 million trainable parameters run at about 0.27 seconds per step. At step 1,000, its fixed-sample contact Brier is 0.171 and precision MAE 0.168; the complete evaluation will determine whether it improves contact ranking and contextual revision.

Production inputs are frozen at `s3://marin-us-east-02a/MarinFold/exp356/data/v1/targets.parquet`; checksums and selection counts are in `data/targets_provenance.json`. The target parquet and provenance are also public under `hf://buckets/open-athena/MarinFold/data/exp356/`. The new held-out splits measure proofreading generalization; they do not establish absence from the backbone's pretraining corpus.

## Conclusion

_(Fill in after results are in.)_
