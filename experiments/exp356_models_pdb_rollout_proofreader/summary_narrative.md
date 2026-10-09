# Summary slides — exp: bidirectional proofreading of MarinFold rollouts on experimental PDB structures

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Can an exp277-initialized bidirectional verifier identify incorrect contacts in actual MarinFold rollouts, using all supplied contacts, from one-contact prefixes through complete rollouts?

## Why

Fine-tuning a pretrained contact-language backbone on experimentally determined structures will teach contextual error detection and calibrated contact probabilities. Their mean estimates precision; a pooled readout estimates recall.

## Results so far

CoreWeave generation smoke: 48 complete experimental chain structures, eight actual exp277 rollouts each. All 384 completed; 381 passed strict parsing. No empty rollouts. Twenty-four emitted pairs were invalid contacts and remain negative examples.

Behavioral tests pass: later evidence affects earlier judgments, padding is invisible, prefixes exclude future tokens, duplicate contacts do not inflate metrics, and saved heads/tokenizers round-trip.

The one-H100 smoke resumed from a durable step-6 checkpoint, reproduced validation metrics exactly, and trained through step 12. The eight-H100 smoke completed four steps with global batch 64; validation loss fell from 0.913 to 0.684. Its checkpoint also completed the evaluation/reporting pipeline. No production-quality claim yet.

All 32 production generation shards completed. The full audit passed: 423,371 retained trajectories across all 53,025 structures, with 109.3 million contact labels. Malformed and empty outputs were excluded; every structure retains usable trajectories.

Production training is running on eight H100s: exp356-exp277-bidir-pdb50k-v1. Three epochs over 399,213 training rollouts, global batch 64, 18,714 steps. Validation selects the final checkpoint; no final held-out quality claim yet.

## Production data and validation

Frozen: 50,000 training structures from 46,933 PDB entries, containing 34,894 distinct supplied sequences in 9,074 related groups. Validation: 1,552 structures / 287 groups. Test: 1,473 structures / 295 groups.

Every supplied residue is observed: canonical sequences or unique contiguous segments with at most 20 terminal residues omitted per end and at least 90% coverage. Internal gaps and truncated contact references are rejected. Benchmark homologs are excluded, and explicit searches check split boundaries.

Test precision, recall and calibration from one contact through full rollouts. Measure whether later evidence improves the first contact's judgment and whether proofreading improves contact selection.

## Early checkpoint: not ready for release

The complete step-500 validation covers 1,552 structures in 287 related groups. It remains weak: mean full-rollout contact AUROC 0.538, Brier 0.241, precision MAE 0.183 and recall MAE 0.189. Weighted aggregation R-precision is 0.519 versus 0.517 for frequency alone.

Later contacts do not yet improve the first contact's judgment. This checkpoint is not a production-quality model. Training continues: the fixed validation sample improves further at step 1,000, with contact Brier 0.215, precision MAE 0.239 and recall MAE 0.051. Sample diagnostics and complete evaluations are reported separately.

## Step 2,000: contextual proofreading emerges

On all 1,552 validation structures, mean full-rollout contact AUROC reaches 0.728 and Brier improves to 0.188. Precision MAE is 0.109 and recall MAE 0.114.

Later contacts improve first-contact Brier by 0.038 (group-bootstrap 95% interval 0.030–0.046). Keeping the highest-scored half gives precision 0.536, versus 0.407 for the first half in emission order. Quality-based rollout selection improves precision from 0.409 to 0.443.

Frequency aggregation remains hard to improve: weighted R-precision 0.519 versus 0.517. These are validation results; the final test split remains reserved.

## Faster architecture comparison

The alternative freezes exp277's causal features and trains a four-layer, 512-wide bidirectional contact encoder. It still uses later contacts to revise earlier judgments. Trainable parameters: 22.18 million; total: 1.488 billion.

Eighteen tests pass. An eight-H100 smoke reloaded model and optimizer and reproduced validation exactly. Training is about four times faster (0.27 seconds per step). The first 1,000 steps already give fixed-sample contact Brier 0.171 and precision MAE 0.168.

The complete validation comparison selects the frozen-backbone model: AUROC 0.763 versus 0.728, Brier 0.167 versus 0.188, precision MAE 0.096 versus 0.109, recall MAE 0.103 versus 0.114.

## Selected architecture: frozen causal features, bidirectional readout

At step 2,000, later context improves first-contact Brier by 0.0176 (95% group-bootstrap interval 0.0123–0.0224). Keeping the highest-scored half gives precision 0.570, versus 0.407 for the first half in emission order. Weighted aggregation improves R-precision from 0.517 to 0.528; rollout selection improves precision from 0.409 to 0.452.

The selected model resumes its existing three-epoch schedule from step 2,000. Full training, final validation, reserved test evaluation and publication remain in progress.
