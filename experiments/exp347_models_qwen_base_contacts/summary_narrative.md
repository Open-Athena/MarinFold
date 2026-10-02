# Qwen3.5 base models for protein contacts

## Question and experiment

Fine-tune pretrained Qwen3.5 base language models at 0.8B, 2B, and 4B.
Compare direct contacts-v1 documents against plain-language instructions,
ordinary amino-acid sequences, and one-based contact pairs.
Six trials, approximately 1B tokens each; at most 48 H100 GPUs.

## Data and comparison

Use exp232's existing CoreWeave copy of exp225-decontaminated AFDB documents.
The two formats contain the same sequence and contact information.
Keep Qwen's original tokenizer and vocabulary. No chat template.
Exclude a protein from both formats if either complete document exceeds 16K.
Hold out sequence clusters, with no use of eval-test for model development.

## Current evidence and limits

The complete corpus contains 920,611 training and 9,235 validation documents.
5.11% of source examples exceeded 16K; both formats exclude the same rows.
The training pool contains 4.493B contacts-v1 tokens and 1.290B prompted tokens. Equal token budgets imply different
protein exposure. The 0.8B training, checkpoint, and optimizer/data/RNG
recovery checks passed. Both 2B training/checkpoint smoke tests also passed.
All four 0.8B/2B production trials are launched, requesting 32 H100s.
At 18:50 UTC, all are waiting for batch admission after three preemptions.
The saved 0.8B contacts-v1 checkpoint can resume at step 253.
Idle-node totals included cordoned nodes; no cluster configuration was changed.
The 4B trials await approval for the combined 15.6 GB model transfer.
No contact accuracy or pretrained-transfer advantage has yet been established.

## Validation and interpretation

Teacher-forced contact likelihood uses 64 held-out AFDB documents.
Greedy generation diagnostics retain malformed, empty, and capped outputs.
They are not the established FoldBench rollout-plus-resample benchmark.
Per-input timings are saved at evaluation time, including worker metadata.
The first raw-format run initially sustained roughly 39–53K tokens/second
over eight H100s. These are engineering observations, not accuracy results.
At 36.92M tokens, its contact-token NLL improved from 0.70685 to 0.59605.
