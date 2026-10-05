# Qwen3.5 base models for protein contacts

## Question and experiment

Fine-tune pretrained Qwen3.5 base language models at 0.8B, 2B, and 4B.
Compare direct contacts-v1 documents against plain-language instructions,
ordinary amino-acid sequences, and one-based contact pairs.
Six trials, approximately 1B tokens each; at most 48 H100 GPUs.

## Data and comparison

Use exp232's existing CoreWeave copy of exp225-decontaminated AFDB documents.
Both formats contain the same sequence and contact information.
Keep the original Qwen tokenizer and vocabulary. No chat template.
The paired pool has 920,611 training and 9,235 validation proteins.
Exclude the same 5.11% of source rows when either complete document exceeds 16K.
The pool has 4.493B raw tokens versus 1.290B prompted tokens.
Equal token budgets imply different protein exposure.

## Current execution

At October 5, five of six production trials have finished their 1B-token budgets.
All three contacts-v1 trials and the 0.8B/4B prompted trials completed.
The 2B prompted trial failed October 3 at 79.61M tokens with a distributed
communication timeout; no preceding out-of-memory error was recorded.
A single controlled resume was submitted October 5, retaining its optimizer,
data cursor, RNG state, run identity, and scientific settings on eight H100s.
It restored 78.14M tokens and completed its first resumed update.
All five final checkpoints and their model/tokenizer/HF exports are verified.
The other five successful runs use the same training code.
The approved 4B model transfer is complete.
Full-budget contact-generation evaluation remains outstanding.

## Validation likelihood and limits

Contacts-v1 final NLL: 0.51342 (0.8B), 0.50577 (2B), 0.49301 (4B).
Prompted final NLL: 1.15186 (0.8B), 1.06211 (4B).
These use the same 64 held-out AFDB documents; larger models score better.
The partial 2B prompted NLL is 1.22514 at 77.14M tokens.
NLLs across the two tokenizations are not directly comparable.
A greedy generation canary ran on both 2B smoke checkpoints: all 16 outputs
hit the 256-token cap, produced zero valid pairs, and scored F1 zero.
This is not the FoldBench rollout-plus-resample benchmark.
No contact accuracy or causal pretrained-transfer advantage is established.
Per-input timings and worker metadata were saved at evaluation time.
