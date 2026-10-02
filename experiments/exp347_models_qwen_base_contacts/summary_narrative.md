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

The 0.8B training, checkpoint, and optimizer/data/RNG recovery checks passed.
Both 2B smoke tests and the 4B training/checkpoint smoke passed.
The approved 4B transfer is complete, with pinned revision and file checksums.
All six 1B-token production trials have been launched, capped at 48 H100s.
The 4B raw run hit GPU memory exhaustion, then recovered with gradient-buffer
reuse and allocator changes, reaching step 21 at about 27-34K tokens/second.
Frequent batch preemptions are repeatedly erasing unsaved work.
The recovery update saves after the first update of every attempt, then every
two minutes, and skips repeat validation at a restored production step.
Trial identities, data, batch size, and token budgets remain fixed.
All six use the corrected recovery code, including explicit training mode on resume.
At 21:10 UTC they are waiting through another set of batch preemptions.
None has finished; no contact-prediction improvement has been established.

## Early evidence and limits

Raw 0.8B validation contact-token NLL fell from 0.70685 to 0.59605 at 36.92M tokens.
Prompted 2B NLL fell from 1.46629 to 1.23746 by step 1250.
NLLs across the two tokenizations are not directly comparable.
A greedy generation canary ran on both 2B smoke checkpoints: all 16 outputs
hit the 256-token cap, produced zero valid pairs, and scored F1 zero.
This is not the FoldBench rollout-plus-resample benchmark.
No contact accuracy or causal pretrained-transfer advantage is established.
Per-input timings and worker metadata were saved at evaluation time.
