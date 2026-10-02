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

The first-shard smoke preserved cyclic residue indexing and all contact pairs.
98 of 1,887 source rows exceeded 16K; one had no contact targets.
The 1,766 retained training documents contain 11.30M raw-format tokens
and 3.24M prompted-format tokens. Equal token budgets imply different
protein exposure. GPU training validation is in progress; no contact
accuracy or pretrained-transfer advantage has yet been established.
