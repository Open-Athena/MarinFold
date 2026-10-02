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
protein exposure. GPU training validation is in progress; no contact
accuracy or pretrained-transfer advantage has yet been established.
