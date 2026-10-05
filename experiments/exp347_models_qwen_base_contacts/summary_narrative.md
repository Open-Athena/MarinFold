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
It restored 78.14M tokens and has advanced beyond 137M tokens.
All five final checkpoints and their model/tokenizer/HF exports are verified.
The other five successful runs use the same training code.
The approved 4B model transfer is complete.
Both 4B pilot checkpoints now have complete eval-val contact evaluations.

## Validation likelihood and limits

Contacts-v1 final NLL: 0.51342 (0.8B), 0.50577 (2B), 0.49301 (4B).
Prompted final NLL: 1.15186 (0.8B), 1.06211 (4B).
These use the same 64 held-out AFDB documents; larger models score better.
The partial 2B prompted NLL is about 1.20954 at 131.8M tokens.
NLLs across the two tokenizations are not directly comparable.
A greedy generation canary ran on both 2B smoke checkpoints: all 16 outputs
hit the 256-token cap, produced zero valid pairs, and scored F1 zero.
This is not the FoldBench rollout-plus-resample benchmark.
No causal pretrained-transfer advantage is established.
Per-input timings and worker metadata were saved at evaluation time.

## Full-scale continuation in preparation

Requested: full-scale 4B training with periodic eval-val R-precision.
Working default: both formats, 100B additional tokens each, eval every1B.
This budget is an announced operator assumption; the existing48-H100 cap remains.
All2,067 co-located AFDB shards prepared:3.717M train and37,256 validation proteins.
One corpus pass contains18.17B raw or5.21B prompted tokens.
Planned allocation:16 training +4 evaluation H100s per4B arm, plus8 for the2B pilot.
Continue the respective1B weights with fresh optimizer and data cursor.
Both4B single-protein100-rollout evaluation smokes passed without truncation.
E8 reference passed: all R=.42438, long R=.36599; all554units, no truncated rollouts.
Two-node training/export passed. Exact optimizer resume and native IB are under test.
Full-corpus production is not yet launched.

## Periodic R-precision contract

Frozen97-protein eval-val only; eval-test is untouched.
100 rollouts, T=1, top-p=.95, no top-k; unchanged exp89 resolved-pair metrics.
Resample contacts-v1 N-terminal offsets and sequence-statement order.
Prompted format uses independent samples of a fixed natural-language prefix.
Translate6L+128 protein-token capacity to native Qwen token capacity.
Require every protein and no capped rollouts before publishing an aggregate.
Save per-protein timings, raw completions, votes, and checkpoint provenance.
Durable periodic exports survive rolling recovery-checkpoint deletion.
A persistent Iris driver logs all/long R-precision against checkpoint tokens.

## 4B eval-val baselines after 1B tokens

97 natural proteins, 100 completed rollouts per protein, no unfinished samples.
contacts-v1: all R=0.14010, long R=0.10083.
Prompted: all R=0.14350, long R=0.09894.
Differences below ~0.005 are ties at the evaluator's resolution.
Existing exp232 m2-p06: all R=0.51980, long R=0.50173 on the same set.
Full decontaminated AFDB KNN: all R=0.40715, long R=0.39211;
this is corpus context, not an exact null for the pilots' 512-shard subset.
One prompted rollout needed continuation of its original sampled prefix.
All97 per-protein metric/timing records per arm are in data/eval_val_pilot.
Small result tables are also public in the HF bucket.
The full-scale curves will show whether more protein training improves accuracy.
