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
It subsequently stopped at 332.0M tokens; its 330.25M-token checkpoint is intact.
All five final checkpoints and their model/tokenizer/HF exports are verified.
The other five successful runs use the same training code.
The approved 4B model transfer is complete.
Both 4B pilot checkpoints now have complete eval-val contact evaluations.

## Validation likelihood and limits

Contacts-v1 final NLL: 0.51342 (0.8B), 0.50577 (2B), 0.49301 (4B).
Prompted final NLL: 1.15186 (0.8B), 1.06211 (4B).
These use the same 64 held-out AFDB documents; larger models score better.
The latest partial 2B prompted NLL is about 1.17544.
NLLs across the two tokenizations are not directly comparable.
A greedy generation canary ran on both 2B smoke checkpoints: all 16 outputs
hit the 256-token cap, produced zero valid pairs, and scored F1 zero.
This is not the FoldBench rollout-plus-resample benchmark.
No causal pretrained-transfer advantage is established.
Per-input timings and worker metadata were saved at evaluation time.

## Full-scale continuation

Requested: full-scale 4B training with periodic eval-val R-precision.
Working default: both formats, 100B additional tokens each, eval every 1B.
This budget is an announced operator assumption; the existing 48-H100 cap remains.
Correction: the first scale-up was AFDB-only and has been stopped.
Current source pool: 235,490,905 documents in 9,112 audited shards (exp343).
Native AFDB/ESM + both MPNN redesign corpora + 3.4M complex train documents.
Stream and tokenize online; preserve lineage holdouts and paired 16K filtering.
235.49M is before filtering, not a measured eligible-protein count.
Allocation: 8 training + up to 4 periodic evaluation H100s per 4B arm.
The partial 2B pilot is inactive; October 21 interim evaluations used 12 workers each.
Continue the respective 1B weights with fresh optimizer and data cursor.
Both 4B single-protein 100-rollout evaluation smokes passed without truncation.
E8 reference passed: all R=0.42438, long R=0.36599; all 554 units, no truncated rollouts.
Two-node training/export passed, but socket networking was slower than one node.
The pinned image lacks libibverbs; production uses the proven 8-GPU profile.
Source-stream GPU recovery passed: restored loss exactly; new step8 checkpoint verified.
Both corrected phases stopped October 5 at 204.0M / 44.9M additional tokens.
Eval every 1B additional tokens; baseline 0 is the corresponding 1Bpilot.
Checkpointed 90-day jobs; the 100B budget may outlast that execution window.

## Checkpoint recovery — October 19

The stalled scalar collective was the checkpoint barrier.
All optimizer shards uploaded; rank-zero model/tokenizer uploads were incomplete.
Nested storage retries could outlast the ten-minute NCCL timeout.
Fix: bounded per-operation retries and an eight-minute checkpoint storage budget.
Previous recovery pointers survive failed uploads; add pre-timeout stack dumps.
36 local tests, Ruff, Pyrefly and a real CoreWeave storage probe passed.
Both resumes advanced and saved complete production checkpoints.

## Restart memory repair — October 21

Raw later failed during its first backward pass after checkpoint restore.
First-update accumulation retained independent gradients before DDP bucket views.
Synchronize the first microbatch early; retain the global token-mean objective.
Two-rank regression covers restored Adam state, gradients, updates, and shared storage.
Both corrected 8-H100 runs passed the first restored update and saved checkpoints.
Verified raw step4621 and prompted step12157, including all8 optimizer shards.
37 tests, Ruff and Pyrefly pass; implementation commit d2836a26.
Interim eval-val completed: raw step4620/498.97M; prompted step12051/450.59M tokens.
Prompted was briefly paused for scoring capacity, then resumed as a04.
Its new step12219 full recovery checkpoint was independently verified at 14:47 UTC.

## Startup metadata repair — October 22

Later preemption restarts stalled after W&B initialization, before training began.
Seven ranks reached the first token-count reduction; rank zero never joined it.
Neither small run-registration object was updated during the failed attempts.
The startup metadata write had escaped the earlier bounded checkpoint-I/O repair.
Registration and eval requests now use a 60-second budget and 20-second attempts.
Diagnostics start before startup bookkeeping; all 37 tests and storage probes pass.
New Iris jobs require explicit CoreWeave credentials; the valid cw profile is verified.
Recovery checkpoints are intact: raw step5229/+574.65M; prompted step12538/+476.91M.
East submissions were canceled before handoff after the authorized RNO2A move.
October 21 R-precision remains the latest completed measurement.

## Regional recovery — October 22

The user approved RNO2A compute with all new checkpoints written in that region.
CoreWeave confirms marin-rno2a BucketRegion=RNO2A; storage PUT/GET succeeds.
Recovery checkpoints, BF16 exports, requests, results and timings all use that bucket.
Evaluation drivers and their GPU workers also use RNO2A.
Existing source URIs preserve the catalog hash, seeded row order and saved cursors.
Old recovery checkpoints are read once; subsequent restarts prefer local successors.
All 39 tests, Ruff and Pyrefly pass, including rejecting inconsistent resume pointers.
A separate eight-H100 replay passed two restored updates and exited successfully.
Verified step5231 / 574.90M tokens: 20 objects, 50.49GB, all eight optimizer shards.
Production identities and the 100B targets / 1B evaluation cadence stay unchanged.
Both production runs now train on eight RNO2A H100s each.
Verified first local saves: raw5230 / +574.765M; prompted12539 / +476.967M tokens.
No new checkpoint objects were written to the original east roots after relocation.

## Periodic R-precision contract

Frozen 97-protein eval-val only; eval-test is untouched.
100 rollouts, T=1, top-p=.95, no top-k; unchanged exp89 resolved-pair metrics.
Resample contacts-v1 N-terminal offsets and sequence-statement order.
Prompted format uses independent samples of a fixed natural-language prefix.
Translate 6L+128 protein-token capacity to native Qwen token capacity.
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
All 97 per-protein metric/timing records per arm are in data/eval_val_pilot.
Small result tables are also public in the HF bucket.
The full-scale curves will show whether more protein training improves accuracy.

## Interim full-corpus R-precision — October 21

Each checkpoint: 97 eval-val proteins, 100 completed rollouts, zero unfinished.
Raw step4620, +498,970,942 tokens: all R=0.14481; long R=0.10787.
Prompted step12051, +450,586,475 tokens: all R=0.15239; long R=0.10738.
Changes from pilots: raw +0.00471 / +0.00704; prompted +0.00889 / +0.00844.
Accuracy remains low; raw overall gain is below the approximate 0.005 resolution.
Long-range scores are tied; unequal token/protein exposure limits format comparisons.
Only 0.45–0.50% of the 100B additional-token targets has been evaluated.
One prompted sample needed two continuation rounds; all samples terminated.
Metrics, per-protein timings, dispatches, and provenance are committed and public.
Periodic eval-val remains every 1B additional tokens; eval-test is untouched.
