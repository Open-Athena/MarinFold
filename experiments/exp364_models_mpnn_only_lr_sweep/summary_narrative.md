## ProteinMPNN-only learning-rate sweep

Experiment #364 asks whether the default 1.47B contacts-v1 model can train on redesigned sequences alone.

162,574,724 documents: 31,702,680 AFDB + 130,872,044 ESM-Atlas designs.

## Controlled training recipe

Three learning rates: 0.0005, 0.001, 0.002. Two complete epochs, trained from scratch.

Preserve exp277 architecture, tokenizer, batch 128, context 8192, optimizer settings, WSD schedule fractions, validation and amino-acid order augmentation.

No native training documents. Fresh global shuffle each epoch.

## Coverage and execution

23,921,189 packed examples per epoch; 373,769 updates for two passes.

Live regional audit passed: both source ledgers, raw tokens, packing and clipping exactly match the default corpus subset. Ten CPU tests, including the cooldown fork, passed.

The 64-H100 smoke passed ten updates with finite losses. Native checkpoint and HF weights plus tokenizer were verified.

All three production trials were launched on 32 H100s each, 96 total, using regional cached data. Saved W&B configurations match the default except for the requested changes.

All trials are training; the two-epoch runs are not yet complete.

## Validation comparison — 2026-10-21

Redesign-only training is behind the completed native-plus-redesign baseline at matched steps on the same full LM validation set.

At step 101,472: full data 3.07419; redesign-only LR 5e-4 / 1e-3 / 2e-3: 3.11629 / 3.13460 / 3.15598.

Latest LR 5e-4 validation: loss 3.09300 at step 181,804 versus full data 3.06074 (+0.03226 nats/token). LR 1e-3 is at 3.12220 at step 205,058. The lowest LR continues to lead this sweep.

## Completed references and interpretation

Full-data first epoch: final loss 2.98441 at step 266,344. Additional full-corpus pass: 2.98099 at absolute step 479,417.

The continuation restores step 213,072 before cooldown; it is not an independent seed. Both completed trajectories include cooldown; current redesign-only runs do not yet.

The current runs also have longer warmup and amino-acid augmentation ramps. Equal optimizer steps match packed token slots, not actual tokens or proteins. The gap is preliminary LM-validation evidence; contact accuracy has not been measured.

## One-epoch cooldown fork — 2026-10-19

Fork the leading 5e-4 trajectory from full-state checkpoint 139,053. Preserve the original two-epoch run independently.

Keep the first-epoch shuffle, data offset, optimizer state, and original augmentation ramp. Retain LR 5e-4 until step 149,508, then linearly cool toward 5e-5 over the final 20% of one epoch.

Stop after 186,885 total updates (final checkpoint 186,884): exactly 23,921,189 packed examples, with 37 real examples in the final batch. The fork adds 47,831 updates to the inherited prefix.

At Oct 21, the fork has reached step 145,821: latest validation 3.10221 at 143,752 versus its parent's 3.10134 at the same step. Cooldown has not started yet. The parent remains independent. All ten CPU tests pass.

Repeated batch-priority preemptions have delayed progress, especially for the fork. Automatic checkpoint resumes are working, but calendar finish estimates remain sensitive to GPU availability.

## Additional v5p cooldown copy: transfer verified
The independent TPU replica restores the same step-139053 full state and finishes at step 186884. Production requests v5p-64 (32 chips), us-east5-a, global batch 128 and batch priority. Existing GPU fork and parent are retained independently.

The approved 267.73 GB transfer finished at 15:59 UTC. All 21,936 objects passed independent inventory/checksum verification. The v5p-8 restore smoke passed ten updates (139054–139063) with finite losses and the expected accelerated LR decay. Final optimizer counters, both HF weight shards and tokenizer files were verified. Steady time: 27.09 s/update on four chips; its capped validation is not a full-data comparison.

Production was submitted at 16:33 UTC from commit 33cc077f and registered in W&B at 16:51:58 UTC. At 17:50 UTC it reached step 139702, with full validation loss 3.10391 and throughput about 244k token slots/s. Roughly 56 compute hours remain at that rate, before validation or interruptions. The source remains the original frozen checkpoint, not the smoke endpoint. Evidence: data/tpu_transfer_plan.json, tpu_input_verification.json, tpu_smoke.csv and tpu_smoke_verification.json.

## Contact evaluation: approved continuation

Score 97 natural eval-val proteins and 19 eval-denovo designs. Eval-test is excluded. Latest complete checkpoints: 184314 (5e-4), 207865 (1e-3), 105722 (2e-3), 146264 (one-epoch fork). Exposure is unequal; cooldown has not started.

Keep the original 100-rollout, temperature-1, top-p-0.95 recipe. Start at 6L+128; the user approved continuing capped samples. Preserve each capped sample's exact prompt and generated tokens, then sample only its suffix within the remaining 8192-token context. Record continuation seeds, budgets and lengths. Require every sample to terminate; never discard capped samples from the score.

Thirty-five CPU checks pass. Natural/design results and viral/low-MSA cuts stay separate. No complete new R-precision scores are available yet.

## Contact evaluation: durable sources restored

CoreWeave writes recovered after the account suspension. All four refreshed native sources are frozen and independently verified: 124 files, 70.63 GB. CPU export jobs are preparing reusable HF weights and tokenizers on CoreWeave. GPU retries will load those exports; model bytes stay regional.

The earlier attempt lost temporary source objects during repeated preemptions. Its diagnostics and snapshot are preserved; no full evaluation groups completed. The fork smoke capped 2/100 samples on 5sbj_A, motivating the approved continuation.

Full-data first-epoch reference: eval-val 0.55510 / 0.53706; eval-denovo 0.69640 / 0.67497 (all / long). The continuation reference also had capped design samples, so the token-budget difference must be stated. No matched redesign-corpus KNN null is available.
