## October 29 — both cooldown forks are complete

GPU final validation: 3.03572; TPU final validation: 3.03582. Both have exact native counters of 186885 updates at checkpoint 186884 and verified finite exported weights.

At 23:03 UTC, the 5e-4 / 1e-3 / 2e-3 parent runs are at 1.567 / 1.740 / 1.175 epochs of two. Latest validation losses: 3.07851 / 3.09601 / 3.34287. Active compute remaining: approximately 62 / 37 / 65 hours, plus interruptions and validation/checkpoint overhead.

The high-LR run remains unstable. Its requested learning rate is preserved.

## October 29 — new contact evaluation

Checkpoint steps: 282830 (5e-4), 314987 (1e-3), 207662 (2e-3), and final 186884 for both cooldown forks.

All-range R-precision, natural / designed: GPU cooldown 0.4630 / 0.6652; TPU cooldown 0.4602 / 0.6695; 5e-4 parent 0.4476 / 0.6248; 1e-3 0.4045 / 0.6009; 2e-3 0.2847 / 0.5457. The two cooldown forks are effectively tied on both all-range scores.

Full-data first epoch: 0.5551 / 0.6964. The GPU cooldown improves over its earlier 0.3862 / 0.5983 snapshot, but still trails the full-data reference. Exposure and schedules differ; the design set is small. All five evaluations use the same CUDA scoring recipe and cover 58000 stopped rollouts across 97 natural and 19 designed proteins; none needed extensions.

## Historical checkpoint — October 27 update

The one-epoch v5p fork finished its four-update repair October 26 at 13:43 EDT. Final validation: 3.03582. Native step 186884 and all optimizer counters confirm exactly 186885 total updates; every exported parameter is finite. Tokenizer and model files are verified.

Completed full-data first epoch: 2.98441; continuation: 2.98099. Different exposure and schedules remain comparison limits. W&B summary is reconciled from durable final metrics; the invalid historical endpoint is preserved.

GPU epochs / target and latest loss: 5e-4 1.177 / 2, 3.08887; 1e-3 1.345 / 2, 3.11757; 2e-3 0.826 / 2, 3.57571; cooldown fork 0.866 / 1, 3.08435. All resumed and advanced.

2e-3 has repeated loss spikes (peak 4.09309). Keep the requested LR. At active compute speed, the GPU fork needs about 37 hours and the two-epoch runs 93–119 hours; interruptions add time. The October 21 R-precision snapshot is unchanged; no final TPU R-precision is available.

## Historical recovery audit — October 26

GPU runs stopped October 22 following CoreWeave credential rotation. Reported success hid incomplete training. Last validation: 5e-4 3.09058; 1e-3 3.11904; 2e-3 3.15627; GPU fork 3.10000. Previous calendar ETAs are invalid.

TPU reached one epoch October 25, but the final padded update produced NaN parameters. Last finite scheduled validation: 3.03731 at step 186032. A tested padding-gradient fix preserves real-example gradients. Recovery restores step 186880 and replays exactly four updates; every exported parameter must pass a finite check.

TPU recovery a02 is queued for v5p-64 capacity. GPU parent a04 is starting after a real TensorStore cache read confirmed the credential fix. All four GPU recovery roots are submitted: 32 / 32 / 64 H100 for the LR sweep and 16 H100 for the GPU fork. New optimizer progress is not yet confirmed. R-precision results remain the completed October 21 snapshots below; no final TPU score is available.

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

The two-epoch runs are incomplete; see the October 26 recovery audit.

## Validation comparison — 2026-10-21

Redesign-only training is behind the completed native-plus-redesign baseline at matched steps on the same full LM validation set.

At step 101,472: full data 3.07419; redesign-only LR 5e-4 / 1e-3 / 2e-3: 3.11629 / 3.13460 / 3.15598.

Latest LR 5e-4 validation: loss 3.09300 at step 181,804 versus full data 3.06074 (+0.03226 nats/token). LR 1e-3 is at 3.12220 at step 205,058. The lowest LR continues to lead this sweep.

## Completed references and interpretation

Full-data first epoch: final loss 2.98441 at step 266,344. Additional full-corpus pass: 2.98099 at absolute step 479,417.

The continuation restores step 213,072 before cooldown; it is not an independent seed. Both completed trajectories include cooldown. The redesign-only two-epoch trials are still before cooldown; the TPU one-epoch fork entered cooldown on October 22.

The current runs also have longer warmup and amino-acid augmentation ramps. Equal optimizer steps match packed token slots, not actual tokens or proteins. The matched-step LM-validation gap and completed pre-cooldown contact evaluations both favor the full-data reference; the redesign-only trajectories are unfinished.

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

All four full evaluations completed: 46,400 rollouts, zero unfinished samples, and zero extensions needed. Thirty-seven CPU checks pass. Natural/design results and viral/low-MSA cuts stay separate.

## Contact evaluation: durable sources restored

CoreWeave writes recovered after the account suspension. All four refreshed native sources are frozen and independently verified: 124 files, 70.63 GB. All four CPU exports passed verification, including both weight shards and tokenizer files. All four H100 evaluations succeeded by 22:47 UTC on October 21, without failures or preemptions. Model bytes stayed regional.

The earlier attempt lost temporary source objects during repeated preemptions. Its diagnostics and snapshot are preserved; no full evaluation groups completed. The fork smoke capped 2/100 samples on 5sbj_A, motivating the approved continuation.

Full-data first-epoch reference: eval-val 0.55510 / 0.53706; eval-denovo 0.69640 / 0.67497 (all / long). The continuation reference had eight capped samples on 5sbj_A. The new full evaluations needed no extensions. No matched redesign-corpus KNN null is available.

## Contact results: natural proteins first

Eval-val, 97 natural proteins, all / long R-precision:

5e-4, step 184314: 0.39711 / 0.37791
1e-3, step 207865: 0.39562 / 0.37381
2e-3, step 105722: 0.38884 / 0.36465
GPU fork, step 146264: 0.38624 / 0.36627
Completed full-data default: 0.55510 / 0.53706

The redesign-only snapshots trail by 0.158–0.169 in all-range precision. The top two LR values are tied (difference 0.00148). Unequal exposure and unfinished cooldown prevent a final causal conclusion. Per-protein metrics, timings, bootstrap intervals and subgroup cuts are committed and public.

## Contact results: 19 designs

Eval-denovo, all / long R-precision:

5e-4: 0.59671 / 0.54092
1e-3: 0.59087 / 0.53839
2e-3: 0.59242 / 0.53713
GPU fork: 0.59834 / 0.55408
Completed full-data default: 0.69640 / 0.67497

The small design set is a sanity check. All four snapshots precede cooldown, including the checkpoint labeled as the one-epoch fork. TPU weights are not part of this evaluation snapshot.

## Historical training progress — October 22, 14:48 UTC

All five production runs are active. Epochs / target and latest validation loss:

5e-4 GPU: 1.072 / 2; loss 3.08945
1e-3 GPU: 1.186 / 2; loss 3.11922
2e-3 GPU: 0.618 / 2; loss 3.16114
GPU one-epoch fork: 0.791 / 1; loss 3.10318
v5p-64 one-epoch fork: 0.809 / 1; loss 3.10050

TPU cooldown has started. About 43 compute hours remain at current speed, putting the earliest finish on October 24; four prior preemptions show why calendar completion can be later. GPU fork: about 114 compute hours. Original two-epoch parent remains active and independent.
