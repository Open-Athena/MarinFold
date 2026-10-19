## ProteinMPNN-only learning-rate sweep

Experiment #364 asks whether the default 1.47B contacts-v1 model can train on redesigned sequences alone.

162,574,724 documents: 31,702,680 AFDB + 130,872,044 ESM-Atlas designs.

## Controlled training recipe

Three learning rates: 0.0005, 0.001, 0.002. Two complete epochs, trained from scratch.

Preserve exp277 architecture, tokenizer, batch 128, context 8192, optimizer settings, WSD schedule fractions, validation and amino-acid order augmentation.

No native training documents. Fresh global shuffle each epoch.

## Coverage and execution

23,921,189 packed examples per epoch; 373,769 updates for two passes.

Live regional audit passed: both source ledgers, raw tokens, packing and clipping exactly match the default corpus subset. Six CPU tests passed.

The 64-H100 smoke passed ten updates with finite losses. Native checkpoint and HF weights plus tokenizer were verified.

All three production trials were launched on 32 H100s each, 96 total, using regional cached data. Saved W&B configurations match the default except for the requested changes.

All trials are training; the two-epoch runs are not yet complete.

## Validation comparison — 2026-10-16

Redesign-only training is behind the completed native-plus-redesign baseline at matched steps on the same full LM validation set.

At step 38,052: full data 3.12896; redesign-only LR 5e-4 / 1e-3 / 2e-3: 3.18634 / 3.19716 / 3.20600.

At step 80,332: LR 5e-4 has loss 3.13276 versus full data 3.08213 (+0.05062 nats/token). The lowest LR currently leads this sweep.

## Completed references and interpretation

Full-data first epoch: final loss 2.98441 at step 266,344. Additional full-corpus pass: 2.98099 at absolute step 479,417.

The continuation restores step 213,072 before cooldown; it is not an independent seed. Both completed trajectories include cooldown; current redesign-only runs do not yet.

The current runs also have longer warmup and amino-acid augmentation ramps. Equal optimizer steps match packed token slots, not actual tokens or proteins. The gap is preliminary LM-validation evidence; contact accuracy has not been measured.

## One-epoch cooldown fork — 2026-10-19

Fork the leading 5e-4 trajectory from full-state checkpoint 139,053. Preserve the original two-epoch run independently.

Keep the first-epoch shuffle, data offset, optimizer state, and original augmentation ramp. Retain LR 5e-4 until step 149,508, then linearly cool toward 5e-5 over the final 20% of one epoch.

Stop after 186,885 total updates (final checkpoint 186,884): exactly 23,921,189 packed examples, with 37 real examples in the final batch. The fork adds 47,831 updates to the inherited prefix.

The source is frozen in same-region S3; restored optimizer counters verified. All ten CPU tests pass. Results are pending.
