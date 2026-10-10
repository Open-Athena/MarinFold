# Summary slides — exp356 rollout proofreading

## Question and delivered model

Predict correctness for each contact in an actual MarinFold rollout, allowing later contacts to revise earlier judgments. Support a single contact and variable-length prefixes. Also estimate current precision and recall.

Released checkpoint: exp356-exp277-frozen-contact-pdb50k-v2/step-18000. Training and reserved-test evaluation are complete. Public download and inference verification remains in progress.

## Architecture

Frozen exp277 step-266344 causal Qwen3 features feed a four-layer, 512-wide bidirectional contact encoder. Each contact contributes all three token states. Global features include the sequence-prompt boundary, last supplied token and prefix size.

22.18 million trainable parameters; 1.488 billion total. Per-contact sigmoid probabilities, mean unique-pair precision and a pooled recall head. Inputs are cropped before encoding; future tokens and labels are absent.

## Experimental data

50,000 training chain structures, 46,933 PDB entries, 34,894 distinct supplied sequences and 9,074 connected related groups. Validation: 1,552 structures / 287 groups. Reserved test: 1,473 structures / 295 groups.

All 32 generation shards completed. Audit retained 423,371 real rollouts, including 399,213 training trajectories and 109.3 million contact labels across all splits. Reference completeness and token/label alignment were checked. Every target retains trajectories.

## Training and checkpoint selection

Eight CoreWeave H100s, three epochs, 18,714 optimizer steps, global batch 64. Prefix sampler: 25% full, 25% short, 50% uniform. Frozen features beat full-backbone bidirectional fine-tuning at matched step 2,000 and ran about four times faster.

Validation selected step 18000 from the predeclared candidate pool; its eight-prefix loss is 0.431166. The choice was frozen before evaluating the reserved test. Storage recovery preserved exact model, optimizer and schedule state.

## Reserved-test quality

All 1,473 test structures / 295 groups. Full-rollout AUROC 0.7933; Brier 0.1438. Precision MAE 0.0521; recall MAE 0.0528. One-contact Brier 0.1350.

At 64 supplied contacts, first-contact Brier improves by 0.0067 (95% group-bootstrap interval 0.0047–0.0088), excluding end markers. Keeping the top-scored half yields precision 0.5788, versus 0.4019 for emission order.

## Downstream use and limits

Selecting a rollout by predicted precision gives 0.4504, versus 0.4033 for random selection. Weighted aggregation R-precision 0.5135, versus 0.4987 for frequency alone.

Measured scope: exp277 single-chain rollouts, 32–1,000 residues, temperature 1, top-p 0.95. Proofreader splits are held out; generator-pretraining exclusion is not established. Other generators, complexes and edited rollouts remain unvalidated.

## Reproducibility and release checks

Public checkpoint includes tokenizer, code, pinned dependencies and hashes. Raw rollouts, targets, split provenance, per-rollout tables, plots and predictor timings are public.

Twenty-one behavioral tests pass. Optimizer recovery was exercised on one and eight GPUs. The final release check will download the checkpoint anonymously and run shipped inference offline for one contact and a complete rollout.
