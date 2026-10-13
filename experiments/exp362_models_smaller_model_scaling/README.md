---
marinfold_experiment:
  issue: 362
  title: 'exp: smaller-model scaling on the full default corpus'
  kind: models
  branch: exp/362-smaller-model-scaling
---

# exp: smaller-model scaling on the full default corpus

**Issue:** [#362](https://github.com/Open-Athena/MarinFold/issues/362) · **Kind:** `models` · **Branch:** `exp/362-smaller-model-scaling`

## Question

Does model scale still improve contact prediction at the current 1.47B default, after training three smaller models to completion on the identical full corpus?

## Hypothesis

Validation cross-entropy and contact-prediction accuracy will improve with parameter count through the 1.47B baseline, providing evidence about whether training larger models is worthwhile. A plateau is also informative.

## Background

The default is exp277's first-epoch `contacts-v1-exp277-m2-p06-full-epoch-1.5B`, final step 266344, not its second-epoch continuation. See #277 and LATEST_RESULTS.md. The approximate 250M-document corpus is exactly 232,090,905 native plus ProteinMPNN documents, containing 248,583,762,834 raw tokens and 34,092,146 packed examples.

## Approach

Train three scratch-initialized Qwen3 models with 4, 8, and 16 layers (approximately 250M, 500M, and 1B parameters), versus the existing 24-layer 1.47B control. Change only depth within the model configuration. Preserve hidden size 2048, intermediate size 8192, attention heads 32 / KV heads 8, QK norm, RoPE, tokenizer, sequence length 8192, global batch 128, AdamW LR 0.001 / weight decay 0.2, WSD warmup 10% / decay 20% / minimum LR ratio 0.1, seeds, finite globally shuffled full-corpus traversal, blocked cross-document attention, and scheduled amino-acid order augmentation.

Each production trial must complete all 266,345 optimizer updates (final checkpoint step 266344); smoke runs are separate identities. Record exact parameter counts and executable configuration comparisons to the baseline. Preserve resume state and publish final HF exports with tokenizers.

Try TRC TPU placement first, requiring verified region-local complete inputs. If that cannot work without an unapproved bulk transfer or fails operationally, use existing CoreWeave inputs and batch-priority GPUs. Do not move more than 10 GB across regions without explicit authorization.

Compare final/full learning curves at matched training exposure, and score eval-val and eval-denovo with the exp82/exp245 fixed rollout recipe and paired uncertainty against exp277. Do not access held-out eval-test. Report the scope explicitly: this is a fixed-recipe depth scaling experiment, with one seed per size; extrapolation beyond the measured range is suggestive, not definitive.

## Success criteria

All three runs complete the entire corpus, with reachable native final checkpoints and HF exports including tokenizers. Commit training/evaluation summaries, per-input evaluation timings, plots and summary PDF. Report whether loss and contact accuracy continue improving up to 1.47B and quantify the observed increments without treating an extrapolation as a measured larger-model result.

## Results

Setup: the exact sizes are 254,941,696; 498,228,224; and 984,801,280 parameters, versus exp277's 1,471,374,336. All 26 configuration and finite-dataset tests plus eight fleet/persistence tests pass. The full resolved configuration matches exp277 after normalizing depth and operational identities.

TRC preflight inspected the configured GCS regions and found no complete regional MPNN caches. The CoreWeave fallback reuses the exact existing inputs without a corpus transfer. The launch and worker environments retain exp277's committed dependency lock.


_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
