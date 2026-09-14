---
marinfold_experiment:
  issue: 288
  title: 'Experiment: Chinchilla size sweep on cleaned contacts-v1 corpora'
  kind: models
  branch: exp/288-chinchilla-size-sweep
---

# Experiment: Chinchilla size sweep on cleaned contacts-v1 corpora

**Issue:** [#288](https://github.com/Open-Athena/MarinFold/issues/288) · **Kind:** `models` · **Branch:** `exp/288-chinchilla-size-sweep`

## Question

For the cleaned native + MPNN-redesigned contacts-v1 corpora from #266/#274, what model size is compute/data efficient? In particular, do 0.7B, 1.5B, and 3B Qwen-style contacts-v1 models show the expected Chinchilla-style tradeoff between parameter count, token budget, validation cross-entropy, and downstream contact R-precision?

## Hypothesis

The new ~248B-token corpus is large enough that the current 1.5B recipe may be under-sized at fixed token budget. A 3B model should improve validation CE and possibly FoldBench R-precision over 1.5B if the extra MPNN/native data carries independent signal; a 0.7B model provides a cheap lower anchor for fitting a scaling trend rather than only comparing two large runs.

## Background

The training-ready corpora are:

- `contacts_v1_decontam/train`
- `contacts_v1_esm_atlas_decontam/train`
- `contacts_v1_mpnn_redesign/train`
- `contacts_v1_esm_atlas_mpnn_redesign/train`

Together they contain about 248B tokens. Prior MarinFold contacts-v1 work mostly optimized recipe around 1.5B models; older size comparisons are not a clean modern decontaminated/native+redesign sweep. This experiment is explicitly intended to gather Chinchilla-style scaling data for the next training recipe rather than to immediately maximize a single model.

The current working corpus accounting is:

| source | dataset | documents | tokens |
|---|---|---:|---:|
| native AFDB | `contacts_v1_decontam/train` | 3.963M | 4.43B |
| native ESM Atlas | `contacts_v1_esm_atlas_decontam/train` | 65.553M | 69.98B |
| MPNN-redesigned AFDB | `contacts_v1_mpnn_redesign/train` | 31.703M | 35.32B |
| MPNN-redesigned ESM Atlas | `contacts_v1_esm_atlas_mpnn_redesign/train` | 130.872M | 138.62B |
| **total** |  | **232.091M** | **248.35B** |

## Approach

Train a three-point model-size sweep on the same cleaned contacts-v1 mixture and recipe:

| arm | role | rough token/parameter ratio for one 248B-token pass |
|---|---|---:|
| ~0.7B | lower anchor for the scaling curve | ~355 |
| ~1.5B | current recipe scale / continuity with exp117-exp232 | ~166 |
| ~3B | first larger-model test suggested by the new corpus size | ~83 |

Hold dataset mixture, tokenizer, sequence length, optimizer recipe, validation set, and checkpoint/eval cadence fixed across sizes where possible. Record token budget, step count, GPU-hours, validation CE curves, and final/selected checkpoint quality. Evaluate promising checkpoints with the shared exp245 FoldBench rollout+resample contact benchmark.

This is a **measurement experiment**, not just a race to the best checkpoint. The primary deliverable is a small Chinchilla-style dataset for MarinFold: validation CE and contact quality as functions of parameters, tokens, and compute. The 0.7B arm is valuable even if it is not competitive, because it anchors the slope; the 3B arm is valuable even if it is expensive, because it tests whether the cleaned/redesigned corpus makes 1.5B under-sized.

## Planned measurements

For each arm, report:

- parameter count from the concrete model config;
- train tokens consumed and tokens/parameter at each durable checkpoint;
- wall-clock time, GPU-hours, and achieved tokens/sec;
- contacts-v1 validation CE vs tokens and vs GPU-hours;
- exp245 eval-val R-precision/AUC for selected checkpoints;
- eval-test only for a final candidate if the sweep changes the recommended recipe.

## Success criteria

- All three size arms have comparable W&B histories and durable checkpoints.
- The README reports token/parameter ratios, validation CE vs tokens, and validation CE vs compute for each arm.
- At least the best checkpoint from each completed arm is evaluated on exp245 eval-val; eval-test is reserved for final claims.
- The conclusion states whether 1.5B remains the practical recipe size or whether the evidence supports moving to 3B+.

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
