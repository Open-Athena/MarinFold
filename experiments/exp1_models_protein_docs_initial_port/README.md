---
marinfold_experiment:
  issue: 1
  title: "Initial port of marin/protein-training-1b modeling experiments"
  kind: models
  branch: main
  baselines: []
---

# Initial port of marin/protein-training-1b modeling experiments

**Issue:** [#1](https://github.com/Open-Athena/MarinFold/issues/1) · **Kind:** `models` · **Branch:** `main`

This is a bulk-port of the protein-docs modeling work that originated on
the `marin/protein-training-1b` branch
([reference](https://github.com/marin-community/marin/tree/protein-training-1b/experiments/protein)).
Rather than create a separate experiment per training script, the
entire family (size sweep, distance-masked vs unmasked ablation,
all-doc-types mixture, continuation script, and all matching HF
export scripts) is collected here as the starting state of MarinFold.

Future experiments will fork off this — e.g. a single training
recipe + analysis becomes its own `exp<N>_models_<slug>/`. The
distogram benchmark plumbing (`protein_distogram_eval.py`) and the
custom tokenizer (`create_protein_tokenizer.py`) are arguably
document-structure / evaluation concerns and will move once those
top-level subprojects have concrete interfaces to fit them into.

## Question

Reproduces the prior research direction: does training a small Llama
on the `protein-docs` `contacts-and-distances-v1-5x` corpus produce a
useful protein-structure LM, and what's the best loss formulation +
model scale for that?

## Hypothesis

Yes — distance-bin-only loss masking concentrates gradient signal on
the prediction of interest and should improve distogram-benchmark
performance over unmasked training at matched compute. Scaling
should follow Pythia-like curves at least up to 1.4B; beyond that
under-training (50K steps × batch 128) is the dominant constraint.

## Approach

All training is via Marin's executor on TPU v5p, pinned to
`us-east5-a` for bucket co-location. Each `train_protein_*.py`
script defines one ExecutorStep graph; running `python -m <script>`
dispatches to iris.

| Script | Model | Recipe |
| --- | --- | --- |
| `train_protein_30m_distance_masked.py`  | 30M (h=512, l=6)  | 50K steps, batch 128, LR 3.5e-4 |
| `train_protein_100m_distance_masked.py` | 108M (h=768, l=12) | 50K steps, batch 128, LR 3.5e-4 |
| `train_protein_100m_unmasked.py`        | 108M | unmasked variant |
| `train_protein_400m_distance_masked.py` | 383M (h=1024, l=24) | 50K steps, batch 128, LR 3.5e-4 |
| `train_protein_420m_deep_distance_masked.py` | 420M (h=768, l=48) | LR 2.5e-4 (deep nets are LR-sensitive) |
| `train_protein_1b.py`                   | 1.4B (h=2048, l=16) | original recipe, no loss mask |
| `train_protein_1b_unmasked.py`          | 1.4B | unmasked variant of the 7d355e recipe |
| `train_protein_1b_distance_masked.py`   | 1.4B | distance-masked, LR 1.05e-3, in-training distogram benchmark |
| `train_protein_1_5b_distance_masked.py` | 1.47B (l=24) | depth ablation against 1B |
| `train_protein_3b_distance_masked.py`   | 3B (h=2560, l=32) | bigger RAM budget; resumes existing run |
| `train_protein_1b_all_docs_unmasked.py` | 1.4B | mixes all 3 document types, v5p-32 |
| `continue_train_protein_1b_distance_masked.py` | 1.4B | resume the 3.5e-4 run via `override_output_path` |

`export_protein_<size>_<variant>.py` converts a trained Levanter
checkpoint to HuggingFace format under `.../hf/step-<N>/`.

Shared building blocks (all in this directory):

- `protein_train_common.py` — tokenizer pin, TPU resource config,
  shared tokenize steps, distance-bin-only loss-weight function,
  `build_distance_masked_train_step` and `build_hf_export_step`.
- `create_protein_tokenizer.py` — builds + pushes the custom
  WordLevel tokenizer.
- `protein_distogram_eval.py` — in-training distogram benchmark:
  registry of target PDBs + builder for pre-tokenized validation
  parquets.

## Compute estimate

- TPU v5p-8 × 1 for most runs, v5p-32 for the all-docs run.
- ~12 wall hours per 50K steps at batch 128 seq 8192 on v5p-8.

## Success criteria

For an individual size:
- Training reaches the configured step count without divergence.
- HF export produces a loadable checkpoint at the expected path.

For the size sweep:
- Loss vs. params follows a Pythia-like curve on
  `eval/protein-docs-cd-val/loss`.

For the loss-mask ablation:
- The distance-masked variant beats the unmasked variant on
  `eval/protein_dist/macro_loss` at matched checkpoint steps.

## Baselines

The original marin runs are the baselines — they're effectively the
same code with slightly different output paths. Notable run names:

- `protein-contacts-1b-3.5e-4-distance-masked-7d355e` — distance-
  masked 1B, LR 3.5e-4, in-flight via the continue script.
- `protein-contacts-1b-2.5e-4-780930` — original 1B unmasked.
- `protein-contacts-3b-distance-masked-ef3aa5` — 3B (resumed via
  `override_output_path`).

## Results

To be filled in as runs complete on the new MarinFold infrastructure.
Pre-MarinFold results live in W&B under the `marin` project
(group `protein-training`).

## Conclusion

Pending — this is the start state, not a result.
