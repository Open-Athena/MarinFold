---
marinfold_experiment:
  issue: 188
  title: 'eval SNR characterization'
  kind: evals
  branch: exp/188-eval-snr-characterization
---

# eval SNR characterization

**Issue:** [#188](https://github.com/Open-Athena/MarinFold/issues/188) · **Kind:** `evals` · **Branch:** `exp/188-eval-snr-characterization`

## Question

How much finite-validation-set noise is there in the canonical contacts-v1 eval
loss? In particular, are the ~0.001–0.01 nat differences used to compare nearby
checkpoints meaningfully above the document-level sampling noise of the eval set?

## Hypothesis

The full contacts-v1 validation set is large enough that the token-weighted loss
stderr should be small, but not obviously small enough to treat sub-0.01 nat
differences as meaningful without measurement. A document-level bootstrap should
produce a more honest uncertainty estimate than treating individual tokens as
independent.

## Approach

This experiment depends on the shared Poisson-bootstrap helper introduced in PR
[#187](https://github.com/Open-Athena/MarinFold/pull/187). The bootstrap unit is a
validation document:

1. Score the exp117 contacts-v1 checkpoint on the canonical contacts-v1 validation
   set, emitting one row per document with `loss_sum` and `token_count`.
2. Compute the usual token-weighted eval loss, `sum(loss_sum) / sum(token_count)`.
3. Draw document weights `w_i ~ Poisson(1)` and recompute the weighted loss across
   10k bootstrap replicates.
4. Report the bootstrap stderr and compare it to representative eval-loss deltas
   from exp117/exp169 (~0.001 and ~0.0076 nats).

The first committed script, [`summarize_per_doc_loss.py`](summarize_per_doc_loss.py),
consumes any per-document loss table with `loss_sum` and `token_count` columns and
writes the bootstrap summary JSON. The model-scoring script will be added on this
branch once the shared helper PR lands or this branch is run directly against it.

## Success criteria

- Per-document loss table for exp117 contacts-v1-val.
- Bootstrap stderr for token-weighted eval loss.
- Short README conclusion on whether ~0.001 / ~0.01 nat deltas are meaningfully above eval noise.

## Results

_Not run yet._

## Conclusion

_Fill in after results are in._
