---
marinfold_experiment:
  issue: 200
  title: 'RL post-training: best-of-N over self-generated contact candidates'
  kind: models
  branch: claude/marinfold-issue-200-plan-5ba0c8
---

# RL post-training: best-of-N over self-generated contact candidates

**Issue:** [#200](https://github.com/Open-Athena/MarinFold/issues/200) · **Kind:** `models` · **Branch:** `claude/marinfold-issue-200-plan-5ba0c8`

## Question

[#163](https://github.com/Open-Athena/MarinFold/issues/163)'s multi-draft model writes ~15 near-disjoint candidate contact maps in one generation. The best of them scores F1 **0.303** while the last — the one a caller would actually read — scores **0.249**, and the base E8 single-shot scores 0.237. Does RL over self-generated candidates close that gap, and does it do so without damaging the ordinary contacts-v1 task?

## Hypothesis

A **dense** reward will move this where a trajectory-level one would not. Every emitted `<contact> <pI> <pJ>` triple is independently checkable against ground truth, so the credit assignment can be per-contact rather than one scalar smeared over ~4,200 tokens. Two terms:

1. **Stepwise**, per contact: `+(1 - p̄)` if the pair is a true contact, `-p̄ · δ^e` if not, where `e` counts earlier errors in that section. The decay reflects that a wrong contact may be the logical consequence of an earlier mistake rather than an independent error. `p̄` is the policy's own recent precision, so the expected stepwise reward is ~0 at current performance and the gradient says only "beat yourself".
2. **Document-level**, per rollout: the best section's F1, baselined leave-one-out across the generations for that protein. This is the term that pays for *spread* — without it the stepwise term alone would push every section toward one best guess and destroy the diversity that makes best-of-N work.

RL runs on a 50:50 mix of `<contacts-v1>` and `<contacts-v1.multi>` prompts, so the base task is optimized directly rather than merely defended.

## Background

`<contacts-v1.multi>` (token id 7 under exp163's renamed tokenizer) is not part of the `marinfold` library — it is an exp163 artifact. `<begin_statements>` means "discard the previous candidate, here is a new one", and only the final section is closed by `<end>`, so `<end>` remains the generation stop token. See exp163's [WRITEUP.md](../exp163_models_teach_contacts_v1_to_refine_a/WRITEUP.md).

Two prior results constrain the design. exp163's v1/v2 refiners lost **41-44%** of base-task R-precision to a single full fine-tune, so a KL anchor and a low learning rate are not optional. And exp163 found conditioning on *external* drafts degrades prediction monotonically (-0.117 F1 at k=16), which is why the candidates here must be on-policy and self-generated.

## Approach

Fully online `RLJob` (`marin.rl`): vLLM rollout workers and a train worker with live weight transfer, on marin iris v5p (interactive band).

- `contact_rewards.py` — the dense reward, walking **token ids** rather than decoded text so rewards land on specific positions. Per-section F1 is verified equal to exp163's `rollout_metrics.score_rollout`, so the document return is the same number the published #163 figures came from.
- `dense_loss.py` — `ContactsDenseLoss(RLOOLoss)`, returning one per-token advantage array per rollout: `A_t = lam_step · token_rewards[t] + lam_doc · (R_doc − RLOO baseline)`.
- `raw_completion_ctx.py` — a raw-completion inference context; marin's vLLM path renders every prompt through a chat template, and this vocab has neither a chat template nor `<|im_end|>`.
- `contacts_env.py` — `ContactsV1RLEnv(MarinEnv)`, two curriculum lessons (plain / multi) at equal weight.

Starting model: `checkpoints/plm-exp163-refine-cv1-1_5b-lr1e-4-e1-cos-tpuF/hf/step-404`.

## Success criteria

- **Primary** — paired Δ best-of-N F1 on the 553-protein exp163 eval set ≥ **+0.02 at ≥3σ**, against the arm-F reference **re-measured at the same `--max-sections` cap** (the published 0.303 was measured uncapped, so comparing to it directly would be wrong by construction).
- **Guardrail** — teacher-forced R-precision in plain `<contacts-v1>` mode ≥ **−0.005** vs arm F's 0.3374.
- **Reported, not gating** — last-section F1 and consensus-vote F1, since best-of-N is an oracle metric; plus `n_sections`, `mean_jaccard`, `frac_improving` and a token-matched independent best-of-N control.

Kill criteria: mean contacts per section below 60% of baseline (reward hacking toward silence), `mean_jaccard` above 0.3 (diversity collapse), or R-precision Δ below −0.01 (the exp163 v1/v2 forgetting failure).

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
