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
- `contacts_env.py` — `ContactsV1RLEnv(MarinEnv)`; one instance per lesson, plain and multi at equal curriculum weight.

marin's vLLM path renders every prompt through a chat template, and this vocab has neither a chat template nor `<|im_end|>`. The context class is picked by a hardcoded `if inference_type == "vllm"` in `rollout_worker.py`, so a subclass cannot be injected. Instead the environment builds prompt token ids itself and calls `inference_ctx.llm.generate` with `TokensPrompt` — the renderer is touched nowhere outside `batch_completions`, and its constructor validates nothing, so setting `canonical_model_name` to a qwen-containing string is enough to get past construction. This is also what exp163's validated `gen_rollouts_worker_exp163.py` does, which makes the Phase-1 parity check a like-for-like comparison rather than a comparison against a reimplementation.

Starting model: `checkpoints/plm-exp163-refine-cv1-1_5b-lr1e-4-e1-cos-tpuF/hf/step-404`.

## Success criteria

- **Primary** — paired Δ best-of-N F1 on the 553-protein exp163 eval set ≥ **+0.02 at ≥3σ**, against the arm-F reference **re-measured at the same `--max-sections` cap** (the published 0.303 was measured uncapped, so comparing to it directly would be wrong by construction).
- **Guardrail** — teacher-forced R-precision in plain `<contacts-v1>` mode ≥ **−0.005** vs arm F's 0.3374.
- **Reported, not gating** — last-section F1 and consensus-vote F1, since best-of-N is an oracle metric; plus `n_sections`, `mean_jaccard`, `frac_improving` and a token-matched independent best-of-N control.

Kill criteria: mean contacts per section below 60% of baseline (reward hacking toward silence), `mean_jaccard` above 0.3 (diversity collapse), or R-precision Δ below −0.01 (the exp163 v1/v2 forgetting failure).

## Results

### Phase 1 — sampling-path parity (PASSED)

All 554 eval proteins x 4 rollouts, uncapped, on one v5p-8 (job
`/bizon/exp200-parity-all554`, 47m). Source: [`data/parity_all554.json`](data/parity_all554.json),
[`data/parity_all554_rollouts.csv`](data/parity_all554_rollouts.csv).

exp200 generates through `ContactsV1RLEnv` and scores by walking **token ids**;
exp163 generated through its own worker and scored by regexing **decoded text**.
Running both over the same 2,216 rollouts is a check on two independent
implementations, not a smoke test.

| metric | exp200 | #163 §4.1 | delta |
|---|---|---|---|
| best_f1 | 0.3015 | 0.3025 | −0.0010 |
| last_f1 | 0.2456 | 0.2493 | −0.0037 |
| first_f1 | 0.1849 | 0.1840 | +0.0009 |
| n_sections | 14.23 | 14.99 | −0.76 |
| mean_jaccard | 0.0677 | 0.0710 | −0.0033 |

Agreement between the two scorers: **max |best_f1 delta| 0.0**, **max |section_f1
delta| 0.0**, 0/2216 malformed prompts. Free-generation termination reproduced
independently at 0.569 against the published ~0.56.

3/2216 rollouts disagree on section COUNT, all of them truncated at the token
budget: exp163's regex keeps a trailing empty section after the final
`<begin_statements>` where the token walk does not. `best_f1` is identical on all
three — an empty section scores zero and never wins a max — so no score is
affected.

Measured per-contact precision is **0.2294**, which sets `p_bar`; the configured
starting value of 0.30 is close enough not to bias the first steps much.

One earlier run on 100 proteins read best_f1 0.1498 and looked like a 50%
regression. It was a sampling bug, not a model result — `--limit` truncated the
target list instead of sampling it, returning 100% foldbench100, where exp163's
own numbers give 0.1296 against 0.2928 on the denovo_pdb rows that are 71% of the
file. Kept as [`data/parity_100_foldbench_biased.json`](data/parity_100_foldbench_biased.json).

### Phase 2 — RL training pool (built)

10,000 proteins x 16 resampled realizations, built in 69s on one CPU pod in
us-east5 (job `/bizon/exp200-prep-pool`). Source:
[`data/train_pool_summary.json`](data/train_pool_summary.json).

`gs://marin-us-east5/protein-structure/MarinFold/exp200/train/{targets.parquet,prompts/}`

| | |
|---|---|
| source | exp53 contacts-v1 train split, AFDB **round 0** (highest pLDDT) |
| mean global pLDDT | 89.0 (min 70.5) |
| L | 31 / 159 / 359 / 512 (min / median / p90 / max) |
| n_gt | 5 / 105 / 139 (min / median / mean) |
| distinct entry ids | 10,000 |
| distinct struct clusters | 10,000 |
| exact sequence overlap with eval554 | **0** |

The zero-overlap number is a real measurement, not a filter that silently never
fired: both sides use the same one-letter alphabet (eval554 additionally contains
`X` for unknown residues), so a match could have fired. Length distribution also
lands close to the eval set (median 159 against eval554's 161), so the training
and evaluation protein populations are comparable in the axis that most drives
contact difficulty.

Homology-level overlap is **not** addressed — only exact sequence identity. exp41
(foldseek train-similarity) is the tool if that gap needs closing.

### Phase 3+ — RL training

_(Pending.)_

**Known prerequisite: the checkpoint needs an HF repo id.**
`vLLMInferenceContext.__init__` calls `levanter.tokenizers.load_tokenizer` on
`VLLMEngineConfig.model_name`, and that resolver accepts only a local directory,
a `mirror://` ref, or an HF Hub repo id — a `gs://` URL raises
`HFValidationError`. The trap is that vLLM *itself* streams weights from GCS
happily (`load_format="runai_streamer"`), so the weights path works and only the
tokenizer path fails, inside a rollout worker after the gang has scheduled.
`rl_config.check_engine_model_path` now rejects it at config-build time.

Note this is specific to the *engine*. `build_worker_configs` resolves
`RLJobConfig.tokenizer` once in the coordinator and ships the object to both
workers, so that half accepts anything loadable at submit time.

The fix is to publish exp163's arm-F export — weights plus the renamed tokenizer
where id 7 is `<contacts-v1.multi>` — as an HF **model repo** and pass the repo
id. It currently lives only in the open-athena *bucket*
(`checkpoints/plm-exp163-refine-cv1-1_5b-lr1e-4-e1-cos-tpuF/hf/step-404`), and
bucket paths are not repo ids. Creating the repo needs an org-scoped token.

## Conclusion

_(Fill in after results are in.)_
