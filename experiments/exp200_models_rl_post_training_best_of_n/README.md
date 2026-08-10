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

### Phase 3 — RL loop bring-up (in progress)

Everything up to the training loop is done and verified. The loop itself runs but
does not yet complete a run.

**The reward mechanism is confirmed on real hardware.** Reading a rollout batch
written by a TPU worker: 4 groups of 4 (uniform, which the replay buffer's
rectangular indexing requires), prompt 413 / response 433 / logprobs 433 /
token_rewards 433 (all aligned), `episode_reward` 0.1533, and `token_rewards`
spanning −0.0669 to 0.2665 with 432/433 nonzero and **not constant**. Those two
extremes are the design arithmetic exactly — `(1 − p̄)/3 = 0.267` and
`−p̄/3 = −0.067` at the observed p̄ ≈ 0.20. Environment, dense reward, weight
transfer and serialization all work.

**Training completes, and the loop is healthy.** W&B for the nano run:
`finished`, 10/10 steps, `train/max_advantages` 0.4746 (dense advantages reach
the loss), `train/mean_advantages` 0.0028 (≈0, which is exactly what centring the
reward on p̄ is designed to produce), `train/ratio_mean` 1.0024 (sampler and
trainer logprobs agree, so clipping is inert and the logprob path is validated),
`train/kl_k3_mean` 0.00052 with `kl_beta` 0.01.

**The bottleneck is rollout supply, not weight transfer.** Throughput metrics:
`train_step_duration` **1.14 s** against `rollout_wait_duration` **36.1 s** of a
37.3 s iteration. An earlier reading of this as a weight-transfer cost was wrong
— it inferred step time from the spacing between rollout batches, which measures
the rollout worker's whole cycle rather than the training step. The lever is
`num_rollout_workers` (now 4).

**Runs cannot terminate themselves, and the cause is upstream.** The
object-storage trace ruled out the obvious explanations first: **one boot id, one
worker_id, zero failures** across 106 rollout batches, so nothing was restarting,
and the `weight_step` cycling `−1 → 4 → −1` was just the weight client falling
back to its "no weights yet" sentinel after the trainer stopped serving.

The coordinator *does* have reaping logic — `train_job.wait()` then
`_terminate_rollout_jobs()`. It never runs because the completion handshake has no
safety on either side:

| side | code | failure mode |
|---|---|---|
| trainer | `runtime.run_state.mark_completed.remote().result()` (`orchestration.py:308`) | **no timeout** — blocks forever if the RPC does not land |
| rollout | `get_snapshot.remote().result(timeout=5.0)` inside `except Exception: pass  # best-effort` | a persistent failure is indistinguishable from "still running" — loops forever |

Either way `train_job.wait()` never returns and nothing reaps the rollout workers.
Measured directly: trace1 logged steps 2→9 and its W&B runtime stopped at **688 s**,
while the rollout worker kept generating for another 40+ minutes.

Two hypotheses were checked and rejected on the way, which is worth recording so
nobody re-runs them: hosted actors are served on a background thread
(`serve_background()`), so the coordinator blocking in `wait()` is not a deadlock;
and the trainer's only background thread is the replay buffer's, which is
`daemon=True`, so a lingering non-daemon thread is not holding the process open.

`reap.py` therefore detects completion through W&B — the channel that demonstrably
works — and stops the job from outside. Fixing the handshake itself would mean
patching marin, whose RL module was deleted upstream two days after this pin.

#### Bring-up failures, and why the earlier gates could not catch them

Five distinct failures, each found by the 10-step nano rather than by a
three-arm sweep:

| # | Failure | Why it was invisible earlier |
|---|---|---|
| 1 | marin deleted the `vllm` extra and `marin.rl` (`e7ef104402`, 2026-08-07) while iris rejects clients older than 14 days | Packaging; surfaces only at pod build |
| 2 | `WANDB_API_KEY` never propagated to workers | `create_environment` forwards it from `os.getenv` of the *calling* process, so the chain works only if the driver was launched with it |
| 3 | `canonical_model_name` is both a renderer substring match and an exact `MODEL_MAPPINGS` key | The exact-key lookup is on the weight-transfer path, which pure generation never touches — the Phase 1 gate ran 2,216 rollouts through the same context without hitting it |
| 4 | `prng_key` is a union of JAX key and plain int, selected by `use_jax_rng = (inference_type == "levanter")` | marin's own `mock_env` calls `jax.random.randint` unguarded, so the union is invisible until you run vLLM inference |
| 5 | `sync_interval_steps=1` costs 372 s/step | Only measurable once the loop actually ran |

None of these were reachable from the Phase 1 parity gate, which exercises
generation and scoring rather than the RL loop. That is an argument for the nano
gate, not against parity: the two cover different surfaces.

### Published artifacts

- **Checkpoint** — [`timodonnell/plm-exp163-refine-cv1-1_5b-lr1e-4-e1-cos-tpuF-step404`](https://huggingface.co/timodonnell/plm-exp163-refine-cv1-1_5b-lr1e-4-e1-cos-tpuF-step404),
  bf16, `rope_theta=500000`, `vocab_size=2845`, id 7 = `<contacts-v1.multi>`,
  verified anonymously loadable. Sourced from the open-athena bucket, **not** the
  GCS bf16 dir: only the bucket copy carries exp163's renamed tokenizer.
- **Training pool** —
  `gs://marin-us-east5/protein-structure/MarinFold/exp200/train/{targets.parquet,prompts/}`
- **Parity evidence** — `data/parity_all554.json`, `data/parity_all554_rollouts.csv`

## Reproducing

This directory is the complete iris workspace (exp166's pattern) — no marin
checkout, and the iris CLI comes from its own venv. The bundle is built from
`git ls-files`, so **commit before submitting**; `_submit.check_clean` enforces it.

```bash
uv sync --extra cpu --extra test
PYTHONPATH=../../marinfold uv run pytest tests/ -q
```

```bash
uv run python dispatch_parity.py --limit 554 --n-generations 4 --max-sections 0
```

```bash
uv run python dispatch_prep.py --n 10000 -k 16
```

```bash
uv run python dispatch_publish.py
```

```bash
EXP200_LRS=1e-6,3e-6,1e-5 EXP200_STEPS=150 uv run python dispatch_rl.py --submit
```

```bash
uv run python read_trace.py --path gs://marin-us-east5/protein-structure/MarinFold/exp200/trace/<run-name>
```

The marin pin is `0.2.76.dev31155643335` and **cannot be advanced**: 0.2.77
(2026-08-08) is the first release without `marin.rl`, which is
`ContactsDenseLoss`'s base class and the whole environment API. marin's RL
direction is now SkyRL, so moving forward is a rewrite rather than a bump.

Capacity note: v5p-16 had 0 ready slices in both zones on 2026-08-09 while v5p-8
had 103 in us-central1-a, so training runs on v5p-8 at `train_batch_size=32`.
marin's DAPO normalisation divides by batch size, so a learning rate swept at
batch 32 does not transfer to a batch-128 run.

## Conclusion

_(Fill in after results are in.)_
