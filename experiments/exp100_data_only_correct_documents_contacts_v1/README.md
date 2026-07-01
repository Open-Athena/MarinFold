---
marinfold_experiment:
  issue: 100
  title: "exp: create contacts-v1 fine tuning set by sampling only-correct documents from base model"
  kind: data
  branch: claude/great-chatterjee-211197
---

# exp: create contacts-v1 fine-tuning set by sampling only-correct documents

**Issue:** [#100](https://github.com/Open-Athena/MarinFold/issues/100) · **Kind:** `data` · **Branch:** `claude/great-chatterjee-211197`

## Question

Ultimately: does fine-tuning on **regenerated** documents beat just re-epoching the
original contacts-v1 training data? This experiment produces the regenerated set.
For each training protein we draw rollouts from the base model under a hard
constraint — it may **only** emit *true* contacts — and keep the one the
*unmodified* model finds most likely. The fine-tuning comparison itself is a later
experiment; the success criterion here is "the regenerated training data is on the
HuggingFace bucket."

## Background

Eric's tuned contacts-v1 1.5B (issues [#61](https://github.com/Open-Athena/MarinFold/issues/61)/[#75](https://github.com/Open-Athena/MarinFold/issues/75),
eval loss **2.7566**) is the model. This builds directly on
[exp98](../exp98_data_generate_rollouts_contacts_v1_train) ([#98](https://github.com/Open-Athena/MarinFold/issues/98),
PR [#99](https://github.com/Open-Athena/MarinFold/pull/99)), which established the
iris/vLLM-TPU rollout path (bf16 model, `tensor_parallel_size=4`, resampled
prefixes, resume-on-restart) on the **train** split. exp100 reuses that whole
scaffold; the one new piece is **constrained ("only-correct") decoding** plus
selection by unmodified likelihood.

## Approach

For each target, **N=10** rollouts, each from a *fresh* contacts-v1 realization
(resampled N-terminus + statement order — exp82's settled recipe; `gen_prompts.py`,
`k=10`). Each rollout is generated in two passes on one shared LLM:

1. **Constrained generate.** A per-rollout logits processor
   ([`constrained_grammar.py`](constrained_grammar.py)) enforces the only-correct
   grammar at every step. The contacts-v1 tokenizer is **WordLevel** (1:1,
   whitespace-separated), so a statement is exactly the 3-token stream
   `[<contact>, <pi>, <pj>]` — no space tokens — i.e. a clean 3-token cycle:
   - statement start → force `<contact>` while true contacts remain, else `<end>`;
   - first endpoint → any position that is an endpoint of a still-remaining true contact;
   - second endpoint → any position that *completes* a still-remaining true contact with the pending first endpoint;
   - `<end>` is masked until **every** true contact has been emitted.

   So every finished document has **precision = recall = 1.0 by construction**; the
   only freedom is the order / orientation. (The worker still parses + scores each
   document and asserts 100%-correct as a built-in check.)
2. **Unmodified NLL, captured in-line.** The same logits processor reads the
   incoming logits *before* masking — that is the **full-vocabulary** next-token
   distribution — and records the log-prob of the token that actually gets
   realized, giving the rollout's **structure-section** NLL (the "original
   likelihood, full output vocabulary" the issue asks for) with **no second
   pass**. This deliberately avoids `prompt_logprobs`, which returns `None` on the
   iris JAX/TPU stack (it bit exp89), and works identically on GPU and TPU.

**Selection.** Per protein, keep the rollout with the lowest **structure-section**
NLL (the ordering the model finds most natural) as the regenerated document.

Ground-truth contacts are read straight from the document text (no pyconfind),
mapped seq-index → wrap-around position → token id for the mask. The grammar is
pure-Python and unit-tested ([`tests/`](tests)) — including a full seq→position→
text→parse round-trip — so it is validated without a model.

- **Inference** — two interchangeable workers writing the same outputs:
  [`gen_constrained_worker_hf_gpu.py`](gen_constrained_worker_hf_gpu.py) (HF
  transformers, local CUDA GPU — the **proven** path, used for validation) and
  [`gen_constrained_worker_vllm_tpu.py`](gen_constrained_worker_vllm_tpu.py) (vLLM
  on iris TPU v5p-8, `tensor_parallel_size=4`, for the eventual scale-out — its
  per-request `logits_processors` support on the JAX/TPU stack is still to be
  spiked). Both use the bf16 model (`…-bc3084/hf_bf16/step-35679`) and resume on
  restart. The GPU worker batches a target's N rollouts with **zero padding**:
  they share prompt length *and* generated length (`3*n_gt+1`), so one decode loop
  finishes all N on the same step.
- **Outputs (GCS working copy)** under `runs/<name>/`: `nll/<entry>.parquet`
  (per-rollout NLLs + correctness), `documents/<entry>.json` (selected document),
  `all_documents/<entry>.json` (all N verbatim), `timings/<entry>.csv`.
- **Publish (public)** — [`publish_to_hf.py`](publish_to_hf.py) consolidates the
  run and uploads `regenerated_documents.parquet` (the deliverable),
  `per_target_nll.parquet`, `rollout_nll_all.parquet` to the **public**
  `open-athena/MarinFold` bucket (`data/contacts-v1-train-only-correct-exp100/`),
  auth-free for Colab.

### Phasing

- **Phase 1 — validate** on exp98's exact 1000 targets (`data/targets.parquet`
  copied from exp98, so the only-correct documents are directly comparable to
  exp98's free rollouts): confirm 100%-correct + full-recall on every document,
  record the N=10 NLL spread, measure throughput/cost.
- **Phase 2 — scale** to the full unique-protein train set, publish, write up.

### Key risk (status)

The only-correct decode needs per-step logit masking + unmodified full-vocab
scoring. Two facts settled this:
1. **`prompt_logprobs` returns `None` on the iris JAX/TPU stack** (exp89) — so the
   NLL is captured *inside* the processor from the pre-mask logits instead of in a
   second pass (works on any backend).
2. **Per-request `logits_processors` on vLLM-`tpu_inference` is unproven** — still
   to be spiked. The **GPU path is fully proven** (below), so it is a guaranteed
   fallback for the whole experiment; the TPU worker is for scale-out only.

The processor also **auto-detects** whether the backend prepends the prompt to
`past_token_ids`, so it is robust to that cross-version difference.

## Run

```bash
# Stage B-prep (local, needs marinfold): 10 resampled prefixes per target
uv run python gen_prompts.py --targets data/targets.parquet -k 10 --out <prompts dir>

# Stage B — local GPU (proven path, used for validation)
python gen_constrained_worker_hf_gpu.py --model <local hf_bf16 dir> \
    --targets data/targets.parquet --prompts <prompts dir> \
    --out <run dir> --shard 0/1 --n-rollouts 10

# Stage B — iris TPU scale-out (once the logits_processor spike passes)
python gen_constrained_worker_vllm_tpu.py --model gs://.../hf_bf16/step-35679 \
    --targets gs://.../targets.parquet --prompts gs://.../prompts \
    --out gs://.../runs/full --shard 0/8 --n-rollouts 10 --tensor-parallel-size 4

# Stage C (local): aggregate + publish
uv run python aggregate_results.py --run <run dir> --out-prefix full
HF_TOKEN=<open-athena-scoped> uv run python publish_to_hf.py --run <run dir>
```

## Results

### Phase-0 validation (local A5000 GPU, real model)

The method is validated end-to-end against the tuned 1.5B (HF bf16). On a spike of
6 targets × 3 rollouts, and a 6-target × 10-rollout worker run:

- **Every constrained document is 100%-correct + full-recall** (`recall = precision
  = 1.0`), a well-formed contacts-v1 document (`<contacts-v1> … <begin_statements>`
  + exactly `n_gt` `<contact>` statements + `<end>`), with generated length exactly
  `3*n_gt + 1` — so the grammar and the seq→position→token coordinate mapping are
  correct.
- **The folded-in `struct_nll` matches an independent teacher-forced forward pass**
  to ~1 nat over hundreds of tokens — the unmodified-likelihood capture is right,
  and the `prompt_logprobs`-on-TPU problem is fully sidestepped.
- **Selection by lowest structure-NLL is meaningful** — the N-rollout NLL spread is
  real (best-of-10 clearly below the mean), so the "keep the most-likely ordering"
  step does something.
- Throughput ~380 tok/s at batch 10 on the A5000 (≈ 3.5 h projected for the full
  1000 × 10 validation).

_Full 1000-target validation run + aggregate/plots: in progress._

## Conclusion

_Pending the full validation run._
