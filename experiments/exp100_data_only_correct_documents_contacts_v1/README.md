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
2. **Score under the unmodified model.** We re-score `prefix + generated` with a
   `prompt_logprobs` pass (exp89's proven NLL path) to get the **full-vocabulary**
   NLL — *not* the masked distribution we sampled from. We persist both the
   **structure-section** NLL (the generated contact statements + `<end>`) and the
   whole-document NLL.

**Selection.** Per protein, keep the rollout with the lowest **structure-section**
NLL (the ordering the model finds most natural) as the regenerated document.

Ground-truth contacts are read straight from the document text (no pyconfind),
mapped seq-index → wrap-around position → token id for the mask. The grammar is
pure-Python and unit-tested ([`tests/`](tests)) — including a full seq→position→
text→parse round-trip — so it is validated without a model.

- **Inference** — [`gen_constrained_worker_vllm_tpu.py`](gen_constrained_worker_vllm_tpu.py):
  vLLM on iris TPU v5p-8, bf16 model (`…-bc3084/hf_bf16/step-35679`),
  `tensor_parallel_size=4`, sharded for the full run, resume-on-restart.
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

### Key risk

vLLM-on-TPU is proven here for `logprobs` / `prompt_logprobs` (exp89) but **not**
for per-request Python `logits_processors`. Phase 0 spikes that on a single
v5p-8 before the full run. If TPU rejects logits processors, the masked-sampling
pass falls back to a single GPU (the 1.5B bf16 model fits trivially; HF/vLLM
logits processors are fully supported there) while scoring stays on TPU. The
processor also **auto-detects** whether vLLM prepends the prompt to
`past_token_ids`, so it is robust to that cross-version difference.

## Run

```bash
# Stage B-prep (local, needs marinfold): 10 resampled prefixes per target
uv run python gen_prompts.py --targets data/targets.parquet -k 10 \
    --out gs://marin-us-east5/protein-structure/MarinFold/exp100_only_correct_contacts_v1_train/prompts

# Stage B (iris TPU, marinfold-free): constrained generate + NLL score
python gen_constrained_worker_vllm_tpu.py \
    --model gs://marin-us-east5/checkpoints/prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084/hf_bf16/step-35679 \
    --targets gs://.../exp100_only_correct_contacts_v1_train/targets.parquet \
    --prompts gs://.../exp100_only_correct_contacts_v1_train/prompts \
    --out     gs://.../exp100_only_correct_contacts_v1_train/runs/full \
    --shard 0/8 --n-rollouts 10 --tensor-parallel-size 4 --tpu-type v5p-8

# Stage C (local): aggregate + publish
uv run python aggregate_results.py --run gs://.../runs/full --out-prefix full
HF_TOKEN=<open-athena-scoped> uv run python publish_to_hf.py --run gs://.../runs/full
```

## Results

_Pending the Phase-0 spike + validation run (paused for compute go-ahead)._

## Conclusion

_Pending._
