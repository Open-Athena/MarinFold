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
  - [`gen_constrained_worker_hf_gpu.py`](gen_constrained_worker_hf_gpu.py) — HF
    transformers, local CUDA GPU. Masks + captures NLL in one in-process decode
    loop; batches a target's N rollouts with **zero padding** (they share prompt
    length *and* generated length `3*n_gt+1`, finishing together). Used for the
    validation run.
  - [`gen_constrained_worker_vllm.py`](gen_constrained_worker_vllm.py) — vLLM, the
    **portable scale-out** path (GPU now, **iris TPU** after plugin registration).
    Masks via a custom vLLM **structured-output backend**
    ([`only_correct_backend.py`](only_correct_backend.py)) — the only per-step
    masking hook the marin TPU stack honors — and recovers unmodified NLL via a
    `prompt_logprobs` pass. ~2× the HF worker's throughput on the same GPU.

  Both use the bf16 model (`…-bc3084/hf_bf16/step-35679`) and resume on restart.
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

### Backends & the iris/TPU port (resolved)

Source review of the pinned marin `tpu_inference` (rev 29faff43) + vLLM V1 fork
(571de56) settled how to run this on TPU:

- **Custom `LogitsProcessor.apply()` is NOT invoked on TPU** (nor are
  `logit_bias`/`allowed_token_ids`/`bad_words`). So the HF worker's per-step
  logits-processor approach can't run there.
- **The structured-output grammar bitmask IS applied on-device in JAX** — the one
  live per-step masking hook. So the constraint is expressed as a custom vLLM
  **`StructuredOutputBackend`** ([`only_correct_backend.py`](only_correct_backend.py))
  whose `fill_bitmask` emits our FSM's allowed-token set each step.
- **`prompt_logprobs` IS supported** on this `tpu_inference` rev (the old exp89
  "None on TPU" note is stale) and is computed from **raw pre-mask logits**, so a
  separate scoring pass recovers the unmodified NLL.

The structured-output path is **GPU-proven** (`gen_constrained_worker_vllm.py`
+ `only_correct_backend.py`: 10/10-correct, NLL matching the HF worker, ~2×
throughput). It is the **same code** on TPU — only registration differs.

**iris/TPU launch (proven).** vLLM picks the structured-output backend by a
hardcoded if/elif in `StructuredOutputManager` (no plugin hook), so
`only_correct_backend.register()` monkeypatches `grammar_init` (pre-setting our
backend, then delegating) and the validation hook. That manager runs in the
EngineCore process, so we set **`VLLM_ENABLE_V1_MULTIPROCESSING=0`** to make the
engine in-process — then the in-worker `register()` reaches it, with **no
entry-point install needed** (iris bundles the launch workspace, so staged files
ship as-is). The marin TPU stack is **vLLM 0.20.1** (not the 0.11.2 in local
caches); `register()` is written to handle both layouts. Recipe: stage the worker
+ its 3 sibling modules into the marin checkout and launch

```bash
uv run --no-sync iris --cluster marin job run --no-wait --enable-extra-resources \
  --tpu=v5p-8 --memory=64GB --disk=64GB --cpu=16 --extra=vllm --extra=tpu \
  --zone=us-east5-a -e VLLM_ENABLE_V1_MULTIPROCESSING 0 \
  -- python exp100_iris/gen_constrained_worker_vllm.py \
     --model gs://marin-us-east5/checkpoints/…/hf_bf16/step-35679 \
     --targets gs://…/exp100_only_correct_contacts_v1_train/targets.parquet \
     --prompts gs://…/exp100_only_correct_contacts_v1_train/prompts \
     --out gs://…/exp100_only_correct_contacts_v1_train/runs/full_tpu \
     --shard 0/1 --n-rollouts 10 --tensor-parallel-size 4
```

Calibration (3 targets) confirmed 10/10-correct on a v5p-8 with NLLs matching the
GPU path, ~1350 tok/s after the one-time XLA compile (77 s).

**Full run on iris (1000 × 10, one v5p-8).** All **1000/1000 selected documents
100%-correct** (mean 9.985/10 rollouts correct; 0 targets with <1 correct), NLLs
matching the GPU run (best-of-N struct NLL mean 971 / median 660; mean−min spread
19.1). Cost: 4.30 M tokens, **1,226 tok/s (tp=4)**, **~1.04 v5p-8-hours** (gen
0.97 + score 0.07) — ~3× the A5000's per-unit rate, and shardable across many
v5p-8 (exp98 used 8×). Per-target table: `data/full_tpu_per_target.csv`.

One caveat, fully mitigated: under vLLM 0.20's fill/accept scheduling at scale our
incremental FSM state occasionally drifts (~0.15 % of rollouts hit a spurious
"grammar rejected" → truncated), which the **select-among-100%-correct** rule
absorbs (a truncated rollout can't win selection). The
**history-authoritative, tolerant grammar** (`only_correct_backend.py`: resync
from the true token stream, never terminate the request) removes even those on
subsequent runs.

## Run

```bash
# Stage B-prep (local, needs marinfold): 10 resampled prefixes per target
uv run python gen_prompts.py --targets data/targets.parquet -k 10 --out <prompts dir>

# Stage B — local GPU (proven path, used for validation)
python gen_constrained_worker_hf_gpu.py --model <local hf_bf16 dir> \
    --targets data/targets.parquet --prompts <prompts dir> \
    --out <run dir> --shard 0/1 --n-rollouts 10

# Stage B — vLLM (GPU now; iris TPU after registering the plugin). In-process
# engine on GPU so the backend monkeypatch reaches EngineCore:
VLLM_ENABLE_V1_MULTIPROCESSING=0 python gen_constrained_worker_vllm.py \
    --model <hf_bf16 dir> --targets data/targets.parquet --prompts <prompts dir> \
    --out <run dir> --shard 0/1 --n-rollouts 10
# iris TPU: install this dir as a package exposing only_correct_backend.register
# as a vllm.general_plugins entry point, then --tensor-parallel-size 4 per the
# exp89/exp98 iris recipe.

# Stage C (local): aggregate + publish
uv run python aggregate_results.py --run <run dir> --out-prefix full
# publish_to_hf.py uploads via `hf buckets`, which needs huggingface_hub>=1.5 (the
# uv-locked venv here has 0.36.2). Use an hf CLI >=1.5 whose token is authed to an
# account in the open-athena org; --dry-run builds the parquets locally without
# uploading. The correctness gate in _read_document keeps only n_gt==0 or
# prec==rec==1.0 docs, so the published set is 100%-correct by construction.
uv run python publish_to_hf.py --run <run dir>            # add --dry-run to preview
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
- Throughput ~380–420 tok/s at batch 10 on the A5000.

### Full validation run (1000 targets × 10 rollouts = 10,000 documents, local A5000)

- **10,000 / 10,000 rollouts 100%-correct + full-recall** (0 warnings) — every
  target yields 10 valid, fully-correct contacts-v1 documents.
- **Cost:** 4.30 M generated tokens, ~419 tok/s, **2.85 GPU-hours** (~2.86 h wall)
  on one A5000. (An H100 / TPU or multi-GPU shard would cut this substantially.)
- **NLL selection is meaningful:** best-of-10 structure NLL averages **18.9 nats**
  below the per-target mean rollout, i.e. picking the model's preferred ordering
  measurably concentrates likelihood. Best-of-10 struct NLL mean 971 / median 659
  (length-dependent). Per-target table: [`data/full_per_target.csv`](data/full_per_target.csv);
  plot: [`plots/full_nll_vs_L.png`](plots/full_nll_vs_L.png).

The regenerated documents (selected + all 10 per target) are on the GCS working
copy; `publish_to_hf.py` uploads them to the public bucket.

### vLLM structured-output backend (GPU, portable to TPU)

The `gen_constrained_worker_vllm.py` + `only_correct_backend.py` path was verified
against the HF worker on a 6-target × 10 sample: **10/10 correct on all 6**, per-target
best struct-NLL matching the HF worker within sampling noise (e.g. 859.5 vs 856.1,
1741 vs 1734), at **~790 tok/s (~2× the HF worker)** on the same A5000. This is
the code that ports to iris/TPU (see backend note above).

### Full round-0 scale-out — the deliverable (941,004 documents)

The entire round-0 contacts-v1 train set was regenerated on iris preemptible TPUs
(`select_round0_all.py`: all 941,028 proteins, ground truth parsed from the original
documents), N=10 rollouts each, selecting the lowest unmodified-NLL document per
protein.

- **Deliverable: 941,004 regenerated documents, verified 0 incorrect.** Every
  contact-bearing document has `precision = recall = 1.0`; contact-bearing targets
  averaged **9.99 / 10** correct rollouts. The 1,783 zero-contact proteins are kept
  (an empty statement list is trivially correct; they score as degenerate
  `prec=0/rec=nan` and are exempted by the publish filter). **24 of 941,028** targets
  are excluded: ~23 whose resampled prefix already fills the 8192 context (no room
  for the structure section) and **1** pathological large protein
  (`AF-A0A7R8UDF5-F1`, `n_gt=338`) whose every rollout desynced.
- **Selection is meaningful at scale:** the selected document sits on average
  **39.7 nats** below the per-target mean rollout NLL.
- **Scale of the set:** protein length mean 256 / max 1994; contacts `n_gt` mean 201
  / median 131 / max 1737; **~5.64 B tokens** generated; selected `struct_nll` mean
  1460 / median 829.
- **Compute (Jul 2–13):** ~592k documents generated on **v5p-8** and ~349k on
  **v6e-4**. (The `tpu_type` column tags the v5p-8 docs `A5000` — a cosmetic
  default-label artifact, not the real device.) The fleet grew from 8× v5p-8, to a
  mixed 16-worker run (4 v5p-8 + 12 v6e-4) once v6e capacity was validated, to a
  24× v6e-4 "tail" sprint that cleared the last single-worker straggler slice.
  ≈1,290 `tp=4` generation-compute-hours (excludes XLA warmup, preemption re-warms,
  and the `prompt_logprobs` scoring pass).
- **Two scale-only bugs, fixed on this branch:** a context overflow that appears
  only at full scale — a tight-fit rollout with `prompt + generated == 8192` tripped
  `prompt (8192) + output (≥1) > max_model_len` in **both** the generate and the
  `prompt_logprobs` scoring pass — fixed by reserving one token of headroom and
  skipping targets that cannot fit (resume-safe skip marker); and a rare grammar
  desync at high concurrency, absorbed by the *select-among-100%-correct* rule plus a
  one-shot re-run of the 745 affected targets (744 recovered, 1 excluded).

**Published** to the public `open-athena/MarinFold` bucket at
`data/contacts-v1-train-only-correct-exp100/`:
[`regenerated_documents.parquet`](https://huggingface.co/datasets/open-athena/MarinFold) (3.6 GB — the training set),
`rollout_nll_all.parquet` (all N rollout NLLs), `per_target_nll.parquet` (per-protein
NLL spread). Auth-free for Colab.

## Conclusion

The only-correct constrained-decoding method works, and the deliverable is done:
**941,004 regenerated round-0 contacts-v1 training documents, verified 100%-correct,
published to the public HuggingFace bucket** — exp100's success criterion. For every
training protein (bar 25 edge cases) we generate valid, 100%-correct, full-recall
documents whose contact ordering is sampled from the base model and selected by
unmodified likelihood. The method was validated end-to-end on 1000 proteins on both a
local A5000 (HF worker) and iris TPU (vLLM structured-output backend), then scaled to
the full set on a mixed v5p-8 + v6e-4 preemptible-TPU fleet. Remaining (a later
experiment): the fine-tuning-on-regenerated-docs vs. re-epoching comparison this set
was built to enable.
