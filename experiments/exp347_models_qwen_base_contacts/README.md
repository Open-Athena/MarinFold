---
marinfold_experiment:
  issue: 347
  title: 'exp: fine-tune Qwen3.5 base models for protein contacts'
  kind: models
  branch: exp/347-qwen-base-contacts
---

# exp: fine-tune Qwen3.5 base models for protein contacts

**Issue:** [#347](https://github.com/Open-Athena/MarinFold/issues/347) · **Kind:** `models` · **Branch:** `exp/347-qwen-base-contacts`

## Question

Can pretrained Qwen3.5 base language models learn protein contacts more effectively than our models trained from scratch? Does a natural-language instruction and ordinary protein sequence improve learning relative to directly continuing pretraining on contacts-v1 documents?

## Hypothesis

General pretraining may supply useful sequence, indexing, and output-format priors. A plain-language contact-prediction prompt may make those priors easier to use. The effect may vary with model size.

## Approach

- Fine-tune the pretrained **base** checkpoints Qwen/Qwen3.5-0.8B-Base, Qwen/Qwen3.5-2B-Base, and Qwen/Qwen3.5-4B-Base. Pin revisions and retain their pretrained tokenizers.
- Compare two document formats at each size: existing contacts-v1 documents; and a simple natural-language format description followed by the protein sequence and the corresponding contacts.
- Use matched examples from the existing exp225/exp232 decontaminated corpus and document all transformations, tokenization expansion, masking, and truncation decisions. Use text-only model components; train the language model weights, not a randomly initialized replacement.
- User-approved first-pass budget: approximately **1B training tokens per trial**, six trials, at most **48 H100 GPUs** concurrently. Begin with correctness and GPU smoke tests, then batch-priority CoreWeave training. No bulk cross-region data transfers.
- Log to open-athena/MarinFold, record run history, retain resumable checkpoints with tokenizer and pinned configuration, and compare validation contact likelihood, output validity, and contact-prediction metrics. Use eval-val for development; do not tune on eval-test.

## Success criteria

All six training runs complete their declared budget and have recoverable model/tokenizer artifacts. Report data and compute exposure, learning curves, and matched contact-prediction validation results, with honest limits on attribution to pretraining and document format. This initial experiment has no matched random-initialization control, so comparisons against existing models are contextual rather than a causal pretraining ablation.

## Implementation

[`common.py`](common.py) pins the three Hugging Face revisions and defines the
plain-language prompt. [`prepare.py`](prepare.py) selects 512 source shards by a
stable SHA256 ordering from the existing exp232 AFDB mirror. Both formats retain
the complete sequence and complete original contact list. A row is excluded from
both formats when either exceeds 16,384 native Qwen tokens or when it has no
contacts. No document is cropped, split into context-free fragments, or packed
with another protein. The prompted arm decodes cyclic positions to the ordinary
sequence and numbers contact endpoints from 1. Its contact order and orientation
match the original document. Sequence-cluster hashes reserve 1% for validation.
Malformed input aborts preparation.

Both arms optimize all next-token targets, including their prefixes. Equal token
budgets therefore compare the complete formatting treatments: they do not isolate
natural language from position numbering, tokenization efficiency, or protein
exposure. The prompted arm sees more proteins per billion tokens. Training reports
document and contact exposure along with token counts. Token NLL across different
formats is not directly comparable; downstream contact metrics remain necessary.

[`train.py`](train.py) loads the pretrained text backbone with no missing weights,
keeps Qwen's tokenizer, and updates all language-model weights. The vision tower
and unused MTP head are omitted. Eight H100s per trial use DDP, FP32 parameters and
Adam states, BF16 autocast, gradient checkpointing, sharded optimizer states, and
Liger's functional fused vocabulary loss. The DeltaNet fast kernels are required;
there is no silent slow fallback or runtime monkey patch. Learning rate is 2e-5,
with 1% token warmup and cosine decay to 10%; AdamW uses betas (0.9, 0.95), weight
decay 0.1, and gradient clipping at 1.0. Four complete documents per GPU form each
accumulated optimizer step, with a global token-normalized objective. The final
step can exceed 1B by at most one global batch.

Every 15 minutes and at completion, each rank saves its optimizer/RNG/data cursor
and rank zero saves the model/tokenizer. A completion marker precedes movement of
the resume pointer; only then can the previous resumable checkpoint be removed.
The final export also includes BF16 inference weights and tokenizer. Resume keeps
the same run identity and currently requires eight ranks. Working artifacts are
under `s3://marin-us-east-02a/MarinFold/exp347_qwen_base_contacts/`; durable public
publication and contact-generation evaluation follow completed training.

[`launch.py`](launch.py) submits one explicitly chosen trial through the Marin
controller at batch priority, pins the PyTorch container digest, forwards W&B
credentials without recording them, and records the dispatch in sweep SQLite.
The training environment is pinned in [`uv.lock`](uv.lock), with the SHA256-pinned upstream causal-conv1d
1.7.0 CUDA 12 / torch 2.10 wheel in
[`gpu_bootstrap.sh`](gpu_bootstrap.sh). Initial Iris submission used marin commit
`16ed2b63cdf810fc444930138ac3a35f53260e71`.

## Results

The one-shard data smoke succeeded on 2026-10-02:
`/timodonnell/exp347-data-smoke-a01`. Of 1,887 source rows, 98 exceeded 16K and one
had no contacts. The retained 1,766 training / 22 validation rows preserve all
contact information. Training text contains 11,302,406 contacts-v1 tokens versus
3,244,662 prompted tokens (3.48x expansion for contacts-v1). These numbers describe
the smoke shard, not the complete corpus.

Full preprocessing completed as `/timodonnell/exp347-prepare-a01`; the
per-shard source hashes and counts are committed in [`data/manifest.json`](data/manifest.json).
From 981,860 source rows, it excluded 50,220 over-length rows (5.11%) and 1,794
empty-contact rows. The shared pool has **920,611 training** and **9,235 validation**
documents, containing 151,261,429 training contacts. The native-token totals are
**4,493,121,575 contacts-v1** and **1,289,984,389 prompted**. Each 1B-token arm can
finish without a second corpus pass, although protein exposure differs by format.
The pinned 0.8B and 2B checkpoints were staged successfully by
`/timodonnell/exp347-stage-small-a02` and their tokenizers are exactly equal.
The first GPU attempt was preempted during dependency compilation. The pinned
upstream CUDA wheel removed that build; attempt
`/timodonnell/exp347-qwen35-0p8b-contacts_v1-smoke-a02` then completed successfully.
It processed 137,698 tokens in four optimizer steps, saved all eight optimizer
shards and model/tokenizer artifacts, and reached 45,007 tokens/s in the last step
with 12.53 GB peak allocated GPU memory. The [smoke W&B run](https://wandb.ai/open-athena/MarinFold/runs/exp347-qwen35-0p8b-contacts_v1-smoke)
has initial/final contact-continuation validation metrics. These tiny smoke values
are engineering checks, not contact-accuracy results. Attempt a03 restored that state, exactly reproduced its validation NLL,
performed step 5 (174,050 cumulative smoke tokens), and saved another checkpoint.

The 2B prompted smoke also completed as
`/timodonnell/exp347-qwen35-2b-prompted-smoke-a01`
([W&B](https://wandb.ai/open-athena/MarinFold/runs/exp347-qwen35-2b-prompted-smoke)).
It processed 109,508 tokens in ten steps, used 29.09 GB peak allocated memory,
and saved its model, tokenizer, and eight optimizer shards at step 10.
The 2B contacts-v1 smoke completed as
`/timodonnell/exp347-qwen35-2b-contacts_v1-smoke-a01`
([W&B](https://wandb.ai/open-athena/MarinFold/runs/exp347-qwen35-2b-contacts_v1-smoke)).
Its step-4 checkpoint contains 137,698 training tokens; peak allocated memory
was 28.74 GB. Both 2B checkpoint directories and all rank shards were verified.

Eleven document/cursor/output-parser tests, eight capacity-helper tests, Ruff,
and Pyrefly passed. A two-rank CPU numerical check also confirmed that the
accumulated gradients match a single globally token-weighted reference loss.
Measured per-input validation times are retained in [`data/timings.csv`](data/timings.csv).

## Conclusion

In progress. This experiment has not yet established a contact-prediction improvement.

## Production training

The 0.8B contacts-v1 arm was launched as
`/timodonnell/exp347-qwen35-0p8b-contacts_v1-1bt-a01`
([W&B](https://wandb.ai/open-athena/MarinFold/runs/exp347-qwen35-0p8b-contacts_v1-1bt)).
The first 17 steps processed 2.63M tokens; steady early steps reached about
39–53K tokens/s over eight H100s. Validation at initialization uses 64 documents
from the held-out sequence clusters. This is a running experiment, not a final
result. At step 250 (36.92M tokens), contact-continuation validation NLL fell
from 0.70685 to 0.59605. This measures held-out likelihood, not contact accuracy.

The 0.8B prompted and both 2B arms have also been launched with the same
1B-token budget:
[0.8B prompted W&B](https://wandb.ai/open-athena/MarinFold/runs/exp347-qwen35-0p8b-prompted-1bt),
[2B contacts-v1 Iris job](https://iris.oa.dev/#/job/%2Ftimodonnell%2Fexp347-qwen35-2b-contacts_v1-1bt-a01),
[2B prompted W&B](https://wandb.ai/open-athena/MarinFold/runs/exp347-qwen35-2b-prompted-1bt).
The 4B arms await the required approval for model staging: the combined pinned
base-model download is approximately 15.6 GB, exceeding the repository's explicit
10 GB cross-region transfer threshold. Only the 0.8B and 2B weights have been staged.

![In-progress validation likelihood curves](plots/validation_nll.png)

[`collect_metrics.py`](collect_metrics.py) plots the committed
[`data/learning_curves.csv`](data/learning_curves.csv), and its optional refresh
mode retrieves the latest production histories from W&B. The two formats use
different tokenizations of the contacts, so their token NLLs are not directly
comparable. The plotted observations are intermediate training results.

As of 2026-10-02 18:50 UTC, all four production jobs remain submitted but are
waiting for Kueue admission. Three were preempted at batch priority; the first
0.8B contacts-v1 run has a resumable step-253 checkpoint, while the two newer
prompted runs had not yet reached their first periodic checkpoint. Iris will
retry the same jobs and training will load the latest available state. The
fourth production arm (2B contacts-v1) had not begun training. Aggregate fleet
GPU availability included cordoned nodes; read-only node inspection and exact
Kueue diagnostics established the placement constraint. No cluster configuration
or priority was changed. The launcher now allows a CPU reservation of 32 (four
threads per rank) for future dispatches, but this alone did not resolve the gate.

[`rollout_validation.py`](rollout_validation.py) provides a separate greedy
completion diagnostic on those held-out AFDB documents, retaining each output,
invalid/duplicate pair count, token-cap flag, precision/recall/F1, and timing. It
is explicitly not the established FoldBench rollout-plus-resample benchmark and
its numbers must not be compared to that benchmark's R-precision.
The eight-protein-per-format generation canary,
`/timodonnell/exp347-generation-smoke-a02`, is also queued. It uses the saved 2B
smoke checkpoints and a deliberately short 256-token cap; it has no accuracy
results yet. This is an additional eight-H100 job while active, for 40 requested
GPUs including the four production runs.
