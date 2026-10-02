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
The training environment is pinned in [`uv.lock`](uv.lock), with causal-conv1d
1.6.0 built against the pinned torch/CUDA environment in
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
GPU correctness testing starts with
`/timodonnell/exp347-qwen35-0p8b-contacts_v1-smoke-a01`.
Seven document-integrity tests, eight capacity-helper tests, and static checking
of the CPU preparation/launch code passed. No training outcome is claimed yet.

## Conclusion

In progress. This experiment has not yet established a contact-prediction improvement.
