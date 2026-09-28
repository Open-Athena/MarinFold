---
marinfold_experiment:
  issue: 343
  title: 'exp: train 1.5B on the exp277 corpus plus the predicted-complex corpus'
  kind: models
  branch: main
---

# exp: train 1.5B on the exp277 corpus plus the predicted-complex corpus

**Issue:** [#343](https://github.com/Open-Athena/MarinFold/issues/343) · **Kind:** `models` · **Branch:** `main`

## Question

Does adding the [#294](https://github.com/Open-Athena/MarinFold/issues/294) predicted-complex corpus to the exp277 native + ProteinMPNN training mixture change contacts-v1 monomer contact accuracy, and does the resulting model actually learn multi-chain complexes?

## Hypothesis

The complex corpus is 11.15 B tokens against exp277's 248.58 B — **4.3% of the mixture**. So:

- **Monomer accuracy is a tie.** eval-val R-precision lands within the predeclared 0.005 threshold of exp277's 0.55375. A 4.3% mixture change, spent on a document type the eval does not contain, should not move a monomer benchmark either way. A loss of more than 0.005 would be the interesting result, not the expected one.
- **Complex modelling is not a tie.** Held-out complex LM loss drops far below what exp277's checkpoint achieves on the same documents. exp277 has never seen a multi-chain document, so it must pay for the shared 2000-index ring, the interleaved chain runs, and inter-chain contacts at any index distance. This is the measurement the experiment exists for.
- Designed-protein (eval-denovo) accuracy is the one monomer number with a plausible mechanism for moving: interface contacts are, geometrically, contacts between two pieces of chain that sequence separation does not explain, which is also what designed folds stress. No direction is predeclared.

## Background

- [#277](https://github.com/Open-Athena/MarinFold/issues/277) trained the current default, `contacts-v1-exp277-m2-p06-full-epoch-1.5B` — one complete epoch over 232,090,905 native + MPNN-redesigned documents / 248,583,762,834 raw tokens / 34,092,146 packed examples, 266,345 steps, from scratch on exp232's m2-p06 settings. R-precision 0.62009 legacy 554 / 0.55375 eval-val / 0.69582 eval-denovo.
- [#294](https://github.com/Open-Athena/MarinFold/issues/294) built and published the predicted-complex corpus: 3,410,738 documents / 11,151,472,024 tokens of protein–protein complexes, 261,833,754 interface contacts, 611,656 heterodimers + 2,799,082 homodimers, decontaminated against all 577 eval2 units on every subunit. Two arms: `afcdb` (2,982,517 predicted dimers) and `pinder` (428,221 experimental heterodimers).
- The multi-chain document format is [#222](https://github.com/Open-Athena/MarinFold/issues/222)'s and adds **no new tokens** — verified here, not assumed (see Notes).
- No MarinFold model has ever been trained on a multi-chain document. This is the first.

## Approach

**Match exp277 in every training setting.** Scratch init, Qwen3 1.5B (`MODEL_CONFIG` from exp232's `training_contract.py`), seq len 8192, global batch 128, Adam LR 1e-3 / weight decay 0.2, WSD linear warmup 10% / decay 20% / min-LR ratio 0.1, packed documents with blocked cross-document attention, scheduled amino-acid order augmentation, `MODEL_SEED=0`, `DATA_SEED=0`, `BlockShuffleConfig(256, 512, feistel)`. 16 nodes x 8 H100 on `cw-us-east-02a` at **batch priority**, which is where the four exp277 caches already live.

**Corpus = exp277's four caches, unchanged, plus the complex corpus.** One epoch, every packed example visited exactly once, one fresh global shuffle over the concatenation, no mixture weighting and no source cycling — exp277's `epoch_data.OneEpochDataConfig`, reused. The step count is measured with the trainer's own packer before launch and pinned, as exp277 did; it will be about 280,000.

**The complex corpus is taken at its `manifest_natural` weighting**, i.e. every document once. `manifest_balanced` exists because PINDER averages 18.7 structures per interface cluster, but applying it would mean weighted sampling, which is exactly the setting exp277 does not use. Declining it is the matched-settings choice, not a claim that it is worse.

**One shard is held out.** `shard-00170-of-00171` (10,738 documents, 0.31% of the corpus) becomes a second validation cache. Without it this experiment has no measurement of the thing it is testing — the monomer eval sets contain no complexes, so a complex-blind run and a complex-fluent one would be indistinguishable. Training uses shards 00000–00169.

Stages, one `launch.py` phase each:

- `stage` — mirror the 171 published shards HF bucket -> `s3://marin-us-east-02a/MarinFold/exp<N>.../documents/`, from an in-region CoreWeave CPU pod. The workstation uplink is ~2 MB/s, so 19.6 GB must not pass through it.
- `prepare` — tokenize the staged shards with `eczech/contacts-v1-tokenizer-5d68a24a899f` under exp277's per-record validation, and independently audit a smoke cache against fresh tokenization. Verify the four exp277 caches by ledger and pinned counts; build nothing that already exists.
- `audit` — count packed examples per corpus with `GreedyPrepackedDataset`, pin `EPOCH_PACKED_EXAMPLES` and the step count.
- `train-smoke` — ten optimizer updates on the full production caches, separate run identity.
- `train` — production.

**Evaluation.** Two halves, and the second is not optional:

1. *Monomers, for comparability.* The fixed exp82 rollout-and-resample recipe at exp277's settings, on legacy 554 + eval-val + eval-denovo, paired per-protein against exp277 with 10,000-resample protein bootstrap intervals. **eval-test stays unread.**
2. *Complexes, for the actual question.* LM loss on the held-out complex shard for this checkpoint and, as the control, for the exp277 default — same documents, same tokenizer, same packing. Reported per arm (`afcdb` / `pinder`) and split by whether a contact crosses a chain boundary, because a model can score well on a complex document by predicting its intra-chain contacts alone.

## Success criteria

- A healthy production run to the pinned step count, with a permanent native checkpoint and an HF export carrying its tokenizer.
- **Monomer:** eval-val R-precision within 0.005 of exp277 (a tie, the predeclared expectation), with paired bootstrap intervals on all three sets. A regression beyond 0.005 is a reportable cost of the mixture change, not a failure of the run.
- **Complex:** held-out complex LM loss materially below exp277's on the same documents, with the inter-chain split reported. This is what decides whether the corpus taught anything.
- Full identity trail: iris job ids, W&B run, commit SHA, checkpoint paths, `history/runs/` entry, per-input eval timings.

## Results

### Corpus preparation

**The complex corpus fits the model's vocabulary.** The corpus publishes a
2,848-token tokenizer beside the data; the 1.5B model has 2,845 embedding rows.
The three extra ids are `<contacts-v1.sequence_only>` (2845), `<retract>` (2846)
and `<contacts-v1.backtracking>` (2847) — appended by other document structures
and unused here. Below 2845 the two vocabularies are *identical*, so tokenizing
with the model's own tokenizer is not an approximation. On shard 00000, all
20,000 documents re-encode to exactly their recorded `num_tokens`, reach token id
2142 at most, and satisfy the contacts-v1 boundary contract. Recorded in
[`data/tokenizer_contract.csv`](data/tokenizer_contract.csv); reproduce with
`verify_tokenizer_contract.py --shard <shard>`. That sample is evidence, not the
guarantee: `prepare.py` applies the same per-record check to every document of
every shard while the caches are built, so a document reaching an out-of-contract
id fails the build rather than indexing past the embedding table.

**Truncation is statement-granular, which the issue's note had slightly wrong.**
The published corpus flags 2.4% of documents `truncated`, and those land at
8190–8192 tokens rather than exactly on the cap — #294 drops whole three-token
contact statements. Only the documents landing exactly on 8192 lose their
appended `<eos>` to the packer's clip: 178 of 20,000 on shard 00000, about 0.9%
of the corpus rather than the 2.4% implied by the `truncated` flag. No document
exceeds the cap.

**Staging: 19.6 GB in about three minutes.** `/bizon/exp343-stage-a01` mirrored
all 171 published shards from the public HF bucket into
`s3://marin-us-east-02a/MarinFold/exp343_models_complex_corpus_training/documents/`
on one in-region CoreWeave CPU pod, splitting the held-out shard out on the way
in. The staged footers reconcile exactly with the published corpus:
**3,410,738 documents = 3,400,000 train across 170 shards + 10,738 validation in
shard 00170**, matching the counts pinned in `config.py` before the job ran. Per
shard size, row count and SHA256 are in `documents/stage-manifest.json`.

All four exp277 token caches were confirmed present and complete in
`marin-us-east-02a` before any of this started, so nothing is re-tokenized: the
native AFDB/ESM caches from exp232 (`2026.08.14`) and the MPNN AFDB/ESM caches
from exp277 (`2026.09.09.1`).

## Conclusion

_(Fill in after results are in.)_
