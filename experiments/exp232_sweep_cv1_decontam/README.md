---
marinfold_experiment:
  issue: 232
  title: "exp: retrain our best model on decontaminated training data"
  kind: models
  branch: exp/232-sweep-cv1-decontam
---

# exp: retrain our best model on decontaminated training data

**Issue:** [#232](https://github.com/Open-Athena/MarinFold/issues/232) · **Kind:** `models` · **Branch:** `exp/232-sweep-cv1-decontam`

## Question

Does the exp199 1.5B recipe retain its contact-prediction quality when trained
from scratch on AFDB and ESM Atlas corpora decontaminated from FoldBench at 30%
sequence identity?

## Approach

`exp232_tokenize.py` first mirrors the two issue #232 training corpora from the
public Hugging Face bucket to CoreWeave S3. It verifies the complete mirror before
tokenizing only the S3 copies with
`eczech/contacts-v1-tokenizer-5d68a24a899f`. The dated caches are:

```text
s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/
  data/afdb       <- data/document_structures/contacts_v1_decontam/train
  data/esm        <- data/document_structures/contacts_v1_esm_atlas_decontam/train
  tokenized/contacts_v1/afdb/2026.08.14
  tokenized/contacts_v1/esm/2026.08.14
```

`exp232_sweep.py` retains exp199's scratch-trained Qwen3 1.5B architecture,
optimizer points, two data-mixture policies, WSD schedule, token budget, full
validation cadence, and scheduled amino-acid augmentation. It removes the
unaugmented arm, leaving ten trials:

```text
m{1,2}-p{01,02,03,04,06}-aug
```

`m1` samples AFDB and ESM equally. `m2` reads the completed caches' token totals
and samples in corpus proportion. Validation remains the exp199 contacts-v1
cache at
`s3://marin-us-east-02a/MarinFold/exp154_qwen_contacts_v1/tokenized/contacts-v1-val/2026.07.25`.

Permanent checkpoints retain exp199's 14,520-step cadence plus the forced final
save. Temporary checkpoints save every 15 minutes. Production runs share one
W&B/checkpoint identity across CoreWeave placement retries; never run two writers
for one trial.

## Tokenization launch

From this directory, source `~/marin.env` without printing it. The default `all`
phase mirrors both corpora, verifies the mirror, and tokenizes AFDB followed by
ESM. Use `--phase mirror` or `--phase tokenize` only when reviewing or resuming
the boundary explicitly.

Before the full run, exercise the same HF-to-S3 and Marin tokenization paths on
the smallest parquet shard from each corpus. `--smoke-test` writes only below
`tmp/tokenization-smoke/2026.08.14.1`, derives the expected document counts from
the mirrored parquet footers, and requires each output cache to contain exactly
that many documents. The fixed prefix makes this command safely resumable.

```bash
uv run iris --cluster marin job run \
  --target-cluster cw-rno2a \
  --priority batch \
  --user eczech \
  --job-name exp232-tokenize-smoke \
  --enable-extra-resources \
  --cpu 4 --memory 16GB --disk 32GB \
  -e MARIN_PREFIX s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam \
  -e HF_TOKEN "$HF_TOKEN" \
  -- python exp232_tokenize.py --smoke-test
```

After that succeeds, launch the complete mirror and tokenization:

```bash
set -a
source ~/marin.env
set +a

uv run iris --cluster marin job run \
  --target-cluster cw-rno2a \
  --priority batch \
  --user eczech \
  --job-name exp232-tokenize \
  --enable-extra-resources \
  --cpu 4 --memory 16GB --disk 32GB \
  -e MARIN_PREFIX s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam \
  -e HF_TOKEN "$HF_TOKEN" \
  -- python exp232_tokenize.py
```

## Sweep entry point

The later sweep operator launches one selected trial at a time. This is the
single-trial shape used for smoke validation; production placement is chosen by
the sweep operator and always uses batch priority.

```bash
uv run iris --cluster marin job run \
  --target-cluster cw-us-east-02a \
  --priority batch \
  --user eczech \
  --job-name exp232-s01-m1-p06-aug-smoke \
  --cpu 2 --memory 6GB --disk 32GB --extra gpu \
  -e MARIN_PREFIX s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam \
  -e WANDB_ENTITY open-athena \
  -e WANDB_PROJECT MarinFold \
  -e TRIAL m1-p06-aug \
  -e CLUSTER cw-us-east-02a \
  -e NODES 1 \
  -e SMOKE 1 \
  -- python exp232_sweep.py --version 2026.08.14.1 --run
```

## Results

The isolated tokenization smoke test completed successfully on `cw-rno2a` on
2026-08-14 as Iris job `/eczech/exp232-tokenize-smoke-v3` (exit 0, 4m34s):

- AFDB: 971 input documents, 971 cached documents, 833,682 tokens.
- ESM Atlas: 19,624 input documents, 19,624 cached documents, 20,498,326
  tokens.

Follow-up Iris job `/eczech/exp232-token-audit` independently scanned every
smoke cache row and compared it with fresh tokenization of its mirrored source.
All 20,595 documents and 21,332,008 tokens matched exactly. It found zero
`<UNK>` (ID 2844), zero pad tokens, no out-of-range IDs, and no malformed
contacts-v1 boundaries. The production pipeline applies the ID-range, OOV,
padding, and boundary checks to every tokenized record before writing it.

Full Iris job [`/eczech/exp232-tokenize`](https://iris.oa.dev/#/job/%2Feczech%2Fexp232-tokenize)
then completed on `cw-rno2a` (exit 0, 47m15s, zero failures or preemptions).
It mirrored and verified all 2,070 AFDB objects and 3,341 ESM Atlas objects,
then wrote the immutable source manifest to
`s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/data/mirror-2026.08.14.json`.
The final production caches are:

- AFDB: 3,963,003 documents and 4,432,940,838 tokens at
  `s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/tokenized/contacts_v1/afdb/2026.08.14`.
- ESM Atlas: 65,553,178 documents and 70,042,923,165 tokens at
  `s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/tokenized/contacts_v1/esm/2026.08.14`.

Every production record passed the same ID-range, OOV, padding, and contacts-v1
boundary checks before it was written. Both consolidated cache ledgers and
statistics matched their expected document counts and positive token totals.
Sweep execution remains pending review.

## Conclusion

The issue #232 training data is mirrored and fully tokenized in the dated
production caches. The ten-trial sweep is ready for its separate smoke and
execution review.
