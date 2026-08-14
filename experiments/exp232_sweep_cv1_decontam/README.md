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
`tmp/tokenization-smoke/2026.08.14`, derives the expected document counts from
the mirrored parquet footers, and requires each output cache to contain exactly
that many documents. The fixed prefix makes this command safely resumable.

```bash
uv run iris --cluster marin job run \
  --target-cluster cw-rno2a \
  --priority batch \
  --user eczech \
  --job-name exp232-tokenize-smoke \
  --cpu 4 --memory 16GB --disk 32GB --extra cpu \
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
  --cpu 4 --memory 16GB --disk 32GB --extra cpu \
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

Pending tokenization and sweep execution.

## Conclusion

Pending results.
