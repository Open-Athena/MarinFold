---
marinfold_experiment:
  issue: 126
  title: "exp: generate think-augmented contacts-v1 corpus at scale (local) + publish"
  kind: data
  branch: claude/exp123-contacts-v1-pause-tokens
---

# exp: generate think-augmented contacts-v1 corpus at scale (local) + publish

**Issue:** [#126](https://github.com/Open-Athena/MarinFold/issues/126) · **Kind:** `data` · **Branch:** `claude/exp123-contacts-v1-pause-tokens`

## Question

Generate the **think-augmented** contacts-v1 training corpus at scale from afdb-24M — a `<think>`-pause-token twin of the exp53 [`contacts-v1`](https://huggingface.co/buckets/open-athena/MarinFold) corpus over the **same** entries / rounds / splits — and publish it to the `open-athena/MarinFold` HF bucket, so it can be used as a drop-in dataset for #124.

## Hypothesis

Pause tokens ([Goyal et al. 2023](https://arxiv.org/abs/2310.02226)) let a model
spend extra compute before committing to a prediction. To test that on contacts-v1
(the goal of #124 / the inference-time eval that follows), we first need a
training corpus that contains `<think>` tokens. This experiment produces it — a
think twin of the exp53 corpus so #124 can train apples-to-apples.

## Background

- Generator: the `GenerationConfig(think=True)` path added in **#123** (PR #125) — reuses the reserved `<think>` token, no tokenizer change, `think=False` byte-identical to the pre-think generator.
- Selection precedent: **#53** (exp53) generated the non-think contacts-v1 corpus (4,213,203 docs / 960,054 clusters, up to 5 pLDDT rounds, round-descending). We reuse its exact selection manifest on GCS (`gs://marin-us-east5/.../exp53_contacts_v1_5x/selection_manifest/{train,val,test}/`) so this corpus lines up 1:1 with the non-think one.
- Downstream: **#124** (train a 1.5B contacts-v1 model on the pause-token dataset).

## Approach

Reuse the exp53 Stage-B per-row worker (`generate_rows.generate_doc_for_row` → `generate_document`) with `think=True`. **Run locally** (48-core workstation; AFDB is public) rather than on iris, shard-parallel over the exp53 manifest, one output shard per manifest shard so the train/val/test + round-descending layout is preserved. Each output row is contacts-v1's `metadata_row()` (now including `think_tokens`) + the manifest provenance columns — schema-identical to exp53's corpus plus the one new column.

## Running it

```bash
uv sync --extra test

# 1. Pull exp53's selection manifest locally (train/val/test, ~2111 small shards).
gcloud storage rsync -r \
  gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/selection_manifest \
  ~/exp126_scratch/manifest

# 2. Generate locally (48 procs; AFDB fetched from the public bucket, structures
#    streamed not stored). Resumable — re-run to fill any missing output shards.
export GOOGLE_CLOUD_PROJECT=hai-gcp-models        # gcsfs quota/billing project
uv run python generate_local.py --split val   --manifest ~/exp126_scratch/manifest --out ~/exp126_scratch/documents
uv run python generate_local.py --split test  --manifest ~/exp126_scratch/manifest --out ~/exp126_scratch/documents
uv run python generate_local.py --split train --manifest ~/exp126_scratch/manifest --out ~/exp126_scratch/documents

# 3. Publish to the HF bucket (shards + README + tokenizer).
uv run python publish_to_hf.py --documents ~/exp126_scratch/documents
```

## Success criteria

- All 4.21M entries generated with `think=True`, per-split/round counts matching the exp53 manifest.
- Every document ≤ 8192 tokens, ends with `<end>`, `<think>` only between `<contact>` statements.
- Published to `open-athena/MarinFold` → `data/document_structures/contacts_v1_think/{train,val,test}/` with a dataset README + co-located tokenizer.

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
