---
marinfold_experiment:
  issue: 67
  title: "exp: train a 1.5B model on contacts-v1 dataset"
  kind: models
  branch: exp/67-contacts-v1-1_5b
---

# exp: train a 1.5B model on contacts-v1 dataset

**Issue:** [#67](https://github.com/Open-Athena/MarinFold/issues/67) · **Kind:** `models` · **Branch:** `exp/67-contacts-v1-1_5b`

## Question

The goal is to get a quick sense of what accuracy is possible by training on the contacts-v1 dataset. 

The plan is that @eric-czech will do another experiment where he uses more carefully tuned settings. The goal for this current experiment is just to do something simple so it can train while we have a team offsite (as well as get an example of training into this repo).

## Hypothesis

Hopefully we will be able to train a better model at contact prediction than our previous 1.5B model trained on contacts-and-distances-v1

## Background

We have a new dataset here: https://huggingface.co/buckets/open-athena/MarinFold/tree/data/document_structures/contacts_v1

More details about it are [here](https://github.com/Open-Athena/MarinFold/tree/main/marinfold/marinfold/document_structures/contacts_v1).

The main difference from our previous contacts-and-distances-v1 is we have gotten rid of distance statements (our evals will be entirely on contact recapitulation). Also note it is much smaller (4B tokens).

See #61 

See our 1.5B training run here: https://github.com/marin-community/marin/tree/protein-training-1b

## Approach

Mirror the published 1.5B recipe from `marin/protein-training-1b` (ported into
`exp0_models_protein_docs_initial_port/`), retargeted at the new contacts-v1
corpus. Code lives in this experiment dir (not a branch of the marin repo).

**Model** — Llama 1.5B, identical shape to exp0's `protein_llama_1_5b`:
`max_seq_len=8192, hidden_dim=2048, intermediate_dim=8192, num_heads=32,
num_kv_heads=8, num_layers=24` (~1.47B params).

**Recipe** — v5p-8 @ `us-east5-a`, LR `3.5e-4`, batch `128`, seq `8192`,
`weight_decay=0.01`, `warmup=0.1`, `pack=True`,
`block_cross_document_attention=True`, bf16 compute / f32 params.

**Two deliberate departures from exp0 for contacts-v1:**
- **Unmasked loss** — next-token loss over the whole document. contacts-v1 has
  no `<distance>` statements, so there is no distance-bin loss mask (we mirror
  `exp0/train_protein_1b_unmasked.py`, not the distance-masked builder).
- **Shuffled training** — `LmDataConfig.shuffle=True` (full Feistel permutation,
  fixed `data_seed=0`). The corpus shards are physically round-descending
  (highest-pLDDT last), so an unshuffled stream is badly biased.

**Length** — 1 epoch ≈ 4,490 steps (train ≈ 4.7B tok / (128×8192 = 1.05M
tok/step)). `num_train_steps=12_000` ≈ **2.7 epochs**. `steps_per_eval=250`,
`steps_per_export=2000`.

**Files**
- `contacts_v1_train_common.py` — shared tokenizer pin, corpus paths, tokenize
  steps, resources, and the `build_train_step` recipe.
- `train_protein_1_5b_contacts_v1.py` — the 1.5B training step (entry point).
- `export_protein_1_5b_contacts_v1.py` — CPU HF-export of a checkpoint.

**Output location** — all executor artifacts (token caches, checkpoints, HF
exports) are pinned under one prefix via `MARIN_PREFIX` (force-set in
`contacts_v1_train_common.py`), per AGENTS.md — never the top-level
`gs://marin-us-east5/{tokenized,checkpoints}/…`:
`gs://marin-us-east5/protein-structure/MarinFold/exp67_contacts_v1_1_5b/…`.

**Launch** (≤10 test runs allowed per the issue) — from the experiment dir, on
the marin cluster:

```
cd experiments/exp67_models_contacts_v1_1_5b
uv run iris --cluster marin job run --no-wait --enable-extra-resources \
    --memory=16GB --disk=16GB --cpu=1 --extra=tpu --zone=us-east5-a \
    -- python -m train_protein_1_5b_contacts_v1
```

The CPU driver runs `executor_main`, which provisions the v5p-8 itself and
launches the tokenize sub-jobs (built with `--extra cpu`) then the training pod
(built with `--extra tpu`). Monitor with
`uv run iris --cluster marin job logs <JOB_ID> --since-seconds 600`.

## Implementation notes & decisions

Three things in the original plan ([HANDOFF.md](HANDOFF.md)) turned out not to
match reality; the resolutions are baked into the code:

1. **Tokenizer lives at `timodonnell/contacts-v1-tokenizer`, not
   `open-athena/`.** The plan / `contacts_v1/cli.py --push` example point at
   `open-athena/contacts-v1-tokenizer`, but that repo was never created — the
   workstation HF token lacks open-athena org-create perms. exp53 published the
   canonical, levanter-loadable copy under `timodonnell/` (2845 vocab, all
   tokenizer files). It is referenced **without** a `@<sha>` revision suffix:
   levanter's training loader accepts `repo@rev`, but the marin *tokenize* step
   loads via `huggingface_hub`, which rejects the suffix (`HFValidationError`).
   exp0 could pin only because its tokenize step was cache-skipped.

2. **Eval runs over the full val split, not a `max_eval_batches` head.** The
   plan assumed `max_eval_batches` would yield a *shuffled* ~5K-doc eval. It
   does not: levanter's `shuffle` config applies only to the **training**
   stream — validation is read sequentially. The published `val` shards are
   round-segregated (shard 0 = all round-4 / lowest-pLDDT … shard 21 = round-0),
   so a head would evaluate only the lowest-pLDDT structures. The val split is
   small (~42K docs ≈ ~45 eval batches), so we evaluate the **entire** held-out
   val split each eval (`max_eval_batches=None`) — unbiased, real cluster-level
   holdout, ~6% compute overhead. (Decision confirmed with Tim.)

3. **Tokenize reads GCS, not the HF bucket.** The corpus is *published* to the
   HF bucket `hf://buckets/open-athena/MarinFold/.../contacts_v1/`, but HF
   buckets are **not `HfFileSystem`-addressable**: levanter's fsspec on the
   worker resolves `open-athena/MarinFold` as a dataset/model repo and 404s
   (`repository not found`), so the tokenize step never reads a byte. We point
   the tokenizer at the byte-identical, region-local (us-east5) GCS working
   copy instead — `gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/documents/<split>/*.parquet`
   — with an explicit `*.parquet` glob so neither marin nor levanter falls back
   to the default `**/*.json.gz` pattern. (exp0 didn't hit this because it read
   an `hf://datasets/` *repo*, which the HF datasets loader handles.)

4. **All executor output pinned under `protein-structure/`** via `MARIN_PREFIX`
   (force-set in `contacts_v1_train_common.py`): token caches, checkpoints, and
   HF exports land under
   `gs://marin-us-east5/protein-structure/MarinFold/exp67_contacts_v1_1_5b/…`,
   never the top-level `gs://marin-us-east5/{tokenized,checkpoints}/…` (which
   belong to the marin protein-experiments convention, per AGENTS.md).

5. **Two dependency-drift fixes** (current floating `marin-latest` is
   `0.99.dev20260529`, newer than when exp0 was written):
   - `models/marinfold_models/defaults.py` imported `versioned` /
     `ensure_versioned` / `this_output_path` / `unwrap_versioned_value` from the
     `marin.execution.executor` submodule; current marin only re-exports them
     from the `marin.execution` package root. Repointed the import (fixes
     training for every experiment, not just this one).
   - `dupekit` (imported unconditionally by `marin.datakit.normalize`) is no
     longer declared as a marin dependency, so `uv sync` skipped it. Added it
     explicitly to this experiment's `dependencies` (it's on the marin
     find-links mirror).

6. **WANDB_API_KEY forwarded into the training pod.** marin's training entry
   raises unless the key is in the pod's `env_vars`/`os.environ`, and the pod
   does not inherit the launching shell's env. `build_train_step` reads the key
   from the driver's environment and threads it into `env_vars` (never
   hard-coded). Launch with `iris ... -e WANDB_API_KEY <key>`.

## marin-latest tokenize↔train cache mismatch — worked around (shim)

As of 2026-06-10 the run got **all the way through tokenization and into the
trainer**, then the training pod died reading the token cache:

```
ValueError: Sharded cache ledger missing input_ids/0 count for shard part-00000-of-00133
```

**Root cause (already fixed upstream — do NOT file a new issue):** marin's
tokenize computes the consolidation exemplar with `_exemplar_for`
(`marin/processing/tokenize/tokenize.py:256` in `0.99.dev20260529`), which
produces a field layout that disagrees with what marin-levanter's reader
reconstructs from `processor.output_exemplar`. On disk everything is the flat
`input_ids` (data dir `part-*/input_ids`, ledger `field_counts[input_ids]`), but
the reader walks the exemplar tree and addresses the field as the leaf-path
**`input_ids/0`** (`levanter/store/cache.py:558`,
`jtu.tree_map_with_path(field_store, self._cache._exemplar)`), so
`_build_flat_field_offsets_async` raises. This is marin issue
[#6008](https://github.com/marin-community/marin/issues/6008) ("Remove
`_exemplar_for` from tokenize"), **CLOSED, fixed by
[#6014](https://github.com/marin-community/marin/pull/6014) (merged
2026-06-02)** — which drops `_exemplar_for` and derives the exemplar from the
shards so writer and reader agree.

**Why we still hit it:** every marin `*-latest` wheel on the find-links mirror
is frozen at **2026-05-29** (verified across all release tags), i.e. ~4 days
*before* #6014 merged. exp0 never hit the bug because it always cache-skips its
tokenize (`override_output_path`); exp67 is the first *fresh* tokenize→train on
this snapshot.

**The on-disk caches are actually correct and flat** — I narrowed the bug to
`levanter.data.text._batch_tokenizer.BatchTokenizer.output_exemplar`, which
returns `{"input_ids": <python list>}`. The cache *writer* flattens that with
`is_leaf=heuristic_is_leaf` (a list of ints is a leaf → one flat `input_ids`
field, matching disk), but the *reader's* `jagged_array_tree` walks the same
exemplar **without** `is_leaf`, so the list becomes the tree node `input_ids/0`.
So no re-tokenize is needed — only the reader's in-memory exemplar is wrong.

**Workaround in place (`sitecustomize.py`):** a startup shim makes
`output_exemplar` return numpy arrays (true leaves), so the reader's field path
collapses back to the flat `input_ids` the cache uses. Verified locally by
reading the real val cache (41,954 rows) end-to-end. It's injected into the
training pod via `PYTHONPATH=/app` (set in `build_train_step` `env_vars`) — the
pod runs marin's `run_levanter_train_lm`, not our code, so an interpreter-startup
hook is the only injection point. **Remove `sitecustomize.py` and the
`PYTHONPATH` env once the experiment is on a marin build that includes #6014.**
Tokenization is durable (both caches persist under
`…/exp67_contacts_v1_1_5b/tokenized/…` and are reused on restart).

## Success criteria

We have a model training run launched and training. **Status: ✅ MET.** The run
is live on a v5p-8 @ us-east5-a and stepping with a healthy, decreasing loss
(8.29 → 7.91 → 7.43 → 6.86 over the first steps; init ≈ ln(2845) = 7.95).

## Results

- **W&B run:** https://wandb.ai/open-athena/MarinFold/runs/protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked-3b5cf2
  (run name `protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked`)
- **Iris job:** `/bizon/iris-run-job-20260610-124627` (launched 2026-06-10)
- **Outputs:** `gs://marin-us-east5/protein-structure/MarinFold/exp67_contacts_v1_1_5b/`
  — token caches under `tokenized/`, checkpoints under `checkpoints/…` every
  2000 steps.
- Token caches (built once, reused): train `tokenized/contacts-v1-663ba6`
  (~4.7B tok), val `tokenized/contacts-v1-val-92827b` (41,954 docs).

_(Fill in after the run completes — final/val loss curve, contact-recapitulation
numbers vs the prior contacts-and-distances-v1 1.5B.)_

## Conclusion

_(Fill in after results are in. Reminder: the `sitecustomize.py` shim + the
`PYTHONPATH=/app` env are temporary workarounds for marin #6008/#6014 — remove
them once the experiment is on a marin build that includes the fix.)_
