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

## ⚠️ Current blocker (open): marin-latest tokenize↔train cache mismatch

As of 2026-06-10 the run gets **all the way through tokenization and into the
trainer**, then the training pod dies reading the token cache:

```
ValueError: Sharded cache ledger missing input_ids/0 count for shard part-00000-of-00133
```

Diagnosis (verified by reading the cache ledgers on GCS): the marin tokenize
step writes `shard_ledger.json` with `field_counts_by_shard[<shard>] =
{"input_ids": <n>}`, but levanter's training-time cache reader
(`levanter/store/cache.py:_build_flat_field_offsets_async`) looks up the field
as **`input_ids/0`** (a tree leaf-path) and raises when that key is absent. Both
the train and val caches use the flat `input_ids` key, and a single
`marin-levanter` version writes and reads them — so this is a writer/reader
field-naming **regression in the floating `marin-latest` (0.99.dev20260529)**,
not an experiment-config issue. exp0 never hit it because it always cache-skips
its tokenize (`override_output_path`); exp67 is the first run to do a *fresh*
tokenize→train on current marin-latest.

**Tokenization is durable** (both caches are built and reused on restart, under
`…/exp67_contacts_v1_1_5b/tokenized/…`), so the fix is purely the marin/levanter
version: pin marin to a snapshot where tokenize and train agree on the ledger
field key (e.g. whatever the carefully-tuned #61 run uses), or land an upstream
fix. Pending a version decision. (6 of the issue's ≤10 allowed test runs used.)

## Success criteria

We have a model training run launched and training. **Status: blocked at the
trainer's cache read by the marin-latest regression above — code is complete and
tokenization succeeds; the run reaches the trainer but cannot read the cache.**

## Results

_(Fill in after the run completes — W&B run URL, val-loss curve, any contact
recapitulation numbers vs the prior contacts-and-distances-v1 1.5B.)_

## Conclusion

_(Fill in after results are in.)_
