---
marinfold_experiment:
  issue: 85
  title: "exp: continue the contacts-v1 1.5B run for another epoch (LR re-heat / warm restart)"
  kind: models
  branch: exp/85-contacts-v1-1_5b-reheat
---

# exp: continue the contacts-v1 1.5B run for another epoch (LR re-heat / warm restart)

**Issue:** [#85](https://github.com/Open-Athena/MarinFold/issues/85) · **Kind:** `models` · **Branch:** `exp/85-contacts-v1-1_5b-reheat`

## Question

Does continuing the quick #67 contacts-v1 1.5B run (`protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked-3b5cf2`) for **another epoch with a re-heated learning rate** (a cosine warm-restart from the final `step-11999` checkpoint) lower eval loss — and does it improve contact recapitulation on the exp82 benchmark?

## Hypothesis

The #67 run was a single un-tuned cosine decay over ~2.7 epochs (final eval/loss **2.980**, bpb 0.4232). Its LR decayed to its floor, so the last steps made little progress. A **warm restart** — reload the weights, re-heat the LR, and run ~1 more epoch of shuffled data — should squeeze out additional loss reduction "for free" (no new data, ~1 epoch of compute) and give a second contacts-v1 checkpoint to compare against both #67 and the #61/#75 tuned sweep. Whether the loss gain translates into better contact prediction is the open question (exp82 showed #67 is near-chance at *de novo* contact prediction).

## Background

- **#67** — the quick/simple 1.5B run. Recipe in `experiments/exp67_models_contacts_v1_1_5b/`: unmasked next-token loss, shuffled (Feistel) train stream, LR 3.5e-4 peak / cosine, batch 128, seq 8192, v5p-8 @ us-east5-a, 12k steps (~2.7 epochs). Final levanter checkpoint: `gs://marin-us-east5/protein-structure/MarinFold/exp67_contacts_v1_1_5b/checkpoints/protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked-3b5cf2/checkpoints/step-11999`.
- **#82** — contact-prediction eval harness + benchmark (heatmaps, seeding, poly-Ala). The reusable harness can score any contacts-v1 checkpoint, so the continued model drops straight in.
- **#61/#75** — eric-czech's tuned sweep on the *same* contacts-v1 corpus (Qwen3 1.5B, LR/WD sweep, epochs 1/2/4/8). Best finished run so far (`e2-lr7e-4-wd0p1`) is only ~0.01 below the #67 baseline; the 4-/8-epoch runs are still in flight. This experiment is a cheap, orthogonal lever on the *existing* #67 model.

## Approach

A **warm-restart continuation** via marin/levanter's `initialize_from_checkpoint_path` (loads model weights only → fresh step-0, fresh optimizer state, fresh LR schedule, fresh shuffled data loader; `reset_data_loader_on_init=True`). Reuse the exact #67 recipe (unmasked loss, shuffle, full-val eval, us-east5-a v5p-8) and change only:

- **init**: weights from `step-11999`.
- **slice + batch**: **v5p-32** (4× #67's v5p-8) with **batch 512** (4×). v5p-8 was thrashing on preemption (10 in 11 min, never reached step 0); v5p-32 has capacity. Per-chip batch stays 32 (identical to #67 ⇒ same memory, no OOM).
- **LR re-heat**: peak **4.0e-4** = 2× the batch-128 value of **2.0e-4** (the chosen moderate re-heat), scaled by √4 for the 4× batch. Same cosine-to-min_lr_ratio shape and 0.1 warmup as #67.
- **length**: **~1,125 steps** (≈1 epoch at batch 512; ¼ of the 4,490 steps it takes at batch 128).
- **data_seed**: 1 (fresh permutation so the extra epoch isn't the identical order as #67's last epoch).

> Recipe evolution: the run was first launched on v5p-8 / batch 128 / LR 2.0e-4 / 4,500 steps, but the v5p-8 preemptible pool thrashed (never reached step 0). It was moved to v5p-32 with proportional batch (×4) and √-scaled LR (×2), per the standard LR-vs-batch rule.

Then export `step-{final}` to HF, publish to the open-athena bucket, and run the exp82 harness (precision@top-{L,L/2,L/5} long/medlong + the benchmark heatmaps/seeding) head-to-head vs #67.

## Success criteria

- A continued/​re-heated checkpoint that trains cleanly and reaches **eval/loss < 2.980** (beats #67's final).
- HF export published to the bucket (tokenizer co-located).
- exp82 harness numbers for the continued model vs #67 (does lower loss → better contact recapitulation?).
- **Stretch:** a small re-heat-peak comparison (e.g. 2.0e-4 vs 3.5e-4) if the first restart looks promising.

## Files

- `train_protein_1_5b_contacts_v1_reheat.py` — the warm-restart training step. Sets `initialize_from_checkpoint_path` to #67's `step-11999`, **v5p-32**, **batch 512**, re-heat peak LR **4.0e-4**, **`num_train_steps=1125`**, `data_seed=1`. Run name `protein-contacts-1_5b-contacts-v1-unmasked-reheat-e3-bs512`.
- `contacts_v1_train_common.py` — copied from exp67 and extended with an `initialize_from_checkpoint_path` knob on `build_train_step` (forwarded to `SimpleTrainConfig`; `default_train` already supports it). Everything else (corpus, tokenizer, **token caches reused via exp67's `MARIN_PREFIX`**, TPU recipe) is unchanged.
- `export_protein_1_5b_contacts_v1_reheat.py` — HF export; fill in the real `-<wandb-runid>` suffix + final step after launch.

### Mechanism — warm restart vs resume

`reset_data_loader_on_init=True` (the default) routes `initialize_from_checkpoint_path` to `TrainLmConfig.initialize_from_checkpoint_path` (`marinfold_models/defaults.py:314`), which loads **model weights only** and starts a fresh run: step 0, fresh optimizer state, fresh cosine LR schedule (so our `learning_rate`/`warmup` = the re-heat), fresh shuffled data loader. This is a true warm restart — *not* `trainer.initialize_from` (which would resume the decayed schedule + optimizer + data position).

### Cache reuse

The module keeps exp67's `MARIN_PREFIX` (`…/exp67_contacts_v1_1_5b`) so the tokenize steps resolve exp67's existing 4.7B-token train + val caches instead of re-tokenizing. The warm-restart run's checkpoints/exports land under their own W&B-run-name subdir, so they don't collide with #67's.

The reused caches are read with `PrebuiltLmDatasetFormat(input_ids_key="input_ids")`
and `auto_build_caches=False`. That matters for the 2026-06-17 cache-reader
failure: the old path reused the raw-text tokenizer format, whose
`BatchTokenizer.output_exemplar` is a Python list and can derive the pytree field
`input_ids/0`; the cache ledger records the array field `input_ids`. Reading the
already-tokenized cache as prebuilt token IDs makes the exemplar an ndarray leaf
and derives `input_ids`, matching the ledger. `auto_build_caches=False` also
prevents a missing cache from silently trying to rebuild raw text with the
prebuilt reader.

## Launch

```bash
cd experiments/exp85_models_continue_the_contacts_v1_15b_run
uv venv && uv sync --extra tpu

# Train (warm restart, ~1 epoch). WANDB_API_KEY must be in the launching env —
# build_train_step forwards it into the pod's env_vars.
WANDB_API_KEY=<key> uv run iris --cluster marin job run --no-wait \
    --enable-extra-resources --memory=16GB --disk=16GB --cpu=1 \
    --extra=tpu --zone=us-east5-a \
    -- python -m train_protein_1_5b_contacts_v1_reheat

# After step-1124 (= num_train_steps-1) lands on GCS, fill the real -<wandb-runid>
# suffix + step into export_protein_1_5b_contacts_v1_reheat.py, then:
uv run iris --cluster marin job run --no-wait --enable-extra-resources \
    --memory=32GB --disk=16GB --cpu=4 -- python -m export_protein_1_5b_contacts_v1_reheat
```

> **iris client freshness (resolved):** the controller rejects root submissions
> whose client is older than a rolling 14-day window
> (`iris/cluster/controller/service.py` → `_check_client_freshness`). Depending on
> PyPI `marin-iris` (its wheel carries a fresh `BUILD_DATE`) makes a plain
> `uv run iris` pass — **no editable-iris workaround needed.** (That hack was only
> required while pinned to the frozen GitHub `marin-*-latest` wheels.)

### Launched run

- **v5p-32 run:** job `/bizon/iris-run-job-20260616-223513` (submitted 2026-06-16 22:35 UTC), v5p-32 @ us-east5-a, batch 512, LR 4.0e-4, ~1125 steps. W&B run `protein-contacts-1_5b-contacts-v1-unmasked-reheat-e3-bs512` (entity open-athena).
- Superseded v5p-8 attempt: job `/bizon/iris-run-job-20260616-214924` — terminated on preemption thrashing before reaching step 0.
- Tokenize steps reused exp67's caches ("already succeeded") — no re-tokenization.

## Status — experiment-side cache-read workaround ready (2026-06-18)

The original code + recipe were blocked before step 0 by a marin/levanter
**cache-reader failure on the iris TPU worker**. Authoritative analysis, a
verified minimal repro, and the upstream reader-fallback fix are captured in
[`MARIN_CACHE_READER_BUG.md`](MARIN_CACHE_READER_BUG.md).

This branch now applies the stronger experiment-side workaround suggested in
that note: read exp67's existing token caches as **prebuilt `input_ids` arrays**
rather than as raw-text tokenizer output. Local verification against the real GCS
caches succeeds through the old failure point:

- train cache:
  `gs://marin-us-east5/protein-structure/MarinFold/exp67_contacts_v1_1_5b/tokenized/contacts-v1-663ba6/train`
  loads with ledger field `input_ids` and `_ensure_shard_field_offsets("input_ids")`.
- validation cache:
  `gs://marin-us-east5/protein-structure/MarinFold/exp67_contacts_v1_1_5b/tokenized/contacts-v1-val-92827b/validation`
  loads with ledger field `input_ids` and `_ensure_shard_field_offsets("input_ids")`.
- smoke config confirms both `DatasetComponent`s use `PrebuiltLmDatasetFormat`,
  their exemplar flattens to `input_ids`, and `LmDataConfig.auto_build_caches`
  is `False`.

This should let a relaunched TPU run step past the `input_ids/0` failure without
waiting for an upstream marin/levanter release.

**Symptom.** The training step dies in the data loader with:
```
ValueError: Sharded cache ledger missing input_ids/0 count for shard part-00000-of-00133
```
The cache ledger's field key is `input_ids`; the reader derives `input_ids/0`
(jax flattens a python-list `output_exemplar`) and the count lookup has no
fallback. Reproduced locally; see the bug doc for the `_resolve_ledger_field` fix.

**Side blockers hit and RESOLVED along the way** (so they don't distract the next person):
1. iris client 14-day freshness gate → use PyPI `marin-iris` (fresh `BUILD_DATE`); no editable-iris hack.
2. tokenizer `repo@rev` rejected at train time → bare repo id.
3. stale GitHub `marin-*-latest` find-links (`0.99.dev20260529`) → depend on the marin source dists (`marin-core` etc.) from **PyPI** with a `<0.3` bound (the frozen `0.99.dev` sorts above the fixed `0.2.x.dev`). `uv.lock` now pins `0.2.19.dev202606171019`.

**The original blocker was NOT versioning — that's the key, surprising finding:**
- The worker's captured W&B `requirements.txt` shows it ran the **fixed** marin
  (`marin-levanter==0.2.19.dev202606171019`; earlier attempts `0.2.0`) and **still
  failed** with `input_ids/0`. So "the worker is on a stale/baked frozen marin" is
  **not** the explanation. *(An earlier hypothesis in git history said the TPU image
  bakes frozen `0.99.dev`; the W&B evidence disproves it.)*
- That same `0.2.19.dev` reads the **same cache fine locally** (it normalizes the
  list exemplar → derives `input_ids`). The end-to-end failure **does not reproduce
  locally**.
- ⇒ **Open question for the handoff:** why does the TPU worker derive `input_ids/0`
  while a local read of the same cache + same marin version derives `input_ids`?
  Candidates: the pre-#6014 cache has no stored exemplar so the reader falls back to
  the list `output_exemplar`; an effective build/exemplar difference on the pod; a
  jax/haliax tree-flattening difference. **The reader-fallback fix in the bug doc
  makes the reader correct regardless of which side derives the mismatch** — landing
  it in marin/levanter remains the cleanest upstream fix, but exp85 no longer
  needs to hit that code path.

**Reference (for whoever investigates):**
- Failing W&B runs: `open-athena/MarinFold`, names `…-reheat-e3-bs512[-v2]`, all `state=failed` (their `requirements.txt` show the installed marin versions).
- Recent failing iris jobs: `/bizon/iris-run-job-20260617-153047`, `…-002506`, `…-001956`.
- Shared cache: `gs://marin-us-east5/protein-structure/MarinFold/exp67_contacts_v1_1_5b/tokenized/contacts-v1-663ba6` (ledger `field_counts = {input_ids}`).
- Local repro of the reader intolerance: see `MARIN_CACHE_READER_BUG.md` (Minimal repro).

**To finish the experiment:**
1. Relaunch (see **Launch**). Success = the run steps past 0 and logs `train/loss` (target eval/loss < 2.980).
2. After `step-1124` lands, fill the `-<wandb-runid>` + step into `export_protein_1_5b_contacts_v1_reheat.py` and run the export.
3. Publish the HF export to the open-athena bucket (tokenizer co-located); run the #82 harness head-to-head vs #67.

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
