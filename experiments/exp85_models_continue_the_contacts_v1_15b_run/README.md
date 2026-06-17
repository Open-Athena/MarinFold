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

- `train_protein_1_5b_contacts_v1_reheat.py` — the warm-restart training step. Sets `initialize_from_checkpoint_path` to #67's `step-11999`, re-heat peak LR `2.0e-4`, `num_train_steps=4500`, `data_seed=1`. Run name `protein-contacts-1_5b-contacts-v1-unmasked-reheat-e3`.
- `contacts_v1_train_common.py` — copied from exp67 and extended with an `initialize_from_checkpoint_path` knob on `build_train_step` (forwarded to `SimpleTrainConfig`; `default_train` already supports it). Everything else (corpus, tokenizer, **token caches reused via exp67's `MARIN_PREFIX`**, TPU recipe) is unchanged.
- `export_protein_1_5b_contacts_v1_reheat.py` — HF export; fill in the real `-<wandb-runid>` suffix + final step after launch.

### Mechanism — warm restart vs resume

`reset_data_loader_on_init=True` (the default) routes `initialize_from_checkpoint_path` to `TrainLmConfig.initialize_from_checkpoint_path` (`marinfold_models/defaults.py:314`), which loads **model weights only** and starts a fresh run: step 0, fresh optimizer state, fresh cosine LR schedule (so our `learning_rate`/`warmup` = the re-heat), fresh shuffled data loader. This is a true warm restart — *not* `trainer.initialize_from` (which would resume the decayed schedule + optimizer + data position).

### Cache reuse

The module keeps exp67's `MARIN_PREFIX` (`…/exp67_contacts_v1_1_5b`) so the tokenize steps resolve exp67's existing 4.7B-token train + val caches instead of re-tokenizing — which also avoids the known marin-latest fresh-tokenize cache-ledger bug (#6008/#6014). The warm-restart run's checkpoints/exports land under their own W&B-run-name subdir, so they don't collide with #67's.

## Launch

### iris client freshness (important)

The iris controller gates **root job submissions** on client freshness: it
rejects clients whose `client_revision_date` is older than a **rolling 14-day
window** (`iris/cluster/controller/service.py` → `_check_client_freshness`,
`FRESHNESS_WINDOW=14d`). That date comes from `iris.version.client_revision_date()`:
a wheel's stamped `_build_info.BUILD_DATE`, else a `git log` on the iris source
tree, else `""` (→ the old default `2026-04-22`).

Consequences for this experiment:
- The **pinned `marin-iris` wheel is frozen at build `2026-05-29`** (the public
  `marin-*-latest` indices stopped updating), which fell outside the window on
  ~2026-06-12 and is now rejected. A git-URL repin does **not** fix it: uv copies
  the source *without* `.git`, so `BUILD_DATE` is empty and the `git log` fallback
  finds no repo → the stale `2026-04-22` default (worse).
- The working fix is an **editable install of iris from a local, recently-pulled
  marin checkout**, so the `git log` fallback reports a fresh date:
  ```bash
  git -C ~/git/marin pull          # ensure lib/iris commit date is within ~14 days
  uv pip install -e ~/git/marin/lib/iris   # after `uv sync`; makes the client fresh
  ```
  Only the **launching client** needs this. The TPU **worker** builds from this
  dir's frozen-wheel `pyproject.toml` and is **exempt** — the driver's child-job
  submission is not a "root" submission, so its stale iris is fine (verified: the
  first launch's child TPU step submitted and ran normally).

### Commands

```bash
cd experiments/exp85_models_continue_the_contacts_v1_15b_run
uv venv && uv sync --extra tpu
uv pip install -e ~/git/marin/lib/iris      # client-freshness fix (see above)

# Train (warm restart, ~1 epoch). WANDB_API_KEY must be in the launching env —
# build_train_step forwards it into the pod's env_vars. Use `uv run --no-sync`
# so the editable iris isn't reverted by an implicit re-sync.
WANDB_API_KEY=<key> uv run --no-sync iris --cluster marin job run --no-wait \
    --enable-extra-resources --memory=16GB --disk=16GB --cpu=1 \
    --extra=tpu --zone=us-east5-a \
    -- python -m train_protein_1_5b_contacts_v1_reheat

# After step-4499 lands on GCS, fill in the runid + step in the export script, then:
uv run --no-sync iris --cluster marin job run --no-wait --enable-extra-resources \
    --memory=32GB --disk=16GB --cpu=4 -- python -m export_protein_1_5b_contacts_v1_reheat
```

### Launched run

- **v5p-32 run:** job `/bizon/iris-run-job-20260616-223513` (submitted 2026-06-16 22:35 UTC), v5p-32 @ us-east5-a, batch 512, LR 4.0e-4, ~1125 steps. W&B run `protein-contacts-1_5b-contacts-v1-unmasked-reheat-e3-bs512` (entity open-athena).
- Superseded v5p-8 attempt: job `/bizon/iris-run-job-20260616-214924` — terminated on preemption thrashing before reaching step 0.
- Tokenize steps reused exp67's caches ("already succeeded") — no re-tokenization.

## Status — BLOCKED on the TPU worker's marin (2026-06-16)

The recipe is complete and correct, but the run cannot reach step 0 because of a
marin-infrastructure issue on the **TPU worker pod** (not the experiment code).

Chain of frozen-wheel blockers hit + fixed:
1. **iris client freshness gate** → editable iris from `~/git/marin` (✅, committed/documented).
2. **tokenizer `repo@rev`** rejected at train time → bare repo id (✅).
3. **cache-ledger reader #6008** → pin the marin workspace to a git source
   (`marin-core` @ e78c54a8 = 2026-06-10, which has the #6014 reader fix). Resolves
   + builds; **verified the fixed reader loads the shared cache locally** (4.1M rows).

**Remaining blocker:** the fix lands on the **CPU driver** but NOT the **TPU worker**:
- A `DIAG85` probe + a local reader test proved the **driver** gets `marin-levanter==0.2.0`
  with the fixed reader (`has_exemplar_for=False`), and the driver builds it from
  source (~7 min).
- The **TPU worker** (`v5p-32`) finishes `uv sync` in **~50 s** — far too fast to build
  from source — and runs the **OLD frozen reader** (fails: `Sharded cache ledger missing
  input_ids/0`). It is fast-syncing a pre-baked/cached **frozen** marin from the TPU pod
  environment, ignoring the git-source lock.
- Disproven experiment-side levers: the committed `uv.lock` (522 KB) pins the git source;
  bumping `data_seed` and changing the run name (fresh task hash `f417f6b7`) did **not**
  change the TPU worker's behavior. The driver and child share the same bundle/lock yet
  resolve marin differently — so this is the **marin/iris TPU runtime image / pod env**,
  which can't be overridden from this pyproject.

**Unblock options (need the marin side):**
1. **Republish fixed `marin-*` wheels** (≥ 2026-06-02) — then revert to the published-wheel
   pyproject and the worker just works. Cleanest; needs eric (on vacation).
2. Rebuild/refresh the **TPU runtime image** so its baked marin is ≥ 2026-06-02, or make
   the worker `uv sync` actually honor the git-source lock (force `--reinstall`).
3. Run exp85's train step from eric's marin-workspace environment (his #75 worker already
   reads this exact cache fine).

Everything is committed (PR #86); the run is one marin-side fix away, no re-tokenize needed.

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
