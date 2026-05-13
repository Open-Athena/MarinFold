# models/AGENTS.md

Rules for agents working under `models/`. Layered on top of the root
`AGENTS.md`.

## Scope

`models/` is the **library** for model-training experiments — not a
collection of training scripts. Specific training pipelines live as
experiments under `experiments/exp<N>_models_<name>/` and import
`marinfold_models.*` from here.

Graduated experiments may also appear as symlinks under this
directory (e.g. `models/contacts_and_distances_v1_baseline/` →
`../experiments/exp<N>_models_contacts_and_distances_v1_baseline/`).
Those are not part of the library; the library is `marinfold_models/`
only.

## Hard rules

1. **W&B project is `MarinFold`.** Every run logs to
   `https://wandb.ai/open-athena/MarinFold`. Don't set
   `WANDB_PROJECT` / `WANDB_ENTITY` to anything else. The vendored
   `marinfold_models.defaults.default_train` hardcodes `project="MarinFold"`.

2. **Cache discipline.** Marin's executor caches every step by the
   hash of its config. Use `versioned(...)` for any scalar that
   should bust the cache when changed — plain dataclass fields are
   ignored by the hasher.

3. **Co-locate TPU with checkpoint bucket.** Pin TPU jobs to
   `us-east5-a` (or whichever zone matches the bucket region) via
   `ResourceConfig.with_tpu(..., zone="us-east5-a")`. Cross-region
   checkpoint I/O is slow and expensive.

4. **Tokenizer revisions are pinned.** Always use the `repo@revision`
   suffix in tokenizer names so levanter's cache is keyed by
   revision. Bumping the revision is a conscious choice — don't
   drop the pin to "just get the latest".

5. **Always save the tokenizer with the model.** When pushing a
   checkpoint to HuggingFace — bucket or public `models` repo — the
   tokenizer files (`tokenizer.json`, `tokenizer_config.json`,
   `special_tokens_map.json`, …) go in the same repo / revision as
   the weights. A model without its tokenizer is unloadable for
   downstream eval, vLLM serving, and reproducibility checks.
   `convert_checkpoint_to_hf_step` (used by every
   `export_protein_*.py`) does this automatically when its
   `tokenizer=` arg is set — keep it set.

6. **Don't import from `experiments/`.** Library code cannot depend
   on experiment code; the dependency is one-way. If two experiments
   share a function, the function belongs here.

7. **Don't dump a function here just because it might be reused.**
   The bar is "≥ 2 experiments would import it, and the abstraction
   is stable." Premature consolidation creates churn.

## Vendored marin code

`marinfold_models/defaults.py` and `marinfold_models/simple_train_config.py`
are frozen copies of `marin/experiments/{defaults,simple_train_config}.py`.
We vendor them because the `marin` wheel only ships `marin/...`, not
the top-level `experiments/` tree.

If marin's `default_train` signature evolves, refresh both vendored
files in one PR — no compatibility shims.

## Graduated symlinks

When an experiment is graduated (see
[`experiments/AGENTS.md`](../experiments/AGENTS.md)), it gets a
symlink here named after the experiment's name (dropping the
`exp<N>_models_` prefix). Don't edit through the symlink — edit the
real path at `experiments/exp<N>_models_<name>/`.

Tools that walk directories (pytest, mypy, ruff) may follow these
symlinks and double-discover code. The fix: invoke the tool on
`marinfold_models/` explicitly (`uv run pytest marinfold_models/`),
not on `.` from this directory.
