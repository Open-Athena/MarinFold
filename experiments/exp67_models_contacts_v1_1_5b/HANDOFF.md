# exp67 — HANDOFF / resume notes

> **STATUS (superseded): the build is complete.** The code, recipe, and the
> three places where this plan diverged from reality (tokenizer repo, eval
> strategy, marin dependency drift) are documented authoritatively in
> [`README.md`](README.md) → *Implementation notes & decisions*. This file is
> kept as the historical plan; where it conflicts with the README, the README
> wins. Only the launch + run-recording remain (see README "Launch").

**Status file location:** `experiments/exp67_models_contacts_v1_1_5b/HANDOFF.md`
(this file). Approved plan: `~/.claude/plans/linear-plotting-hartmanis.md`.
Issue: [#67](https://github.com/Open-Athena/MarinFold/issues/67) (the *quick/simple*
1.5B run; #61 is the carefully-tuned one by @eric-czech).

## Confirmed design decisions (from the user)
- **Loss = UNMASKED** — next-token loss on the whole document, no `loss_weight_fn`
  (mirror `exp0/train_protein_1b_unmasked.py`, not the distance-masked builder).
- **Length ≈ 2–3 epochs.** 1 epoch ≈ 4,490 steps (train ≈ 4.7B tok / (128×8192 =
  1.05M tok/step)). Target **~12,000 steps (~2.7 epochs)**.
- Otherwise reuse the 1.5B recipe: v5p-8 @ **us-east5-a**, LR **3.5e-4**, batch
  **128**, seq **8192**, weight_decay **0.01**, warmup **0.1**.
- **Must shuffle** training data (issue requirement) — corpus shards are physically
  **round-descending (highest-pLDDT last)**, so unshuffled training is badly biased.
- **Downsample eval to ~5000 docs**, shuffled (via `max_eval_batches`).

## Key facts (verified during exploration)
- **Corpus** (exp53, published): `hf://buckets/open-athena/MarinFold/data/document_structures/contacts_v1/<split>/shard_*.parquet`,
  splits `train`/`val`/`test`, text column **`document`**. 4.21M docs, mean 1132
  tok/doc; train ≈ 4.13M docs (~4.7B tok), val ≈ 41.9K docs.
- **Tokenizer:** contacts-v1 tokenizer = **2845 tokens**, distinct from the old
  `protein-docs-tokenizer`. Built via `cd marinfold && uv run contacts-v1 tokenizer`.
  `open-athena/contacts-v1-tokenizer` does NOT exist yet (verified 404) → must push
  a pinned, levanter-loadable copy. Pushing under `open-athena/` may need an
  org-scoped HF token (workstation default may be timodonnell-only; `hf auth whoami`
  shows org membership — open-athena IS listed).
- **Recipe to mirror:** `experiments/exp0_models_protein_docs_initial_port/` —
  `protein_train_common.py`, `train_protein_1_5b_distance_masked.py` (the 1.5B
  LlamaConfig: max_seq_len=8192, hidden_dim=2048, intermediate_dim=8192,
  num_heads=32, num_kv_heads=8, num_layers=24), `train_protein_1b_unmasked.py`
  (no-mask variant), `build_hf_export_step` for the export script. All call
  vendored `marinfold_models.defaults.{default_train,default_tokenize}`.

## DONE so far
- [x] Branch `exp/67-contacts-v1-1_5b` created (currently checked out).
- [x] Scaffolded `experiments/exp67_models_contacts_v1_1_5b/` via
  `scripts/scaffold.py --issue 67 --kind models --name contacts_v1_1_5b`
  → created `README.md`, `build_summary.py`, `summary_narrative.md`.

## NOT yet done (next steps, in order)
1. **pyproject + venv**: `cp ../exp0_models_protein_docs_initial_port/pyproject.toml .`,
   rename `[project].name` → `exp67-models-contacts-v1-1-5b` (and description),
   `mkdir data plots && touch data/.gitkeep plots/.gitkeep`,
   `uv venv --python 3.11 && uv sync --extra tpu`.
2. **Tokenizer**: `cd marinfold && uv sync --extra contacts-v1 && uv run contacts-v1
   tokenizer --push open-athena/contacts-v1-tokenizer`. Capture the commit SHA;
   reference as `open-athena/contacts-v1-tokenizer@<sha>`. Verify vocab size 2845.
3. **`contacts_v1_train_common.py`** (adapt `protein_train_common.py`):
   - `CONTACTS_V1_TOKENIZER = "open-athena/contacts-v1-tokenizer@<sha>"`
   - `HF_DATASET_BASE = "hf://buckets/open-athena/MarinFold/data/document_structures/contacts_v1"`
   - `default_tokenize` for `train/` and `val/`, `TextLmDatasetFormat(text_key="document")`,
     **no `override_output_path`** (fresh corpus).
   - `PROTEIN_RESOURCES_USE5` = v5p-8 @ us-east5-a (verbatim from exp0).
   - `build_train_step(...)`: components `pack=True`, `block_cross_document_attention=True`,
     **NO `loss_weight_fn`**; data config **`shuffle=True`** + a `data_seed`;
     `max_eval_batches ≈ 6` (~5.5K docs at ~7 docs/packed-seq, eval batch 128).
   - ⚠️ **Confirm exact levanter field names** once venv synced: `LmDataConfig.shuffle`
     (bool/era) and `TrainerConfig`/`SimpleTrainConfig.max_eval_batches`.
4. **`train_protein_1_5b_contacts_v1.py`** (adapt `train_protein_1_5b_distance_masked.py`):
   1.5B `LlamaConfig` verbatim; `num_train_steps=versioned(12_000)`, LR `versioned(3.5e-4)`,
   batch 128, seq 8192, wd 0.01, warmup 0.1, steps_per_eval=250, steps_per_export=2000,
   `env_vars={"WANDB_ENTITY":"open-athena"}`. `default_train(name="protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked",
   wandb_name=<same as name>, wandb_group="protein-training",
   tags=["protein","contacts-v1","1_5b","llama","unmasked"])`. `executor_main([...])`.
5. **`export_protein_1_5b_contacts_v1.py`** (optional, mirror exp0 `build_hf_export_step`,
   `tokenizer=CONTACTS_V1_TOKENIZER`, canonical `checkpoints/<run>/hf/step-<N>/`).
6. **README + summary_narrative.md**: fill question/hypothesis/approach/success;
   then `cd scripts && uv run python itemize.py` to regen `experiments/index.md`.
7. **Smoke + launch**: `python -c "import train_protein_1_5b_contacts_v1"` builds the
   step graph; smoke run (tokenize step + first ~100–200 steps, watch loss); then
   launch the full ~12K-step run via `uv run iris --config=... job run ... --zone=us-east5-a
   -- python -m train_protein_1_5b_contacts_v1`. Issue allows ≤10 test runs.
8. **Record**: after W&B URL is in hand, `python scripts/history.py new --wandb-url …
   --wandb-name … --experiment exp67_models_contacts_v1_1_5b --kind models --short "…"
   --iris-jobs …` then `python scripts/history.py update-index`. Open a PR vs
   `origin/main` (branch+PR rule). **Do NOT close the issue** — human-only.

## Working-tree note
`marinfold/.../contacts_v1/explore_documents.ipynb` shows as modified — this was
**already dirty at session start**, unrelated to exp67; leave it alone.

## Task list (harness)
#1 scaffold (in progress → nearly done), #2 tokenizer, #3 training code,
#4 export script, #5 README/narrative, #6 smoke+launch+record.
