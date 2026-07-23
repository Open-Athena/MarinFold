# models/

Shared library for MarinFold model-training experiments.

The actual training pipelines (per-size, per-loss-mask variant, etc.) live
under `experiments/exp<N>_models_<name>/` as standalone marin executor
graphs. This directory holds only the code those experiments import.

## Scored documents

`marinfold_models.scored_document` bridges the document-structure prototype to
Levanter's public `Trainer` loss callback. Convert a packed document batch with
`levanter_scored_document_batch(...)`, then construct the trainer with
`scored_document_loss` as its loss function. The loss function performs one LM
forward pass and applies each document scorer to its annotated logits range.

```python
batch = levanter_scored_document_batch(packed, Pos=model.Pos)
with Trainer(trainer_config, optimizer, scored_document_loss) as trainer:
    ...
```

Token and explicit-target values are dynamic JAX leaves. Callback identities
and range bounds are static, so production input pipelines should bucket equal
packing/range layouts to reuse compiled train steps.
Vocabulary identity is also static batch metadata. Tagged documents retain
their declaration fingerprint through packing, and the loss rejects a model
whose logits axis is smaller than the declared vocabulary.

Block-causal documents are lowered to an explicit Levanter attention mask:
tokens attend bidirectionally within their integer attention block and to all
earlier blocks in the same packed segment. This is suitable for local
JAX/vanilla prototyping. Levanter's TPU Splash backend does not yet lower
explicit masks; a structured Splash block-mask lowering is required before
using this layout for a production TPU run.

Experiments using this bridge must declare both `marinfold-models` and
`marinfold` as direct dependencies. uv source mappings are not transitive, so
the experiment should map both packages to their respective repository
subdirectories.

## Streaming documents

`marinfold_models.streaming_documents.StreamingDocumentDataset` is the shared
on-the-fly construction path. It owns Parquet shard/row shuffling, process
partitioning, bounded best-fit packing, and checkpointable stream state. An
experiment supplies projected source columns, a deterministic
`row -> Document | None` generator with a versioned `generator_id`, and a
`documents -> Levanter example` adapter.

The included `causal_lm_example_from_documents` adapter implements ordinary
packed next-token training. Other attention and scoring layouts can reuse the
stream and packer with a different adapter.

Loader checkpoints do not pickle `Document` objects or scorer callbacks.
Instead they save the shuffled source cursor and source-row references for
documents held in open and ready packing bins. Restore rereads those few rows
and reruns the deterministic generator. A changed generator fails via its
`generator_id` instead of silently resuming with mixed semantics.

Levanter prefetches ahead of the optimizer. When `global_batch_size` is set,
the dataset retains exact states at the start of each requested optimizer step.
Use `save_checkpoint(path, step=N)` for model checkpoint step `N`, not
`save_checkpoint(path)` after prefetch. Each JAX process needs its own sidecar
because shard assignments and packing bins are process-local. The generic state
is ready for a training-entrypoint checkpoint hook:
`save_model_checkpoint_sidecar(checkpoint_path, step=N)` writes the canonical
`input/streaming-documents-process-<rank>.json`, and the matching load method
restores it before constructing Levanter's `DataLoader`. Stock
`levanter.main.train_lm` does not yet expose a user callback that can invoke
these methods.

## Layout

```
models/
├── pyproject.toml                # subproject pyproject (uses marin via wheels)
├── README.md                     # this file
├── AGENTS.md                     # rules for working under models/
└── marinfold_models/             # importable library
    ├── __init__.py
    ├── defaults.py               # vendored marin default_train / default_tokenize
    ├── scored_document.py        # Document scorer -> Levanter loss bridge
    ├── streaming_documents.py    # On-the-fly construction, packing, state
    └── simple_train_config.py    # vendored marin SimpleTrainConfig
```

`marinfold_models/` is the whole of `models/` — experiment code never
gets copied in here. If an experiment needs something from this
library it imports it (see below); if two experiments need the same
helper, the helper moves here.

## Setup

```bash
cd models
uv venv --python 3.11
uv sync
```

## Importing from experiments

A model-training experiment under `experiments/exp<N>_models_<name>/`
imports the library via:

```python
from marinfold_models.defaults import default_tokenize, default_train
from marinfold_models.simple_train_config import SimpleTrainConfig
```

The experiment's own `pyproject.toml` declares `marinfold-models` as a
local path dependency:

```toml
[tool.uv.sources]
marinfold-models = { path = "../../models" }
```

## Vendored marin code

`marinfold_models/defaults.py` and `marinfold_models/simple_train_config.py`
are frozen copies of `marin/experiments/{defaults,simple_train_config}.py`.
We vendor them because the `marin` wheel only ships `marin/...`, not
the top-level `experiments/` tree.

If marin's `default_train` signature evolves and we need the new
behavior, refresh both vendored files in one PR. No compatibility shims.

## What does NOT belong here

- Specific training scripts (`train_protein_<size>.py`,
  `export_*.py`) — those are experiments.
- Run-specific paths, hyperparameter sweeps, eval glue — those are
  experiments.

If you find yourself adding a function here that only one experiment
uses, it should live in that experiment instead. The bar for promoting
something to the library is "≥ 2 experiments would import it, and the
abstraction is stable enough to not churn under them."
