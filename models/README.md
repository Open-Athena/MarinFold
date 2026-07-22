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

Experiments using this bridge must declare both `marinfold-models` and
`marinfold` as direct dependencies. uv source mappings are not transitive, so
the experiment should map both packages to their respective repository
subdirectories.

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
