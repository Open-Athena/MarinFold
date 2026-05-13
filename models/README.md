# models/

Shared library for MarinFold model-training experiments.

The actual training pipelines (per-size, per-loss-mask variant, etc.) live
under `experiments/exp<N>_models_<name>/` as standalone marin executor
graphs. This directory only holds code those experiments import — and
symlinks to experiments that have been **graduated** (see
[`../experiments/README.md`](../experiments/README.md#graduating-an-experiment)).

## Layout

```
models/
├── pyproject.toml                # subproject pyproject (uses marin via wheels)
├── README.md                     # this file
├── AGENTS.md                     # rules for working under models/
├── marinfold_models/             # importable library
│   ├── __init__.py
│   ├── defaults.py               # vendored marin default_train / default_tokenize
│   └── simple_train_config.py    # vendored marin SimpleTrainConfig
└── <graduated symlinks>          # e.g. contacts_and_distances_v1_baseline/
                                  # → ../experiments/exp<N>_models_<name>/
```

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
