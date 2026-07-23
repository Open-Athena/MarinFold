# models/

Shared library for MarinFold model-training experiments.

The actual training pipelines (per-size, per-loss-mask variant, etc.) live
under `experiments/exp<N>_models_<name>/` as standalone marin executor
graphs. This directory holds only the code those experiments import.

## Weighted document targets

`marinfold_models.document_loss` bridges document target distributions to
Levanter's public `Trainer` loss callback. Convert a packed document batch with
`levanter_document_batch(...)`, then construct the trainer with
`document_loss` as its loss function. The loss function performs one LM
forward pass and applies weighted categorical cross-entropy.

```python
batch = levanter_document_batch(packed, Pos=model.Pos)
with Trainer(trainer_config, optimizer, document_loss) as trainer:
    ...
```

Each explicit target range stores one sparse candidate-token vector and a dense
`weights[position, candidate]` matrix. Rows are normalized distributions;
`with_targets(...)` is the one-hot convenience API. The bridge flattens
nonzero weights into dynamic JAX leaves before the train step.
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

## Fixed-quota shard documents

`marinfold_models.shard_documents.FixedQuotaShardDocumentDataset` is the shared
on-the-fly construction path. It maps every example index to an epoch, a
deterministically shuffled Parquet shard, and a fixed output slot. An experiment
supplies projected source columns, a deterministic `row -> Document | None`
generator, and a `documents -> Levanter example` adapter.

The included `causal_lm_example_from_documents` adapter implements ordinary
packed next-token training. Other attention and scoring layouts can reuse the
shard packer with a different adapter.

Each shard is shuffled per epoch, constructed completely, and packed with
best-fit decreasing. It then emits exactly `examples_per_shard` slots. Overfull
shards uniformly sample packed bins without replacement; underfull shards
insert zero-loss padding slots. Uniform bin selection gives every document in
an overfull shard the same conditional inclusion probability.

The dataset is genuinely random access and carries no semantic loader state.
Its small LRU cache only avoids reconstructing a shard during sequential
prefetch. Checkpoint resume therefore continues to use Levanter's ordinary
global example indices without loader sidecars or prefetch coordination.

## Layout

```
models/
├── pyproject.toml                # subproject pyproject (uses marin via wheels)
├── README.md                     # this file
├── AGENTS.md                     # rules for working under models/
└── marinfold_models/             # importable library
    ├── __init__.py
    ├── defaults.py               # vendored marin default_train / default_tokenize
    ├── document_loss.py          # Weighted document targets -> Levanter loss
    ├── shard_documents.py        # Stateless fixed-quota shard construction
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
