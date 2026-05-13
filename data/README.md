# data/

Shared library for MarinFold data-generation experiments.

Individual data pipelines (e.g. AFDB → tokenizable docs, ESM
metagenomic atlas ingestion, ProteinGym mirroring) live as experiments
under `experiments/exp<N>_data_<name>/`. This directory holds the
**production wrappers** those experiments import to launch zephyr
pipelines and publish datasets to GCS / HF, plus symlinks to graduated
data experiments.

## Layout

```
data/
├── pyproject.toml             # subproject pyproject
├── README.md                  # this file
├── AGENTS.md                  # rules for working under data/
├── marinfold_data/            # importable library (skeleton for now)
│   └── __init__.py
└── <graduated symlinks>       # e.g. afdb_contacts/ → ../experiments/expN_data_afdb_contacts/
```

## Status

Currently a skeleton. The first data-gen experiment that needs
production infra will land its shared helpers here.

## Where datasets end up

- **HuggingFace**: `huggingface.co/datasets/timodonnell/<name>` for
  text/tokenized datasets that need to be loaded by levanter via
  `hf://datasets/...` URIs. Pre-existing: `protein-docs`,
  `afdb-1.6M`.
- **GCS**: `gs://marin-us-east5/<...>` for large intermediate
  artifacts produced by zephyr pipelines (e.g. tokenized parquets,
  cached structure features).

The git repo never holds large dataset files — see the root
`README.md` for the policy.

## Importing from experiments

A data experiment under `experiments/exp<N>_data_<name>/` would
declare:

```toml
[tool.uv.sources]
marinfold-data = { path = "../../data" }
```

and import via `from marinfold_data.<module> import ...`.

## See also

- [`../document_structures/`](../document_structures/) — the
  `DocumentStructure.generate(input)` interface defines how a single
  input structure becomes a single training document. Data
  experiments wrap that to produce corpora at scale.
