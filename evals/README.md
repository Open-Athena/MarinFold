# evals/

Shared library for MarinFold model-evaluation experiments.

Individual eval pipelines (e.g. FoldBench, distogram benchmark,
ProteinGym) live as experiments under `experiments/exp<N>_evals_<name>/`.
This directory only holds the **production wrappers** those experiments
import — code that submits iris jobs, materializes eval-result tables,
publishes summaries to GCS / HF — plus symlinks to graduated eval
experiments.

## Layout

```
evals/
├── pyproject.toml             # subproject pyproject
├── README.md                  # this file
├── AGENTS.md                  # rules for working under evals/
├── marinfold_evals/           # importable library (skeleton for now)
│   └── __init__.py
└── <graduated symlinks>       # e.g. foldbench/ → ../experiments/expN_evals_foldbench/
```

## Status

Currently a skeleton. The first eval experiment that needs production
infra will land its shared helpers here; we're not pre-designing the
library before there's a real impl to inform the shape.

## Importing from experiments

An eval experiment under `experiments/exp<N>_evals_<name>/` would
declare:

```toml
[tool.uv.sources]
marinfold-evals = { path = "../../evals" }
```

and import via `from marinfold_evals.<module> import ...`.

## See also

- [`../document_structures/`](../document_structures/) — shared
  toolkit for document-structure impls (`EvalResult`,
  `build_tokenizer`, output writers). Each impl exposes its own
  `python cli.py evaluate ...` for local testing; this library is
  where the production / scaled-out eval wrappers live.
