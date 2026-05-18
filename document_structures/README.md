# document_structures/

Shared toolkit for MarinFold document-structure implementations.

A **document structure** is a protein-document format. Each format
lives as a self-contained experiment under
`experiments/exp<N>_document_structures_<name>/` with its own
`cli.py` and the modules it dispatches to (`generate.py`,
`inference.py`, plus shared `vocab.py` / `parse.py`).

This library is intentionally tiny — it's the three pieces every
impl needs from a common place:

- `EvalResult` — the return shape of an impl's `evaluate` function.
- `build_tokenizer(tokens)` — build a `PreTrainedTokenizerFast` from
  an ordered token list (WordLevel + whitespace pre-tokenization,
  `<pad>` at id 0 and `<eos>` at id 1).
- `write_docs` / `write_predictions` / `write_eval` — parquet / jsonl
  writers for the three standard output shapes, with consistent
  schema across formats (the `structure` column, `extras` flattening,
  per-example sidecars).

Everything else — argparse, model loading, IO conventions specific
to one format — is the impl's responsibility.

## Layout

```
document_structures/
├── pyproject.toml
├── README.md
├── AGENTS.md
├── marinfold_document_structures/
│   ├── __init__.py       # re-exports the public API
│   ├── core.py           # EvalResult + build_tokenizer
│   └── writers.py        # write_docs / write_predictions / write_eval
└── <graduated symlinks>  # e.g. contacts_and_distances_v1/ → ../experiments/...
```

## Setup

```bash
cd document_structures
uv venv --python 3.11
uv sync
```

Impls have their own pyproject.toml + uv environment; they install
this library via a path dependency.

## Authoring a document structure

```
experiments/exp<N>_document_structures_<name>/
├── vocab.py        # ordered token list + NAME + CONTEXT_LENGTH
├── parse.py        # input parsing (gemmi, biopython, …)
├── generate.py     # generate_documents(...) -> Iterator[str]
├── inference.py    # predict(cfg) -> Iterator[dict]; evaluate(cfg) -> EvalResult
├── cli.py          # argparse driver
└── pyproject.toml
```

`vocab.py` defines the token list. `parse.py` reads inputs.
`generate.py` and `inference.py` are plain library modules — they
export functions, not classes wrapping the Protocol shape.

`cli.py` is the entry point — argparse subparsers for `generate`,
`infer`, `evaluate`, `tokenizer`. Each subcommand assembles a
config dataclass from the args, calls the relevant function, and
hands the result to one of the library's writers:

```python
# cli.py (sketch)
import argparse
from pathlib import Path

from marinfold_document_structures import (
    build_tokenizer, write_docs, write_predictions, write_eval,
)

import generate
import inference
from vocab import NAME, all_domain_tokens


def cmd_generate(args):
    docs = generate.generate_documents(
        input_path=args.input, num_docs=args.num_docs,
        context_length=args.context_length,
        config=generate.GenerationConfig(...),
    )
    write_docs(args.out, docs, structure_name=NAME)


def cmd_evaluate(args):
    cfg = inference.InferenceConfig(model=args.model, ...)
    result = inference.evaluate(cfg)
    write_eval(args.out, result, structure_name=NAME)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    # ... subparsers, each with .set_defaults(func=cmd_*)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

Run via `python cli.py <subcommand> ...` from the impl dir (with the
impl's venv active).

## See also

- [`../experiments/exp1_document_structures_contacts_and_distances_v1/`](../experiments/exp1_document_structures_contacts_and_distances_v1/)
  — the reference impl.
- [`../data/`](../data/) — production data-gen wrappers built on top
  of document structures.
- [`../evals/`](../evals/) — production eval wrappers built on top
  of document structures.
