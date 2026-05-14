# document_structures/

The shared interfaces + local-testing CLI for MarinFold document
structures.

A **document structure** is split across two files under
`experiments/exp<N>_document_structures_<name>/`:

- **`generate.py`** — exports `get_generator() -> Generator`. Builds
  training documents from input structures (PDB / mmCIF / AFDB).
- **`inference.py`** — exports `get_inference() -> Inference`. Runs
  a trained model against inputs (with or without ground truth).
  Powers both the `infer` (no GT) and `evaluate` (with GT) CLI
  subcommands.

Both files expose `name`, `context_length`, `tokens()` (they must
agree on the vocab — shared via a private `_vocab.py` in the same
dir) and an `add_args(parser)` hook for adding their own CLI flags.

## Layout

```
document_structures/
├── pyproject.toml                          # subproject pyproject; CLI entry point
├── README.md                               # this file
├── AGENTS.md                               # rules for working under document_structures/
├── marinfold_document_structures/
│   ├── __init__.py
│   ├── interface.py                        # Generator + Inference Protocols, EvalResult, build_tokenizer, load_generator/inference
│   └── cli.py                              # marinfold-document-structure CLI
└── <graduated symlinks>                    # e.g. contacts_and_distances_v1/ → ../experiments/exp<N>_document_structures_<name>/
```

## Setup

```bash
cd document_structures
uv venv --python 3.11
uv sync
```

That installs the `marinfold-document-structure` CLI command.

## The CLI

Four subcommands. All take `<impl_dir>` (the experiment directory
containing `generate.py` + `inference.py`); the rest of the args
are declared by the impl itself via `add_args`. Run any subcommand
with `--help` to see the impl-specific flags.

```bash
# Generate training documents from input structures:
uv run marinfold-document-structure generate \
    ../experiments/exp<N>_document_structures_<name>/ \
    --input /path/to/cifs/ \
    --num-docs 100 \
    --out /tmp/sample-docs.parquet

# Run a trained model on inputs (no ground truth needed):
uv run marinfold-document-structure infer \
    ../experiments/exp<N>_document_structures_<name>/ \
    --model open-athena/<model-name> \
    --input /path/to/seqs/ \
    --out /tmp/predictions.parquet

# Run a trained model + score against ground truth:
uv run marinfold-document-structure evaluate \
    ../experiments/exp<N>_document_structures_<name>/ \
    --model open-athena/<model-name> \
    --input /path/to/gt-structures/ \
    --out /tmp/metrics.json

# Build / save / push the tokenizer implied by tokens():
uv run marinfold-document-structure tokenizer \
    ../experiments/exp<N>_document_structures_<name>/ \
    --save-local /tmp/my-tokenizer          # local copy for inspection
uv run marinfold-document-structure tokenizer \
    ../experiments/exp<N>_document_structures_<name>/ \
    --push open-athena/my-structure-tokenizer
```

The CLI loads the impl by file path
(`importlib.util.spec_from_file_location`), so experiment dirs
don't need to be `pip install`ed — but they DO need to have their
own deps installed in the active venv (the impl is imported in-
process). The loader puts the impl dir on `sys.path` so private
sibling modules (`_vocab.py`, `_parse.py`, …) resolve cleanly.

Production data-gen and evals go through `data/` and `evals/`
respectively — this CLI is local-testing-only.

## Authoring a document structure

```
experiments/exp<N>_document_structures_<name>/
├── _vocab.py        # canonical token list + NAME + CONTEXT_LENGTH
├── _parse.py        # input parsing (gemmi, biopython, …)
├── generate.py      # Generator + get_generator()
├── inference.py     # Inference + get_inference()
└── pyproject.toml
```

`generate.py`:

```python
import argparse
from collections.abc import Iterator
from pathlib import Path

from _vocab import NAME, CONTEXT_LENGTH, all_domain_tokens


class MyGenerator:
    name = NAME
    context_length = CONTEXT_LENGTH

    def tokens(self) -> list[str]:
        return all_domain_tokens()

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--input", type=Path, required=True)
        parser.add_argument("--num-docs", type=int, default=None)
        # ... any generator-specific knobs

    def run(self, args: argparse.Namespace) -> Iterator[str]:
        ...


def get_generator() -> MyGenerator:
    return MyGenerator()
```

`inference.py`:

```python
import argparse
from collections.abc import Iterator
from pathlib import Path

from marinfold_document_structures import EvalResult

from _vocab import NAME, CONTEXT_LENGTH, all_domain_tokens


class MyInference:
    name = NAME
    context_length = CONTEXT_LENGTH

    def tokens(self) -> list[str]:
        return all_domain_tokens()

    def add_args(self, parser: argparse.ArgumentParser, *, subcommand: str) -> None:
        # Common flags
        parser.add_argument("--model", required=True)
        parser.add_argument("--input", type=Path, required=True)
        # ... model knobs

        if subcommand == "infer":
            parser.add_argument("--keep-bin-probs", action="store_true")
        # 'evaluate' might add --ground-truth or other comparison flags

    def predict(self, args: argparse.Namespace) -> Iterator[dict]:
        # Yield one record per input. Used by the `infer` CLI.
        ...

    def evaluate(self, args: argparse.Namespace) -> EvalResult:
        # Compute metrics against ground truth. Used by the `evaluate` CLI.
        # Typically calls predict() internally and adds scoring.
        ...


def get_inference() -> MyInference:
    return MyInference()
```

Both `Generator` and `Inference` are `@runtime_checkable` Protocols,
so a shape mismatch fails at load time with a clear message. The
tokenizer is whatever
[`build_tokenizer(component)`](marinfold_document_structures/interface.py)
constructs from `tokens()` — WordLevel + whitespace-split, with
`<pad>` at id 0 and `<eos>` at id 1. Works on either a `Generator`
or an `Inference` (both expose `tokens()`).

## See also

- [`../data/`](../data/) — production data-gen wrappers built on top
  of document structures.
- [`../evals/`](../evals/) — production eval wrappers built on top
  of document structures.
