# document_structures/

The shared interface + local-testing CLI for MarinFold document
structures.

A **document structure** is a recipe that:

1. **Generates** training documents from input data (PDB, AFDB, mmCIF,
   …) — `DocumentStructure.generate_documents`.
2. **Evaluates** trained models against ground-truth structures using
   the same format — `DocumentStructure.evaluate`.

The shape is captured by the
[`DocumentStructure`](marinfold_document_structures/interface.py)
Protocol. Each concrete structure (e.g. `contacts-and-distances-v1`)
lives as an experiment under
`experiments/exp<N>_document_structures_<slug>/`, and exposes a
`get_structure()` function returning a `DocumentStructure` instance.

## Layout

```
document_structures/
├── pyproject.toml                          # subproject pyproject; CLI entry point
├── README.md                               # this file
├── AGENTS.md                               # rules for working under document_structures/
├── marinfold_document_structures/
│   ├── __init__.py
│   ├── interface.py                        # DocumentStructure Protocol + load_structure
│   └── cli.py                              # marinfold-document-structure CLI
└── <graduated symlinks>                    # e.g. contacts_and_distances_v1/ → ../experiments/exp<N>_document_structures_<slug>/
```

## Setup

```bash
cd document_structures
uv venv --python 3.11
uv sync
```

That installs the `marinfold-document-structure` CLI command into the
local venv.

## The CLI

```bash
# Smoke-test generate from input data:
uv run marinfold-document-structure generate \
    ../experiments/exp<N>_document_structures_<slug>/structure.py \
    /path/to/input.parquet \
    --num-docs 100 \
    --context-length 8192 \
    --out /tmp/sample-docs.parquet

# Run an eval locally:
uv run marinfold-document-structure evaluate \
    ../experiments/exp<N>_document_structures_<slug>/structure.py \
    gs://marin-us-east5/checkpoints/.../hf/step-50000 \
    /path/to/ground_truth.jsonl \
    --out /tmp/eval-result.json
```

The CLI loads the impl by file path
(`importlib.util.spec_from_file_location`), so experiment dirs don't
need to be `pip install`ed — but they DO need to have their own deps
installed in the active venv, since the impl module gets imported
in-process.

Production data-gen and evals go through `data/` and `evals/`
respectively — this CLI is local-testing-only.

## Authoring a document structure

In your experiment dir (`experiments/exp<N>_document_structures_<slug>/`),
create `structure.py`:

```python
from marinfold_document_structures import DocumentStructure, EvalResult

class MyDocumentStructure:
    name = "my-structure"
    tokenizer = "timodonnell/my-tokenizer@<sha>"
    context_length = 8192

    def iter_inputs(self, path):
        # yield input records the CLI passes to generate_documents
        ...

    def iter_ground_truth(self, path):
        # yield ground-truth records the CLI passes to evaluate
        ...

    def generate_documents(self, input_records, *, context_length=None, num_docs=None):
        # yield strings
        ...

    def evaluate(self, *, model_path, ground_truth_records) -> EvalResult:
        # return EvalResult(metrics={...}, per_example=[...], extras={...})
        ...

def get_structure() -> DocumentStructure:
    return MyDocumentStructure()
```

The `DocumentStructure` Protocol is `runtime_checkable`, so
`isinstance(structure, DocumentStructure)` validates the shape at load
time without requiring you to inherit from it.

## See also

- [`../data/`](../data/) — production data-gen wrappers built on top
  of document structures.
- [`../evals/`](../evals/) — production eval wrappers built on top
  of document structures.
