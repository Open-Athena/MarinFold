# document_structures/

Graduated document-structure implementations.

A **document structure** is a protein-document format. Each format
starts life as an in-flight experiment under
`experiments/exp<N>_document_structures_<name>/` and, once results
are validated, **graduates** into this directory as a proper Python
package (`document_structures/<name>/`). The original experiment
dir stays frozen as the historical record.

The shared toolkit every impl pulls from — `EvalResult`,
`build_tokenizer`, the output writers — lives in
[`../marinfold/`](../marinfold/) as `marinfold.document_structures`.
This directory holds only the graduated impl packages themselves.

## Layout

```
document_structures/
├── README.md
├── AGENTS.md
└── <name>/                       # one per graduated impl
    ├── pyproject.toml
    ├── README.md
    ├── src/
    │   └── <name>/               # the actual Python package
    │       ├── __init__.py
    │       ├── cli.py
    │       ├── vocab.py
    │       ├── parse.py
    │       ├── generate.py
    │       └── inference.py
    └── tests/
```

Graduated impls are proper installable packages and import as
`from <name> import inference`. The top-level
[`marinfold`](../marinfold/) package declares each graduated impl
as a path dep so the `marinfold` CLI can dispatch to it by name.

## Authoring a document structure

Start in `experiments/exp<N>_document_structures_<name>/` with a
flat layout (no `__init__.py`, sibling imports) — see
[`experiments/exp1_document_structures_contacts_and_distances_v1/`](../experiments/exp1_document_structures_contacts_and_distances_v1/)
as the reference. Each impl exposes:

- `vocab.py`        — ordered token list, `NAME`, `CONTEXT_LENGTH`.
- `parse.py`        — input parsing (gemmi, biopython, …).
- `generate.py`     — `generate_documents(...) -> Iterator[str]`.
- `inference.py`    — `predict(cfg)` and `evaluate(cfg)` plus an
                      `InferenceConfig` dataclass.
- `cli.py`          — argparse driver (`generate` / `infer` /
                      `evaluate` / `tokenizer`).

When graduating, copy the impl into `document_structures/<name>/`,
add a `pyproject.toml` + `src/<name>/__init__.py`, and convert
sibling imports to intra-package relative imports
(`from .vocab import ...`). The per-impl `cli.py` still works for
direct invocation; the top-level `marinfold infer` / `marinfold
evaluate` CLI gives a higher-level interface that picks the
document structure from `MODELS.yaml`.

## See also

- [`../marinfold/`](../marinfold/) — shared toolkit + backends + CLI.
- [`../experiments/`](../experiments/) — in-flight (pre-graduation)
  impls.
- [`../data/`](../data/) — production data-gen wrappers built on top
  of document structures.
- [`../evals/`](../evals/) — production eval wrappers built on top
  of document structures.
