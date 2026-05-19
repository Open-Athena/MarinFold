# document_structures/AGENTS.md

Rules for agents working under `document_structures/`. Layered on top
of the root `AGENTS.md`.

## Scope

`document_structures/` holds **graduated document-structure
implementations**. Each impl is a proper Python package under
`document_structures/<name>/` with its own `pyproject.toml`,
`src/<name>/` package dir, and tests.

The **shared toolkit** every impl pulls from — `EvalResult`,
`build_tokenizer`, and the output writers (`write_docs` /
`write_predictions` / `write_eval`) — does not live here. It lives in
[`../marinfold/`](../marinfold/) as `marinfold.document_structures`,
and impls import it as `from marinfold.document_structures import …`
(or via the top-level re-exports: `from marinfold import EvalResult`,
etc.).

Each impl typically has:

- `vocab.py` — ordered token list + `NAME` / `CONTEXT_LENGTH`.
- `parse.py` — input parsing (gemmi / biopython / …).
- `generate.py` — training-document generation as a plain function
  (e.g. `generate_documents(...) -> Iterator[str]`).
- `inference.py` — `predict(cfg)` and `evaluate(cfg)` as plain
  functions; takes an `InferenceConfig` dataclass.
- `cli.py` — argparse driver (`generate` / `infer` / `evaluate` /
  `tokenizer`). Still useful for direct invocation; the high-level
  `marinfold infer` / `marinfold evaluate` CLI gives a friendlier
  interface for the common cases.

## Hard rules

1. **Don't reintroduce a centralized CLI dispatcher inside this
   directory.** A prior design used `Generator` / `Inference`
   Protocols + dynamic module loading + a
   `marinfold-document-structure` entry point. It accumulated subtle
   bugs (sys.modules cache leaks, two-pass argparse weirdness,
   silently-different output schemas across formats) for very little
   ergonomic gain.

   The top-level `marinfold` CLI (in [`../marinfold/`](../marinfold/))
   imports impls as **proper packages** declared as path deps — no
   dynamic loader, no entry-point registry. Adding a new graduated
   impl means adding it to `marinfold`'s path deps once.

2. **Output writers go through `marinfold.document_structures`.** The
   three standard shapes (docs, predictions, eval) all use the
   `write_*` helpers. They handle the parquet vs. jsonl branching,
   the `structure` column, the eval extras flattening, and the
   per-example sidecars consistently — don't reimplement these in
   any impl's `cli.py`.

3. **Per-impl `cli.py` is local-testing-only.** Don't add production-
   scaling features (iris launch, GCS upload, multi-shard
   parallelism) to it. Those belong in `data/` and `evals/`.

## Graduating an experiment

When a document-structure experiment is graduated, copy the impl into
`document_structures/<name>/` and restructure it as a proper package:

```
document_structures/<name>/
├── pyproject.toml
├── README.md
├── src/
│   └── <name>/
│       ├── __init__.py
│       ├── cli.py
│       ├── vocab.py
│       ├── parse.py
│       ├── generate.py
│       └── inference.py
└── tests/
```

Inside the package, convert sibling imports (`from vocab import …`)
to intra-package relative imports (`from .vocab import …`).

Then add the impl as a path dep in
[`../marinfold/pyproject.toml`](../marinfold/pyproject.toml) so the
top-level `marinfold` CLI can dispatch to it. The original
`experiments/exp<N>_document_structures_<name>/` stays frozen as the
historical record.
