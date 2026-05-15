# document_structures/AGENTS.md

Rules for agents working under `document_structures/`. Layered on top
of the root `AGENTS.md`.

## Scope

`document_structures/` is a **small shared toolkit**: the `EvalResult`
dataclass, `build_tokenizer`, and the parquet/jsonl output writers
(`write_docs` / `write_predictions` / `write_eval`). That's it.

Concrete implementations live under
`experiments/exp<N>_document_structures_<name>/` and own their full
CLI surface. Each impl typically has:

- `vocab.py` — ordered token list + `NAME` / `CONTEXT_LENGTH`.
- `parse.py` — input parsing (gemmi / biopython / …).
- `generate.py` — training-document generation as a plain function
  (e.g. `generate_documents(...) -> Iterator[str]`).
- `inference.py` — `predict(cfg) -> Iterator[dict]` and
  `evaluate(cfg) -> EvalResult` as plain functions; takes a config
  dataclass the CLI builds.
- `cli.py` — argparse driver with `generate` / `infer` / `evaluate` /
  `tokenizer` subcommands. Run as `python cli.py <subcommand> ...`.

Graduated experiments may appear here as symlinks. Those are not
part of the library; the library is `marinfold_document_structures/`
only.

## Hard rules

1. **Keep the library lightweight.** `marinfold_document_structures/`
   should depend only on stdlib + pyarrow + tokenizers/transformers
   (for `build_tokenizer`). Heavy deps (gemmi, biopython, vllm,
   torch, …) belong in the impls, not here. If an impl needs a new
   shared helper, prefer copying once and lifting when a second impl
   needs the same shape.

2. **Don't reintroduce a centralized CLI or Protocol layer.** A
   prior design used `Generator` / `Inference` Protocols + dynamic
   module loading + a `marinfold-document-structure` entry point.
   It accumulated subtle bugs (sys.modules cache leaks, two-pass
   argparse weirdness, silently-different output schemas across
   formats) for very little ergonomic gain. Each impl ships its own
   `cli.py` instead.

3. **Output writers go through the library.** The three standard
   shapes (docs, predictions, eval) all use
   `marinfold_document_structures.write_*`. They handle the parquet
   vs. jsonl branching, the `structure` column, the eval extras
   flattening, and the per-example sidecars consistently — don't
   reimplement these in `cli.py`.

4. **CLI is local-testing-only.** Don't add production-scaling
   features (iris launch, GCS upload, multi-shard parallelism) to
   any impl's `cli.py`. Those belong in `data/` and `evals/`.

## Graduated symlinks

When a document-structure experiment is graduated, it gets a symlink
here named after the experiment's name (dropping the
`exp<N>_document_structures_` prefix). Don't edit through the symlink
— edit at `experiments/exp<N>_document_structures_<name>/`.
