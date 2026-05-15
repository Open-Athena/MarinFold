# document_structures/AGENTS.md

Rules for agents working under `document_structures/`. Layered on top
of the root `AGENTS.md`.

## Scope

`document_structures/` holds the **interfaces** (`Generator` +
`Inference` Protocols), the `build_tokenizer` helper, and the
**local-testing CLI** (`marinfold-document-structure`). Concrete
implementations are experiments under
`experiments/exp<N>_document_structures_<name>/`, each with two
public files: `generate.py` (exports `get_generator()`) and
`inference.py` (exports `get_inference()`). Shared vocab + parsing
live in private `_vocab.py` / `_parse.py` modules in the same dir.

Graduated experiments may appear here as symlinks. Those are not
part of the library; the library is `marinfold_document_structures/`
only.

## Hard rules

1. **Both interfaces evolve carefully.** A change to `Generator`
   or `Inference` (new method, changed signature) breaks every
   existing implementation. When evolving, update all known impls
   in the same PR — don't ship a Protocol that no current impl
   satisfies.

2. **The two files MUST agree on vocab.** `generate.py`'s
   `tokens()` and `inference.py`'s `tokens()` produce the same
   list (typically by importing a shared `_vocab.all_domain_tokens()`).
   A mismatch silently makes the tokenizers incompatible — tests
   that pin `gen.tokens() == inf.tokens()` are mandatory for any
   new impl.

3. **Keep the library lightweight.** `marinfold_document_structures/`
   should only depend on stdlib + pyarrow + tokenizers/transformers
   (for `build_tokenizer`). Heavy deps (gemmi, biopython, vllm,
   torch, …) belong in the impls, not here.

4. **CLI is local-testing-only.** Don't add production-scaling
   features (iris launch, GCS upload, multi-shard parallelism).
   That belongs in `data/` and `evals/`. The CLI's job is "let a
   researcher confirm the impl works on 100 inputs before scaling".

5. **The CLI owns `<impl_dir>` and `--out`.** Everything else is
   the impl's responsibility — register flags via
   `add_args(parser)` (Generator) or
   `add_args(parser, subcommand=)` (Inference). Don't add
   "standard" generation / inference flags at the CLI level; that
   constrains impls.

6. **Load by path, not by import name.** The CLI uses
   `importlib.util.spec_from_file_location` and puts the impl dir
   on `sys.path` so private siblings (`_vocab.py`, …) resolve.
   Don't introduce a "registry" that requires impls to declare
   themselves.

## Graduated symlinks

When a document-structure experiment is graduated, it gets a symlink
here named after the experiment's name (dropping the
`exp<N>_document_structures_` prefix). Don't edit through the symlink
— edit at `experiments/exp<N>_document_structures_<name>/`.
