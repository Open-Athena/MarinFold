# document_structures/AGENTS.md

Rules for agents working under `document_structures/`. Layered on top
of the root `AGENTS.md`.

## Scope

`document_structures/` holds the **interface** (`DocumentStructure`
Protocol) and the **local-testing CLI**
(`marinfold-document-structure`). Concrete document-structure
implementations are experiments under
`experiments/exp<N>_document_structures_<slug>/`.

Graduated experiments may appear here as symlinks. Those are not
part of the library; the library is `marinfold_document_structures/`
only.

## Hard rules

1. **The interface evolves carefully.** A change to
   `DocumentStructure` (new method, changed signature) breaks every
   existing implementation. When evolving, update all known impls
   in the same PR — don't ship a Protocol that no current impl
   satisfies.

2. **Keep the library lightweight.** `marinfold_document_structures/`
   should only depend on stdlib + pyarrow (for `--out parquet`).
   Heavy deps (transformers, biopython, jax, …) belong in the impls,
   not here.

3. **CLI is local-testing-only.** Don't add production-scaling
   features (iris launch, GCS upload, multi-shard parallelism). That
   belongs in `data/` and `evals/`. The CLI's job is "let a
   researcher confirm the impl works on 100 docs before scaling".

4. **Load by path, not by import name.** The CLI uses
   `importlib.util.spec_from_file_location` so experiment dirs don't
   need to be installed packages. Keep it that way; don't introduce
   a "registry" that requires impls to declare themselves.

## Graduated symlinks

When a document-structure experiment is graduated, it gets a symlink
here named after the experiment's slug (dropping the
`exp<N>_document_structures_` prefix). Don't edit through the symlink
— edit at `experiments/exp<N>_document_structures_<slug>/`.
