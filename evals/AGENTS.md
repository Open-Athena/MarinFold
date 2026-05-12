# evals/AGENTS.md

Rules for agents working under `evals/`. Layered on top of the root
`AGENTS.md`.

## Scope

`evals/` is the **library** for eval experiments — shared
production-wrapper code (iris job submission, result materialization,
summary publishing) that eval experiments import. Specific eval
pipelines live as experiments under `experiments/exp<N>_evals_<slug>/`.

Graduated experiments may appear here as symlinks (e.g.
`evals/foldbench/` → `../experiments/exp<N>_evals_foldbench/`). Those
are not part of the library; the library is `marinfold_evals/` only.

## Hard rules

1. **No experiment-specific code.** Anything that's specific to one
   eval (one model, one dataset, one threshold) belongs in the
   experiment, not the library.

2. **W&B project is `MarinFold`** when an eval emits W&B metrics.

3. **Eval outputs go to GCS / HF, not the repo.** Summary CSVs that
   feed a plot in the experiment README are fine to commit (small).
   Full per-example score JSONs, predicted structures, etc. live on
   GCS (`gs://marin-us-east5/eval/...`) or HF (under
   `buckets/timodonnell/MarinFold`).

## Graduated symlinks

When an eval experiment is graduated, it gets a symlink here named
after the experiment's slug (dropping the `exp<N>_evals_` prefix).
Don't edit through the symlink — edit at
`experiments/exp<N>_evals_<slug>/`.
