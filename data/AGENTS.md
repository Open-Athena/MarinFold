# data/AGENTS.md

Rules for agents working under `data/`. Layered on top of the root
`AGENTS.md`.

## Scope

`data/` is the **library** for data-generation experiments — shared
wrappers around zephyr pipelines that materialize training and eval
corpora. Specific data pipelines live as experiments under
`experiments/exp<N>_data_<slug>/`.

Graduated experiments may appear here as symlinks. Those are not
part of the library; the library is `marinfold_data/` only.

## Hard rules

1. **Large datasets do not live in git.** Outputs go to HuggingFace
   (`huggingface.co/datasets/timodonnell/<name>`) or GCS
   (`gs://marin-us-east5/<...>`). The repo holds only the pipeline
   code + small metadata / config artifacts. See the root
   `README.md` for the policy.

2. **Pin HF revisions when consuming HF datasets.** Refresh by
   running `curl -s https://huggingface.co/api/datasets/<id> | jq -r .sha`.
   Bumping the pin is a deliberate choice and busts the marin
   executor's cache.

3. **No experiment-specific code.** Anything that's specific to one
   data source / one document-type / one filter belongs in the
   experiment, not the library.

## Graduated symlinks

When a data experiment is graduated, it gets a symlink here named
after the experiment's slug (dropping the `exp<N>_data_` prefix).
Don't edit through the symlink — edit at
`experiments/exp<N>_data_<slug>/`.
