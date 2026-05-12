# experiments/AGENTS.md

Rules for agents working under `experiments/`. Layered on top of the
root `AGENTS.md`.

## Scope

Each experiment is a directory at
`experiments/exp<N>_<kind>_<slug>/`. `<N>` is the GitHub issue number,
`<kind>` is one of `models` / `evals` / `data` / `document_structures`,
and `<slug>` is a snake_case descriptor (5–6 words max).

The `README.md` is prose only — question, hypothesis, approach,
results, plots, conclusion. **It is not an executable notebook.**
Launchable code lives as `.py` files in the same directory; large
artifacts go to GCS / HuggingFace, never into git.

If you need to compute something inline (data wrangling, a quick
plot), put it in a `.py` script in the experiment dir. Don't reach
for jupyter or jupytext.

## Hard rules

1. **One issue, one experiment dir, one kind.** Filename convention is
   `exp<N>_<kind>_<slug>`. Don't create a second directory for the
   same issue. If the slug needs to change, rename in place and
   update the README frontmatter (including `kind` if it changed).

2. **Don't commit ad-hoc binaries.** Commit small CSVs to `data/` and
   plots to `plots/`. Large model weights, intermediate parquets,
   prediction dumps, etc. go to GCS (`gs://marin-us-east5/...`) or
   HuggingFace (`buckets/timodonnell/MarinFold` for checkpoints,
   `datasets/timodonnell/<name>` for datasets). See the root
   `README.md` for the policy.

3. **Cite your runs.** When you reference a training or eval run in
   the README, include the W&B run name, the GCS output path, or the
   git SHA — whatever is enough for a future reader to find the
   underlying artifacts.

4. **Keep the frontmatter accurate.** `issue`, `title`, `kind`, and
   `branch` in `marinfold_experiment:` are read by `marinfold itemize`
   to regenerate the index.

5. **Use `gh issue comment --edit-last` for progress updates.** Don't
   spam the issue with new comments per progress check. Agent
   comments start with 🤖. Agent-opened PRs/issues carry the
   `agent-generated` label.

6. **Library code stays in kind dirs, not experiments.** If an
   experiment's helper function would clearly be reused by future
   experiments, *that's a sign it should move to the kind library
   eventually* — but only after a second use case actually exists.
   Don't pre-emptively promote.

7. **Every W&B-logged run gets a `history/runs/*.md`.** After
   `wandb.init()` returns and you have the URL, run
   `marinfold history new --wandb-url … --wandb-name … --experiment
   <this-experiment-dir-name> --kind … --short "…"`. Append iris job
   IDs on restart with `marinfold history add-iris-job …`. The
   uniqueness key is the W&B `run_id`, not the filename — multiple
   things going to the same W&B run share one history file. See the
   root `AGENTS.md` and `history/README.md` for the schema.

## Importing from kind libraries

An experiment that needs marin or a kind library has its own
`pyproject.toml` with a path dep:

```toml
[tool.uv.sources]
marinfold-models = { path = "../../models" }
```

Then the experiment can `from marinfold_models.defaults import default_train`.
See `exp1_models_protein_docs_initial_port/pyproject.toml` for the
worked example (full marin wheel pins + the kind-library path dep).

Pure-analysis experiments (no marin imports) don't need a pyproject
at all — they can just sit as `.py` files in a dir and run with the
user's system python.

## Graduating an experiment

Once results are important / high-quality enough to make this
experiment a first-class artifact, run:

```bash
uv run marinfold graduate exp<N>_<kind>_<slug>
```

This symlinks the experiment dir into the corresponding kind dir
under a name that drops the `exp<N>_<kind>_` prefix (override with
`--slug`). The experiment itself stays put.

The symlink lets the kind dir present a clean view of "important
work" without losing the historical record in `experiments/`. Don't
edit through the symlink — always edit at `experiments/exp<N>_...`.

## When a researcher replies with a variant

- Small tweak (kwarg, new sub-run): edit the experiment in place,
  re-run, update the README's Results section.
- Different hypothesis: open a new issue that links back; don't
  inflate the original.

## Closing out

Before the issue is closed:

- The `## Conclusion` section answers the question directly. A
  reader should be able to get the answer without scrolling through
  the approach or results in detail.
- Every plot in the README has its source data committed under
  `data/` (small CSVs) so the plot can be regenerated without
  rerunning the underlying pipeline.
- The issue title and the README title match.
