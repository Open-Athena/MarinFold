# experiments/AGENTS.md

Rules for agents working under `experiments/`. Layered on top of the
root `AGENTS.md`.

## Scope

Each experiment is a directory at
`experiments/exp<N>_<kind>_<name>/`. `<N>` is the GitHub issue number,
`<kind>` is one of `models` / `evals` / `data` / `document_structures`,
and `<name>` is a snake_case descriptor (5–6 words max).

The `README.md` is prose only — question, hypothesis, approach,
results, plots, conclusion. **It is not an executable notebook.**
Launchable code lives as `.py` files (or `.ipynb` notebooks; see
below) in the same directory; large artifacts go to GCS /
HuggingFace, never into git.

### Notebooks

Jupyter notebooks (`.ipynb`) are welcome in an experiment dir when
the work is genuinely interactive — exploratory analysis,
plot-heavy eval reports, prototyping a new doc structure.

**Commit notebooks with their cell outputs.** Outputs (plots,
small tables, printed metrics) are part of the experiment record
and let a reader skim the result without re-running. Re-execute
top-to-bottom (`Restart & Run All` or `jupyter nbconvert --execute
--inplace`) before committing so the saved state matches the code.
Keep notebooks small (under a few MB); push large arrays / weights
to GCS or HuggingFace and reference them by URL from the notebook.

If the inline compute is small and one-shot (a single plot, a
quick data wrangle), a plain `.py` script is fine too — pick
whichever fits the work.

## Hard rules

1. **`<N>` is the GitHub issue number.** The dir is
   `exp<N>_<kind>_<name>` where `<N>` is the integer issue ID and
   the frontmatter's `issue:` field MUST match. One issue → one
   experiment dir → one kind. Don't create a second dir for the
   same issue. If the name needs to change, rename in place and
   update the frontmatter (including `kind` if it changed).

   The sole exception is **`exp0_*`**, reserved as a sentinel for
   work that predates the experiment system (e.g.
   `exp0_models_protein_docs_initial_port/`, the marin port). New
   work always gets a real issue first.

2. **Don't commit ad-hoc binaries.** Commit small CSVs to `data/` and
   plots to `plots/`. Large model weights, intermediate parquets,
   prediction dumps, etc. go to GCS (`gs://marin-us-east5/...`) or
   HuggingFace (`buckets/open-athena/MarinFold` for checkpoints,
   `datasets/timodonnell/<name>` for datasets). See the root
   `README.md` for the policy.

3. **Cite your runs.** When you reference a training or eval run in
   the README, include the W&B run name, the GCS output path, or the
   git SHA — whatever is enough for a future reader to find the
   underlying artifacts.

4. **Keep the frontmatter accurate.** `issue`, `title`, `kind`, and
   `branch` in `marinfold_experiment:` are read by
   `python scripts/itemize.py` to regenerate the index.

5. **Use `gh issue comment --edit-last` for progress updates.** Don't
   spam the issue with new comments per progress check. Agent
   comments start with 🤖. Agent-opened PRs/issues carry the
   `agent-generated` label.

6. **Agents never close experiment issues.** Posting results, updating
   the README's Conclusion, and marking the work done in the
   experiment dir is fine — but the final `gh issue close` is a
   human-only action. If you think an experiment is done, say so in
   a final comment on the issue and stop there; let the user decide.

7. **Library code stays in kind dirs, not experiments.** If an
   experiment's helper function would clearly be reused by future
   experiments, *that's a sign it should move to the kind library
   eventually* — but only after a second use case actually exists.
   Don't pre-emptively promote.

8. **Every W&B-logged run gets a `history/runs/*.md`.** After
   `wandb.init()` returns and you have the URL, run
   `python scripts/history.py new --wandb-url … --wandb-name …
   --experiment <this-experiment-dir-name> --kind … --short "…"`.
   Append iris job IDs on restart with
   `python scripts/history.py add-iris-job …`. The uniqueness key
   is the W&B `run_id`, not the filename — multiple things going to
   the same W&B run share one history file. See the root
   `AGENTS.md` and `history/README.md` for the schema.

9. **Capture per-input timings whenever you run a predictor.** Bake
   timing capture into the wrapper at evaluation time and commit a
   `data/timings.csv` to the experiment dir — don't plan to
   reconstruct from logs after the fact (Modal's ephemeral-app logs
   get pruned). Use the schema documented in the root `AGENTS.md`
   "Capture timings for every predictor run" section so cross-
   experiment timing comparisons (e.g. exp12 Protenix vs exp20
   MarinFold-1B) join on `(stem, n_residues)` without bespoke
   munging.

## Importing from kind libraries

An experiment that needs marin or a kind library has its own
`pyproject.toml` with a path dep:

```toml
[tool.uv.sources]
marinfold-models = { path = "../../models" }
```

Then the experiment can `from marinfold_models.defaults import default_train`.
See `exp0_models_protein_docs_initial_port/pyproject.toml` for the
worked example (full marin wheel pins + the kind-library path dep).

Pure-analysis experiments (no marin imports) don't need a pyproject
at all — they can just sit as `.py` files in a dir and run with the
user's system python.

## Graduating an experiment

Once an experiment's results are validated and the code should keep
evolving as a first-class object, **copy** the directory into the
matching kind dir, dropping the `exp<N>_<kind>_` prefix:

```bash
cp -r experiments/exp<N>_<kind>_<name>/ <kind>/<name>/
# e.g. cp -r experiments/exp42_models_protein_1b/ models/protein_1b/
```

**Leave the original `experiments/exp<N>_*/` directory untouched.**
It's the frozen historical record of what was tried at the time of
the experiment — the README, the data, the plots, the conclusion.
The kind-dir copy is the working version going forward; edits land
there, not in the experiment dir.

After the copy, you'll typically want to:

- Trim or rewrite the experiment-style README if the kind dir has a
  different docs convention (kind-dir code isn't always
  question/hypothesis/result-shaped).
- Decide whether to keep the experiment's `pyproject.toml` and venv
  or consolidate with the kind library's setup.
- Update any internal links / commit references that pointed at the
  experiment dir.

The kind dir and the experiment dir then diverge: the experiment
stays frozen as the historical snapshot, the kind dir evolves
freely.

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
