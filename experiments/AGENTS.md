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
   prediction dumps, etc. go to GCS (the region-co-located
   `gs://marin-<region>/protein-structure/MarinFold/...` — write to
   the bucket matching your job's compute zone; see the root
   `AGENTS.md` "GCS bucket" rule) or HuggingFace
   (`buckets/open-athena/MarinFold` for checkpoints,
   `datasets/timodonnell/<name>` for datasets). See the root
   `README.md` for the policy.

3. **Cite your runs.** When you reference a training or eval run in
   the README, include the W&B run name, the GCS output path, or the
   git SHA — whatever is enough for a future reader to find the
   underlying artifacts.

4. **Keep the frontmatter accurate.** `issue`, `title`, `kind`, and
   `branch` in `marinfold_experiment:` are read by
   `python scripts/itemize.py` to regenerate the index.

5. **Post new issue comments for progress updates.** Don't edit an
   existing issue comment unless the user explicitly asks for that.
   Avoid per-poll noise by batching routine status into meaningful
   milestone updates. Agent comments start with 🤖. Agent-opened
   PRs/issues carry the `agent-generated` label.

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

## Summary slides — `plots/summary.pdf`

Maintain a `plots/summary.pdf` while the experiment is in flight.
Regenerate it interactively as plots and findings accumulate — it's a
living deliverable, not an end-of-experiment one-shot. Treat it as
the readable face of the experiment: README is the canonical prose
record, `summary.pdf` is what you'd actually show someone.

Structure:

1. **Narrative section (first).** What we're doing, why, and the
   current state of the results. A reader who opens only the PDF
   should walk away knowing the question and where it stands.
2. **Plot appendix (last).** Each misc plot from the experiment, with
   a short caption: what it shows, and how it was generated.

For every embedded plot, print the **generating script name and its
full argument list** in small text on the same page (footer or
caption is fine). The point is that anyone reading the PDF can rerun
or edit the plot without spelunking the repo.

Individual plot files still live alongside under `plots/` —
`summary.pdf` aggregates them, it doesn't replace them.

Regeneration must be fast: a single script invocation that assembles
the PDF from the existing per-plot PNGs and narrative text (e.g.
matplotlib's `PdfPages`, or ReportLab). Don't rerun expensive
analysis at PDF-build time — the build script consumes already-saved
plots and CSVs.

### Implementation

`scripts/scaffold.py` drops two files into every new experiment:

- **`build_summary.py`** — the renderer. Copied verbatim from
  [`scripts/templates/build_summary.py`](../scripts/templates/build_summary.py);
  run it with `python build_summary.py` (or `uv run python …`) from
  the experiment dir. Reads narrative + plots, writes
  `plots/summary.pdf`. Self-contained; depends only on matplotlib.
- **`summary_narrative.md`** — the narrative source. One `## `
  heading per slide; body text under it becomes the slide. The
  agent edits this through the experiment.

Plot metadata lives in sidecar files **`plots/<plot>.<ext>.meta.json`**
written by the plotting script via the `save_plot_with_meta(...)`
helper exported from `build_summary.py`:

```python
from build_summary import save_plot_with_meta

save_plot_with_meta(
    fig,
    "plots/my_plot.png",
    caption="MAE vs k for ten proteins; lower is better.",
)
# default: script = sys.argv[0], args = sys.argv[1:]
```

The sidecar holds `{"script": ..., "args": [...], "caption": ...}`.
`build_summary.py` reads it to print the script + invocation in
small text on each plot's slide. Plots without a sidecar still
appear in the PDF, with a placeholder caption nudging you to wire
the helper in.

Commit the `*.meta.json` sidecars alongside the PNGs — they're
small (a few hundred bytes) and essential for the PDF build.

When backfilling an existing plotting script: replace
`fig.savefig(path)` with `save_plot_with_meta(fig, path, caption=...)`.
For a one-off plot you don't want to re-run, drop a hand-written
`plots/<plot>.<ext>.meta.json` next to the PNG with the same schema.

Posting the link to `summary.pdf` on the experiment issue is a
**user-triggered** step. When the user says the experiment is done,
add a comment on the issue with a link to the PDF (raw GitHub URL or
HuggingFace if it's too big to commit). Don't do this on your own
initiative — see rule #6.

## Writing a data-generation pipeline (Zephyr / Iris)

For any `exp<N>_data_*/` that runs a `map_shard` pipeline on the marin
Iris cluster, read the
[`zephyr-pipeline-performance`](../.agents/skills/zephyr-pipeline-performance/SKILL.md)
skill **before** drafting `cli.py`. It covers the six decisions that
dominate Zephyr pipeline correctness and wall-clock: source data via
the manifest's `gcs_uri` pointer (not bulky inline cif), thread the
per-row fetch, pin the region, memoize per-worker init, use the
shared gzip-safe reader, and default to fail-loud on data-quality
failures (a silently-dropped row corrupts the training corpus
weeks-of-debugging downstream). Each decision is paired with the
code pattern that implements it.

The skill exists because the same handful of mistakes keep eating
hours-to-days of cluster time on each new data experiment. The cost of
the 10 minutes it takes to read is much less than the cost of the next
overnight run that fails on a known issue. **Both `exp5` (8 min for
1.6 M structures) and `exp53` (31 min for 4.2 M)** are the reference
implementations cited throughout.

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
- `plots/summary.pdf` is up to date and its build script runs
  cleanly from the committed `data/` + `plots/` inputs.
- The issue title and the README title match.

When the user gives the go-ahead, post a comment on the experiment
issue with a link to `plots/summary.pdf` (use a new issue comment per
rule #5). Don't close the issue — that's a human action (rule #6).
