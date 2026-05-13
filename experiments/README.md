# experiments/

One directory per experiment, keyed by GitHub issue number AND tagged
with its **kind**. Each directory's `README.md` is the prose record —
question, hypothesis, approach, results, conclusion.

Directory naming: `experiments/exp<N>_<kind>_<name>/` where:

- `<N>` **must equal the GitHub issue number.** One issue ⇄ one
  experiment dir; the frontmatter's `issue:` field has to match.
- `<kind>` is one of `models`, `evals`, `data`, `document_structures`.
- `<name>` is a snake_case descriptor (5–6 words max).

Examples:
- `exp10_models_train_1b`
- `exp11_evals_foldbench`
- `exp12_data_esm_metagenomic_atlas`
- `exp13_document_structures_contacts_and_distances_v1`

**Sentinel:** `exp0_*` is reserved for work that predates the
experiment system (no real issue). Don't use it for new work — file
an issue first.

## What each kind means

| Kind | Purpose | Top-level dir for graduated work |
|---|---|---|
| `models` | Train models | [`../models/`](../models/) |
| `evals` | Run evals on trained models | [`../evals/`](../evals/) |
| `data` | Generate training/eval datasets | [`../data/`](../data/) |
| `document_structures` | Define an interface for both generating documents from input data and evaluating models — see [`../document_structures/`](../document_structures/) | [`../document_structures/`](../document_structures/) |

## Tooling

Three PM commands live in `marinfold_experiments`:

```bash
cd experiments
uv venv --python 3.11
uv sync

uv run marinfold scaffold --issue 42 --kind models       # create dir from issue
uv run marinfold itemize                                  # regenerate index.md
uv run marinfold graduate exp42_models_protein_1b_distance_masked
```

The scaffolder pulls Question/Hypothesis/Approach/Compute/Success/
Baselines sections from the GitHub issue body. It picks `--kind` from
a `Kind:` line in the issue body if not passed explicitly.

## Flow

1. **File an issue** with the `experiment` label using the
   [issue template](../.github/ISSUE_TEMPLATE/experiment.md). Fill in
   question, hypothesis, approach, success criteria, compute estimate,
   baselines, **and the kind**.
2. **Scaffold** the experiment dir:
   ```bash
   uv run marinfold scaffold --issue <N> --kind <models|evals|data|document_structures>
   ```
   Creates `experiments/exp<N>_<kind>_<name>/` with a README pre-filled
   from the issue body.
3. **Implement**. Add `.py` files in the experiment dir; if the
   experiment touches marin, add a `pyproject.toml` declaring a path
   dep on the relevant kind library (e.g. `marinfold-models`). See
   [`exp1_models_protein_docs_initial_port/pyproject.toml`](exp1_models_protein_docs_initial_port/pyproject.toml)
   as the worked example.
4. **Launch**. Marin's executor hash-caches step outputs:
   ```bash
   cd experiments/exp<N>_<kind>_<name>
   uv sync
   uv run iris --config=... -- python -m <script>
   ```
5. **Record results** in the experiment's README. Commit small CSVs
   to `data/`, plots to `plots/`. Large artifacts (model weights, big
   intermediate parquets, predictions) go to GCS or HuggingFace; see
   the root `README.md` for the policy.
6. **Regenerate the index**:
   ```bash
   uv run marinfold itemize
   ```
7. **Close** the issue once the conclusion lands.
8. **(Optional) Graduate.** If the experiment's results are
   important / high-quality enough to become a first-class object,
   symlink it into the kind dir:
   ```bash
   uv run marinfold graduate exp<N>_<kind>_<name>
   # → models/<name>/  (or evals/<name>/, data/<name>/, document_structures/<name>/)
   ```
   The symlink drops the `exp<N>_<kind>_` prefix. The experiment dir
   stays where it is — graduation is non-destructive. The README's
   frontmatter still records the original issue.

## Main vs. branch

By default, experiments live on `main`. Branches (`exp/<N>-<name>`)
are appropriate when an experiment needs speculative changes to the
shared kind libraries that may not ship. Set
`marinfold_experiment.branch` in the README frontmatter accordingly.

## Directory layout per experiment

```
experiments/exp<N>_<kind>_<name>/
  README.md          # prose: question, hypothesis, approach, results,
                     # conclusion. Frontmatter (issue, title, kind, branch).
  pyproject.toml     # optional; needed if the experiment imports marin or a kind lib
  data/              # small CSVs committed; the source for any plot
  plots/             # PNGs/PDFs committed; embedded in README.md
  *.py               # launchable scripts + analysis
```

## Frontmatter

The `README.md` starts with a YAML frontmatter block read by
`marinfold itemize`:

```yaml
---
marinfold_experiment:
  issue: 42
  title: "Distance-masked vs unmasked at 100M scale"
  kind: models
  branch: main
  baselines:
    - protein-contacts-1b-3.5e-4-distance-masked-7d355e
---
```

Keep `issue`, `title`, `kind`, and `branch` accurate.

## See also

- [Open-Athena/helico](https://github.com/Open-Athena/helico) — a
  richer experiment system we drew inspiration from but deliberately
  kept ours simpler than (no jupytext, no Modal-specific wrappers, no
  cost gates).
