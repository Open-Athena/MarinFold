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

| Kind | Purpose | Shared library it imports from |
|---|---|---|
| `models` | Train models | [`../models/`](../models/) |
| `evals` | Run evals on trained models | — (no shared library yet) |
| `data` | Generate training/eval datasets | — (no shared library yet) |
| `document_structures` | Define an interface for both generating documents from input data and evaluating models | [`../marinfold/marinfold/document_structures/`](../marinfold/marinfold/document_structures/) |

An experiment dir stays under `experiments/` for its whole life — it
is never copied or promoted into a kind dir. Reusable code goes in the
kind library from the start and the experiment imports it. A document
structure in particular is always implemented in
[`marinfold.document_structures`](../marinfold/marinfold/document_structures/);
see [`../marinfold/README.md`](../marinfold/README.md).

## Tooling

Two repo-management scripts live in [`../scripts/`](../scripts/):

```bash
cd scripts
uv venv --python 3.11
uv sync

# create an experiment dir from a GitHub issue:
python scaffold.py --issue 42 --kind models
# regenerate ../experiments/index.md from gh + frontmatter:
python itemize.py
```

The scaffolder pulls Question / Hypothesis / Background / Approach /
Success criteria sections from the GitHub issue body. It picks
`--kind` from the issue's `kind/<kind>` label (then a `Kind:` line in
the issue body) if not passed explicitly.

`itemize.py` groups the index by kind. A dir-backed experiment's kind
comes from its dir; a dir-less issue's kind comes from its
`kind/<kind>` label — so **every experiment issue should carry a
`kind/<kind>` label** (`kind/models`, `kind/evals`, `kind/data`, or
`kind/document_structures`).

## Flow

1. **File an issue** with the `experiment` label using the
   [issue template](../.github/ISSUE_TEMPLATE/experiment.md). Fill in
   question, hypothesis, approach, success criteria, **and the kind** —
   add the matching `kind/<kind>` label (the index groups by it).
2. **Scaffold** the experiment dir:
   ```bash
   python scripts/scaffold.py --issue <N> --kind <models|evals|data|document_structures>
   ```
   Creates `experiments/exp<N>_<kind>_<name>/` with a README pre-filled
   from the issue body.
3. **Implement**. Add `.py` files in the experiment dir; if the
   experiment touches marin, add a `pyproject.toml` declaring a path
   dep on the relevant kind library (e.g. `marinfold-models`). See
   [`exp0_models_protein_docs_initial_port/pyproject.toml`](exp0_models_protein_docs_initial_port/pyproject.toml)
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
   the root `README.md` for the policy. Keep `plots/summary.pdf`
   (narrative + plot appendix) updated as you go — see
   [`AGENTS.md`](AGENTS.md).
6. **Regenerate the index**:
   ```bash
   python scripts/itemize.py
   ```
7. **Close** the issue once the conclusion lands.

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
  plots/summary.pdf  # living presentation slides — narrative + plot appendix.
                     # See experiments/AGENTS.md for the contract.
  *.py               # launchable scripts + analysis
```

## Frontmatter

The `README.md` starts with a YAML frontmatter block read by
`python scripts/itemize.py`:

```yaml
---
marinfold_experiment:
  issue: 42
  title: "Distance-masked vs unmasked at 100M scale"
  kind: models
  branch: main
---
```

Keep `issue`, `title`, `kind`, and `branch` accurate.

## See also

- [Open-Athena/helico](https://github.com/Open-Athena/helico) — a
  richer experiment system we drew inspiration from but deliberately
  kept ours simpler than (no jupytext, no Modal-specific wrappers, no
  cost gates).
