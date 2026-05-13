# MarinFold

Can a vanilla LLM predict protein structures if its training "documents" are structured
in the right way? MarinFold is an exploratory project to answer this question. Our models are trained using
[Marin](https://github.com/marin-community/marin) infrastructure on the TPU Research Cloud.

This is a research codebase for an ongoing project. It is an experiment in open development.
We do not currently have models that anyone should use!

## Layout

```
MarinFold/
├── RESOURCES.md            # datasets, tokenizers, W&B projects, prior repos
├── AGENTS.md               # shared agent rules
├── .github/ISSUE_TEMPLATE/experiment.md
├── experiments/            # one dir per GitHub issue tagged `experiment`
│   ├── marinfold_experiments/      # PM tooling (scaffold, itemize, graduate, history)
│   ├── pyproject.toml
│   ├── README.md
│   ├── AGENTS.md
│   ├── TEMPLATE.md
│   └── exp<N>_<kind>_<name>/       # individual experiments
├── models/                 # library for model-training experiments
├── evals/                  # library for eval experiments
├── data/                   # library for data-generation experiments
├── document_structures/    # interface + local CLI for document-structure impls
└── history/                # one file per W&B-logged run + summary RUNS.md
```

Each top-level dir under the repo root is a **small library** for one
kind of work. Concrete experimental work begins as an issue and a
sub-directory under `experiments/` and may pull in helpers from the
relevant library. Important / high-quality experiments get
"graduated" — symlinked into the kind dir.

## Experiment kinds

Every experiment is one of four kinds, indicated by the second token
in its directory name (`exp10_<kind>_<name>`):

| Kind | What it does | Library lives in |
|---|---|---|
| `models` | Train models | [`models/`](models/) |
| `evals` | Run evals on trained models | [`evals/`](evals/) |
| `data` | Generate training / eval datasets | [`data/`](data/) |
| `document_structures` | Define a generate-from-input + evaluate-against-ground-truth interface for one protein-document format | [`document_structures/`](document_structures/) |

A **document structure** is a recipe with two responsibilities: turn
input data (e.g. a PDB) into a training document string, and score a
trained model against ground-truth structures using the same format.
The interface is defined in
[`document_structures/marinfold_document_structures/interface.py`](document_structures/marinfold_document_structures/interface.py),
and the `marinfold-document-structure` CLI in that subproject can
generate or evaluate against a local implementation file (no
`pip install` required) — see
[`document_structures/README.md`](document_structures/README.md).

## Experiment workflow

1. **File an issue** with the `experiment` label using the
   [issue template](.github/ISSUE_TEMPLATE/experiment.md). Specify
   the `Kind:` in the issue body.
2. **Scaffold** the experiment dir:
   ```bash
   cd experiments
   uv sync                                                          # one-time setup
   uv run marinfold scaffold --issue <N> --kind <kind>
   ```
   Creates `experiments/exp<N>_<kind>_<name>/` with a README
   pre-filled from the issue body.
3. **Implement.** Add `.py` files in the experiment dir. If the
   experiment imports marin, add a `pyproject.toml` declaring a path
   dep on the relevant kind library; see
   [`exp1_models_protein_docs_initial_port/pyproject.toml`](experiments/exp1_models_protein_docs_initial_port/pyproject.toml)
   as the worked example.
4. **Launch.** Marin's executor hash-caches step outputs, so a rerun
   with no config changes is a no-op:
   ```bash
   cd experiments/exp<N>_<kind>_<name>
   uv sync
   uv run iris --config=... -- python -m <script>
   ```
5. **Record results** in the experiment's README. Commit small CSVs
   to its `data/`, plots to its `plots/`. Large artifacts go to GCS
   or HuggingFace (see below).
6. **Regenerate the index**: `uv run marinfold itemize`.
7. **Close the issue** once the conclusion lands.

Most work happens on `main`. Use a branch (`exp/<N>-<name>`) only
when an experiment needs speculative changes to a shared kind
library.

## Graduating an experiment

When an experiment's results are important / high-quality enough to
become a first-class object in the repo, **graduate** it:

```bash
uv run marinfold graduate exp<N>_<kind>_<name>
```

This creates a symlink under the relevant top-level kind dir,
named with the experiment's name only (dropping the `exp<N>_<kind>_`
prefix). The experiment dir itself stays put — graduation is
non-destructive and the historical record in `experiments/` is kept
forever. The README's `marinfold_experiment.issue` frontmatter still
links back to the original issue.

Example: graduating `exp42_models_protein_1b_distance_masked` creates
`models/protein_1b_distance_masked/ → ../experiments/exp42_models_protein_1b_distance_masked/`.

## Run history

Every W&B-logged run gets a markdown file under `history/runs/`.
A **run** is anything with a W&B link — training, evals, data-gen
pipelines that emit metrics. Multiple processes contributing to the
same W&B `run_id` share one history file.

Each file has YAML frontmatter (user, launch time, W&B URL, iris
job IDs, git SHA, kind, experiment, short description) plus a
free-form body for the detailed plan, changes from prior runs, and
notes. `history/RUNS.md` is a generated summary table sorted newest-
first with links out to W&B + the detail file.

After `wandb.init()` returns and you have the W&B URL in hand:

```bash
marinfold history new \
    --wandb-url https://wandb.ai/open-athena/MarinFold/runs/<id> \
    --wandb-name <display-name> \
    --experiment exp<N>_<kind>_<name>   # or no_experiment
    --kind <models|evals|data|document_structures|other> \
    --short "<one-line description>" \
    --iris-jobs <iris-job-id>

marinfold history add-iris-job <run-stem> <new-iris-job-id>   # on preempt-restart
marinfold history update-index                                # regenerate RUNS.md
marinfold history sync                                        # catch missed runs (needs wandb extra)
marinfold history check                                       # CI gate
```

See [`history/README.md`](history/README.md) for the full schema and
policy.

## Where artifacts go

We try hard to avoid committing large files into the repo. The
authoritative homes for non-source artifacts:

- **HuggingFace bucket** (`buckets/open-athena/MarinFold`) — single
  bucket for **both data artifacts and model checkpoints**. Inside,
  use top-level `data/...` and `checkpoints/...` prefixes so the
  distinction is explicit. Checkpoint names should embed the W&B
  run name. (See `AGENTS.md` "HF bucket" for the splitting policy.)
- **HuggingFace datasets** (`huggingface.co/datasets/timodonnell/<name>`)
  — first-class published text / tokenized corpora that levanter
  loads via `hf://datasets/` URIs. Long-tail / in-flight data
  artifacts go to the bucket instead.
- **GCS** (`gs://marin-us-east5/<...>`) — large intermediate
  artifacts produced by marin's executor (tokenized parquets,
  cached features, predictions).
- **W&B** (`https://wandb.ai/open-athena/MarinFold`) — training and
  eval metrics, run metadata.

The repo holds source, prose, small CSVs that feed plots, and
plots themselves. Anything bigger than ~1 MB needs a deliberate
reason to be checked in.

## Tooling reference

| Command | Purpose |
|---|---|
| `marinfold scaffold --issue N --kind K` | Create an experiment dir from a GitHub issue |
| `marinfold itemize` | Regenerate `experiments/index.md` |
| `marinfold graduate exp<N>_<kind>_<name>` | Symlink an experiment into its kind dir |
| `marinfold history new ...` | Create a run history file for a W&B run |
| `marinfold history add-iris-job ...` | Append an iris job ID (preemption / restart) |
| `marinfold history sync` | Pull W&B runs; skeleton-file the missing ones (needs `wandb` extra) |
| `marinfold history update-index` | Regenerate `history/RUNS.md` |
| `marinfold history check` | CI gate: exit non-zero if W&B has runs without history files |
| `marinfold-document-structure generate IMPL INPUT --out OUT.parquet` | Local doc-gen smoke test |
| `marinfold-document-structure evaluate IMPL MODEL GROUND_TRUTH --out OUT.json` | Local eval smoke test |

The `marinfold` command (and its subcommands) is installed by `uv sync` in
`experiments/` (with `uv sync --extra wandb` for `history sync` /
`history check`). `marinfold-document-structure` is installed by
`uv sync` in `document_structures/`.

## Status

Initial port (commit-level) from the
[`marin/protein-training-1b`](https://github.com/marin-community/marin/tree/protein-training-1b/experiments/protein)
branch. All training/export scripts live under
[`experiments/exp1_models_protein_docs_initial_port/`](experiments/exp1_models_protein_docs_initial_port/);
shared marin glue is in [`models/marinfold_models/`](models/marinfold_models/).
Evals, data-gen, and document_structures are scaffolded but empty —
their first experiments will land soon.
