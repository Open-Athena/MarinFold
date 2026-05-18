# MarinFold

Can a vanilla LLM predict protein structures if its training "documents" are structured
in the right way? MarinFold aims to answer this question. Our models are trained
from scratch (without natural language data) on [Marin](https://github.com/marin-community/marin) infrastructure.

This is a research codebase for an ongoing project. It is an experiment in open development.
We do not currently have models that anyone should use!

## Layout

```
MarinFold/
├── RESOURCES.md            # datasets, tokenizers, W&B projects, prior repos
├── AGENTS.md               # shared agent rules
├── .github/ISSUE_TEMPLATE/experiment.md
├── scripts/                # repo-management scripts (scaffold, itemize, history)
├── experiments/            # one dir per GitHub issue tagged `experiment`
│   ├── README.md
│   ├── AGENTS.md
│   ├── TEMPLATE.md
│   └── exp<N>_<kind>_<name>/       # individual experiments
├── models/                 # library for model-training experiments
├── evals/                  # library for eval experiments
├── data/                   # library for data-generation experiments
├── document_structures/    # shared toolkit for document-structure impls
└── history/                # one file per W&B-logged run + summary RUNS.md
```

Each top-level dir under the repo root is a **small library** for one
kind of work. Concrete experimental work begins as an issue and a
sub-directory under `experiments/` and may pull in helpers from the
relevant library. Important / high-quality experiments get
**graduated** — copied into the kind dir, where the copy keeps
evolving while the original `experiments/exp<N>_*/` stays frozen as
the historical record.

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
Each format is a self-contained experiment dir with its own `cli.py`
driver (`generate` / `infer` / `evaluate` / `tokenizer` subcommands)
on top of a small shared toolkit in
[`document_structures/`](document_structures/) (`EvalResult`,
`build_tokenizer`, parquet/jsonl writers). The reference impl is
[`experiments/exp1_document_structures_contacts_and_distances_v1/`](experiments/exp1_document_structures_contacts_and_distances_v1/).

## Experiment workflow

1. **File an issue** with the `experiment` label using the
   [issue template](.github/ISSUE_TEMPLATE/experiment.md). Specify
   the `Kind:` in the issue body.
2. **Scaffold** the experiment dir:
   ```bash
   cd scripts
   uv sync                                                          # one-time setup
   python scaffold.py --issue <N> --kind <kind>
   ```
   Creates `experiments/exp<N>_<kind>_<name>/` with a README
   pre-filled from the issue body.
3. **Implement.** Add `.py` files in the experiment dir. If the
   experiment imports marin, add a `pyproject.toml` declaring a path
   dep on the relevant kind library; see
   [`exp0_models_protein_docs_initial_port/pyproject.toml`](experiments/exp0_models_protein_docs_initial_port/pyproject.toml)
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
6. **Regenerate the index**: `python scripts/itemize.py`.
7. **Close the issue** once the conclusion lands.

Most work happens on `main`. Use a branch (`exp/<N>-<name>`) only
when an experiment needs speculative changes to a shared kind
library.

## Graduating an experiment

When an experiment's results are validated and the code should keep
evolving as a first-class object, **copy** the directory into the
matching kind dir, dropping the `exp<N>_<kind>_` prefix:

```bash
cp -r experiments/exp<N>_<kind>_<name>/ <kind>/<name>/
# e.g. cp -r experiments/exp42_models_protein_1b/ models/protein_1b/
```

The original `experiments/exp<N>_*/` directory stays **frozen** as
the historical record — the README, data, plots, and conclusion
remain as they were at the time of the experiment. The kind-dir
copy is the working version going forward; edits land there.

After the copy, trim or rewrite the experiment-style README to fit
the kind dir's docs convention, decide whether to keep or merge the
experiment's `pyproject.toml`, and update any internal links that
pointed at the experiment dir.

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
python scripts/history.py new \
    --wandb-url https://wandb.ai/open-athena/MarinFold/runs/<id> \
    --wandb-name <display-name> \
    --experiment exp<N>_<kind>_<name>   # or no_experiment
    --kind <models|evals|data|document_structures|other> \
    --short "<one-line description>" \
    --iris-jobs <iris-job-id>

python scripts/history.py add-iris-job <run-stem> <new-iris-job-id>   # on preempt-restart
python scripts/history.py update-index                                # regenerate RUNS.md
python scripts/history.py sync                                        # catch missed runs (needs wandb extra)
python scripts/history.py check                                       # CI gate
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

Repo-management scripts live in [`scripts/`](scripts/) and are run
with plain `python`:

| Script | Purpose |
|---|---|
| `python scripts/scaffold.py --issue N --kind K` | Create an experiment dir from a GitHub issue |
| `python scripts/itemize.py` | Regenerate `experiments/index.md` |
| `python scripts/history.py new ...` | Create a run history file for a W&B run |
| `python scripts/history.py add-iris-job ...` | Append an iris job ID (preemption / restart) |
| `python scripts/history.py sync` | Pull W&B runs; skeleton-file the missing ones (needs `wandb` extra) |
| `python scripts/history.py update-index` | Regenerate `history/RUNS.md` |
| `python scripts/history.py check` | CI gate: exit non-zero if W&B has runs without history files |

For document-structure CLIs, run `python cli.py {generate,infer,evaluate,tokenizer}`
from the impl's experiment directory (e.g.
`experiments/exp1_document_structures_contacts_and_distances_v1/`).

To set up the scripts venv: `cd scripts && uv venv --python 3.11 && uv sync`
(add `--extra wandb` for `history sync` / `history check`).

The `marinfold` CLI name is reserved for a future user-facing
command (running inference, etc.); it is not currently in use.

## Status

Initial port (commit-level) from the
[`marin/protein-training-1b`](https://github.com/marin-community/marin/tree/protein-training-1b/experiments/protein)
branch. All training/export scripts live under
[`experiments/exp0_models_protein_docs_initial_port/`](experiments/exp0_models_protein_docs_initial_port/);
shared marin glue is in [`models/marinfold_models/`](models/marinfold_models/).
Evals, data-gen, and document_structures are scaffolded but empty —
their first experiments will land soon.
