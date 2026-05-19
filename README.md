# MarinFold

Can a vanilla LLM predict protein structures if its training "documents" are structured
in the right way? MarinFold aims to answer this question. Our models are trained
from scratch (without natural language data) on [Marin](https://github.com/marin-community/marin) infrastructure.

This is a research codebase for an ongoing project. It is an experiment in open development.
We do not currently have models that anyone should use!

## Layout

```
MarinFold/
├── MODELS.yaml             # registry of trained models (nickname → HF URL)
├── RESOURCES.md            # datasets, tokenizers, W&B projects, prior repos
├── AGENTS.md               # shared agent rules
├── .github/ISSUE_TEMPLATE/experiment.md
├── scripts/                # repo-management scripts (scaffold, itemize, history)
├── experiments/            # one dir per GitHub issue tagged `experiment`
│   ├── README.md
│   ├── AGENTS.md
│   ├── TEMPLATE.md
│   └── exp<N>_<kind>_<name>/       # individual experiments
├── marinfold/              # top-level package: backends, doc-structure toolkit, graduated impls, `marinfold` CLI
├── models/                 # library for model-training experiments
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
| `evals` | Run evals on trained models | — (no shared library yet) |
| `data` | Generate training / eval datasets | — (no shared library yet) |
| `document_structures` | Define a generate-from-input + evaluate-against-ground-truth interface for one protein-document format | [`marinfold/marinfold/document_structures/`](marinfold/marinfold/document_structures/) |

Kind libraries are only created when a second experiment needs the
same helper. Today `evals/` and `data/` kinds exist as experiment
kinds (e.g. `experiments/exp9_evals_*`) but have no shared library —
the first experiment in each kind that finds itself sharing code
with a sibling creates the kind dir at that point.

A **document structure** is a recipe with two responsibilities: turn
input data (e.g. a PDB) into a training document string, and score a
trained model against ground-truth structures using the same format.
Each format is a self-contained experiment dir with its own `cli.py`
driver (`generate` / `infer` / `evaluate` / `tokenizer` subcommands)
on top of the shared toolkit in
[`marinfold.document_structures`](marinfold/marinfold/document_structures/)
(`EvalResult`, `build_tokenizer`, parquet/jsonl writers). The
reference impl is
[`experiments/exp1_document_structures_contacts_and_distances_v1/`](experiments/exp1_document_structures_contacts_and_distances_v1/);
graduated impls live as subpackages of `marinfold.document_structures`.

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

## Running inference

Trained models are listed in [`MODELS.yaml`](MODELS.yaml) by
nickname. The `marinfold` CLI looks up the model, picks the first
document structure it supports, and dispatches to the graduated
impl. Two subcommands:

```bash
cd marinfold
uv sync --extra mlx        # or --extra vllm, or --extra transformers

# Predict residue-pair distances for a sequence (no ground truth).
uv run marinfold infer \
    --backend mlx --input-sequence SIINFEKLLLSKP \
    --out /tmp/preds.json

# Evaluate predictions against ground-truth structures.
uv run marinfold evaluate \
    --backend mlx --input-dir /path/to/pdbs/ \
    --out /tmp/preds.json --metrics-out /tmp/metrics.json
```

| Backend | Platform | Extra |
|---|---|---|
| `vllm` | Linux + NVIDIA GPU (production / scaled eval) | `--extra vllm` |
| `mlx` | Apple Silicon (fastest local) | `--extra mlx` |
| `transformers` | Anywhere torch installs (Apple MPS, CPU, CUDA) | `--extra transformers` |

`--model` accepts a [`MODELS.yaml`](MODELS.yaml) nickname or a
local checkpoint directory. Omit it to use the entry marked
`default: true`. `--document-structure` overrides the impl
selection; without it the first supported impl wins. See
[`marinfold/README.md`](marinfold/README.md) for the full backend
matrix and `marinfold infer --help` / `marinfold evaluate --help`
for the full flag set.

For impl-specific flags (seed-N sweeps, distance cap, batch size,
etc.) each graduated impl ships its own lower-level CLI as a
console script:

```bash
cd marinfold
uv sync --extra mlx --extra contacts-and-distances-v1
uv run contacts-and-distances-v1 evaluate \
    --backend mlx --model 1B \
    --input /path/to/pdbs/ --seed-n-values 0,5,20,50 \
    --out /tmp/metrics.json
```

## Graduating an experiment

When an experiment's results are validated and the code should keep
evolving as a first-class object, **copy** the directory into the
matching kind dir, dropping the `exp<N>_<kind>_` prefix:

```bash
# models / evals / data: peer kind directory
cp -r experiments/exp<N>_<kind>_<name>/ <kind>/<name>/
# e.g. cp -r experiments/exp42_models_protein_1b/ models/protein_1b/

# document_structures: subpackage of marinfold
cp -r experiments/exp<N>_document_structures_<name>/ \
      marinfold/marinfold/document_structures/<name>/
```

The original `experiments/exp<N>_*/` directory stays **frozen** as
the historical record — the README, data, plots, and conclusion
remain as they were at the time of the experiment. The graduated
copy is the working version going forward; edits land there.

After the copy: convert sibling imports to intra-package relative
imports (`from .vocab import …`), add an `__init__.py` re-export
of the public surface, and — for document_structures impls — add
an optional-deps extra to `marinfold/pyproject.toml` for any
heavy parser deps (e.g. `gemmi`). Tests move next to similar
tests under each kind dir's `tests/`.

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

For impl-specific CLI surfaces (e.g. `generate` and `tokenizer`
subcommands), see the per-impl console script — graduated impls
expose one as `<structure-name>` (e.g. `contacts-and-distances-v1
{generate,infer,evaluate,tokenizer} ...`) installed alongside the
top-level `marinfold` command.

To set up the scripts venv: `cd scripts && uv venv --python 3.11 && uv sync`
(add `--extra wandb` for `history sync` / `history check`).

## Status

Initial port (commit-level) from the
[`marin/protein-training-1b`](https://github.com/marin-community/marin/tree/protein-training-1b/experiments/protein)
branch. All training/export scripts live under
[`experiments/exp0_models_protein_docs_initial_port/`](experiments/exp0_models_protein_docs_initial_port/);
shared marin glue is in [`models/marinfold_models/`](models/marinfold_models/).
The `contacts-and-distances-v1` document structure is graduated at
[`marinfold/marinfold/document_structures/contacts_and_distances_v1/`](marinfold/marinfold/document_structures/contacts_and_distances_v1/);
its in-flight history lives at
[`experiments/exp1_document_structures_contacts_and_distances_v1/`](experiments/exp1_document_structures_contacts_and_distances_v1/).
Eval experiments (e.g. `experiments/exp9_evals_*`) have started
landing; a shared `evals` kind library will be created when a second
eval experiment needs the same helper.
