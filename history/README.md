# history/

Persistent audit trail for every training / eval / data-gen run that
MarinFold launches. A **run** is anything with a W&B link — multiple
processes contributing to the same W&B run share one history file.

```
history/
├── README.md          # this file
├── RUNS.md            # generated summary table (sorted by date, newest first)
└── runs/
    └── <YYYYMMDD>_<experiment>_<wandb_run_name>.md
```

## File contents

Each `runs/*.md` has YAML frontmatter (machine-parseable) + free-form
body. The frontmatter is the source of truth for the summary table.

```yaml
---
marinfold_run:
  user: timodonnell
  launched_at: 2026-05-12T14:23:00Z
  experiment: exp13_models_train_1b   # or "no_experiment" outside any experiment
  kind: models                         # models | evals | data | document_structures | other
  short_description: "First 1B distance-masked run at v5p-32 scale"
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/abc123def
    entity: timodonnell
    project: MarinFold
    run_id: abc123def              # W&B's immutable internal ID — the uniqueness key
    run_name: fuzzy_cloth           # W&B's human-readable display name
  git_sha: <full sha>
  iris_job_ids:
    - <job_id_1>
    - <job_id_2>                    # appended on preemption / restart
---

# 2026-05-12 · exp13_models_train_1b · fuzzy_cloth

(detailed description, changes from prior runs, notes go below)
```

## Filename convention

`<YYYYMMDD>_<experiment>_<wandb_run_name>.md`, e.g.
`20260512_exp13_models_train_1b_fuzzy_cloth.md`.

Components:
- `<YYYYMMDD>`: launch date, UTC.
- `<experiment>`: the experiment dir name (`exp<N>_<kind>_<name>`), or
  `no_experiment` for runs outside any experiment.
- `<wandb_run_name>`: the W&B-side display name.

The filename is for human sortability and grep. The **uniqueness key
for a run is `wandb.run_id`**, not the filename — if two W&B-emitting
processes share a W&B run_id, they share one history file.

## Tooling

All subcommands live under `marinfold history` (installed by
`uv sync` in `experiments/`):

```bash
# Create a new history file (after `wandb.init()` returns):
marinfold history new \
    --experiment exp13_models_train_1b \
    --kind models \
    --wandb-url https://wandb.ai/open-athena/MarinFold/runs/<id> \
    --short "First 1B at v5p-32 scale" \
    --iris-job <job-id>

# Append an iris job id (e.g. after a preemption restart):
marinfold history add-iris-job 20260512_exp13_models_train_1b_fuzzy_cloth <new-job-id>

# Pull runs from W&B and create skeleton history files for any missing one:
uv sync --extra wandb                # one-time
marinfold history sync --limit 200

# Regenerate the summary table:
marinfold history update-index

# CI gate: exit non-zero if any W&B run lacks a history file:
marinfold history check
```

`sync` and `check` need the optional `wandb` extra (the W&B Python SDK).
`new`, `add-iris-job`, and `update-index` are stdlib + pyyaml only.

## Policy

Every W&B-logged run must have a history file. The expected workflow
is:

1. **At launch time**: run `marinfold history new ...` once you have
   the W&B URL (printed by `wandb.init()`).
2. **On preemption / restart**: `marinfold history add-iris-job ...`
   with the new iris job ID.
3. **Periodically** (or before merging): `marinfold history sync` to
   catch anything missed, then `marinfold history update-index` to
   refresh `RUNS.md`.

See the root `AGENTS.md` for the agent-facing version of this rule.
