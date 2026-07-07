# AGENTS.md

Rules and conventions for AI agents working in this repo. Claude Code,
Codex, Cursor, and similar tools should treat these as overriding
defaults. Layered atop these are per-subdirectory `AGENTS.md` files
(`experiments/AGENTS.md`, `models/AGENTS.md`, …) which add subsystem-
specific rules.

## Project shape

MarinFold trains protein-structure language models on Marin
infrastructure. Concerns at the repo root:

- `experiments/` — one dir per GitHub issue tagged `experiment`.
  All new work starts here as `exp<N>_<kind>_<name>/`. Holds prose
  READMEs, launchable `.py` files, and small artifacts (CSVs feeding
  plots, plots themselves).
- `scripts/` — repo-management scripts (`scaffold.py`, `itemize.py`,
  `history.py`). Run with plain `python scripts/<name>.py`.
- `marinfold/` — the top-level Python package. Backends, model
  registry, document-structure shared toolkit, graduated
  document-structure impls (as subpackages of
  `marinfold.document_structures.<name>`), and the user-facing
  `marinfold infer` / `marinfold evaluate` CLI.
- `models/` — library for model-training experiments
  (`marinfold_models.defaults`, `marinfold_models.simple_train_config`,
  …). Kept separate from `marinfold/` because the training dep
  stack (marin/levanter/jax) is heavy and platform-coupled and
  has no business in the inference path.

Kind libraries for `evals` and `data` (and any other future kind)
are created on demand — when a second experiment in that kind
needs the same helper. Don't pre-scaffold an empty library.

Each kind dir / top-level Python package is a self-contained
directory with its own `pyproject.toml` and its own `.venv`.

Experiments may import from any kind library via path deps in their
own `pyproject.toml`. Libraries DO NOT import from experiments — that
direction is forbidden. If two experiments need the same helper,
promote it to the kind library once a second use case actually exists
(not before).

See `experiments/README.md` for the workflow and graduation flow,
and `marinfold/README.md` for the inference/CLI surface. For any
data-generation pipeline that runs on the marin Iris cluster via
Zephyr, read the [`zephyr-pipeline-performance`](.agents/skills/zephyr-pipeline-performance/SKILL.md)
skill before drafting `cli.py`. It captures the handful of decisions
that separate a fits-in-budget run from an overnight one.

## Shared coding practices

Mirrored from `marin-community/marin-experiments/AGENTS.md` — keep them
consistent unless we deliberately diverge.

### Tooling

- Assume Python >= 3.11.
- Always use `uv run` for Python entry points. If that fails, try
  `.venv/bin/python` directly.
- Use type hints.
- Prefer `pyrefly` for type-checking.

### Communication & commits

- NEVER SAY "You're absolutely right!"
- Never credit yourself in commits. NEVER EVER EVER credit yourself in
  commit messages.

### Code style

- Put all imports at the top of the file. Avoid local imports unless
  technically necessary (e.g. to break circular dependencies or guard
  optional dependencies).
- Prefer top-level functions when code does not mutate shared state;
  use classes to encapsulate data when that improves clarity.
- Prefer top-level Python tests and fixtures.
- Disprefer internal mutation of function arguments, especially config
  dataclasses. Prefer returning a modified copy
  (`dataclasses.replace(...)`) so call sites stay predictable.
- Use early returns (`if not x: return None`) when they reduce nesting.
- Do not introduce ad-hoc compatibility hacks like
  `hasattr(m, "old_attr")`; update the code consistently instead.
- Do not use `from __future__ import ...` statements.
- Document public APIs with concise Google-style docstrings.

### Error handling

- Let exceptions propagate by default.
- Only catch exceptions when you can add meaningful context and re-
  raise, or when you are intentionally altering control flow.
- NEVER EVER SWALLOW EXCEPTIONS unless specifically requested.

### Deprecation

**No backward compatibility**: do not add deprecation warnings,
fallback paths, or compatibility shims. Update all call sites instead.
Only add backward compatibility if the user explicitly requests it.

### Comments

Write detailed comments when describing behavior as a whole, e.g. at
module or class level, or when describing some subtle behavior.
Do not generate comments that merely restate the code.

### Testing

- Always fix tests if you broke them.
- Do not fix tests by relaxing tolerances or hacking around them.
- Avoid "tautological" tests that merely restate implementation logic.
- Run the appropriate tests for your changes.

## Hard rules

### Branch + PR for substantive work; don't push directly to main

Substantive changes — new code, multi-file edits, design decisions —
go on a feature branch and land via a GitHub PR, even when the
intent is to merge straight into `main`. The branch doesn't need to
live long: open the PR, run review (e.g. `/ultrareview` against
`origin/main`), merge, delete the branch.

Branch naming: `<thread>/<short-name>` (e.g. `exp1/eval-impl`,
`docs/agents-update`). For an experiment that lives entirely on a
branch (the `marinfold_experiment.branch` frontmatter field), use
`exp/<N>-<slug>` per the existing convention.

What can still go direct to `main`:

- Pure typo / one-line doc fixes.
- Regenerating index files (`python scripts/itemize.py`,
  `python scripts/history.py update-index`).
- Hotfix reverts when something is actively broken.

What goes through a PR by default:

- New `.py` files or non-trivial edits to existing ones.
- New experiments (the whole `exp<N>_<kind>_<slug>/` dir).
- AGENTS / README / RESOURCES policy changes.
- Anything an agent would benefit from independent review on
  (`/ultrareview`-able).

The point isn't to slow merges into `main` — most PRs should be
short-lived. It's to give `/ultrareview` (and any future review
tooling) a real diff to chew on.

### Never monkey-patch

Do not replace functions, methods, or attributes of imported modules
at runtime. Monkey-patches are silent, non-local, and frequently don't
work the way you expect.

If a third-party library has a hard-coded behavior that doesn't fit
our needs:

1. Pad / preprocess inputs so the library's code path works (preferred)
2. Wrap or subclass the library's exposed API
3. Open an issue / contribute a patch upstream
4. As a last resort: vendor a small fork of the offending file with a
   clear explanation

If none of those work without significant engineering, **ask the user**
before introducing a workaround.

### W&B routing

All training runs log to **`https://wandb.ai/open-athena/MarinFold`**
(`WANDB_PROJECT=MarinFold`, `WANDB_ENTITY=open-athena`). Do not set
either env var to a different value when launching a run — single-
project routing is what makes the leaderboard view (per-issue
comparisons, x-axis sweeps) useful.

For one-off scratch work that shouldn't pollute the shared project,
prefix the run name (`debug-cuda-oom`, `exp9-lrsweep-3e-4`) — don't
fork the project.

### HF bucket: `open-athena/MarinFold`

We use a single HF bucket — `https://huggingface.co/buckets/open-athena/MarinFold` —
for **both data artifacts and model checkpoints**. First-class
published datasets and released models live in their canonical HF
dataset / model repos; the bucket holds the long tail — intermediate
parquets, eval outputs, predicted structures, in-flight checkpoints.
The bucket may be split later if listing gets unwieldy or different
retention/access policies are needed.

**Inside the bucket, two top-level prefixes:**

- `data/...` — data artifacts (intermediate parquets, predicted
  structures, eval inputs, anything that isn't a model weight).
- `checkpoints/...` — model weights (Levanter checkpoints, HF
  exports, anything loadable as a model).

**Checkpoint paths must include both the W&B run name AND the step
number.** The canonical layout is

```
checkpoints/<wandb-run-name>/step-<N>/
```

so e.g. a Levanter-native checkpoint lands at
`checkpoints/protein-contacts-1b-3.5e-4-distance-masked-7d355e/step-31337/`
and the HF export at
`checkpoints/protein-contacts-1b-3.5e-4-distance-masked-7d355e/hf/step-31337/`.

Both the W&B run name (so you can cross-reference back to metrics)
and the step number (so you can tell which point in training you
loaded) need to be in the path. Don't store "just `final/`" or
"just `latest/`" — those obscure which checkpoint a downstream eval
actually ran against and are a reproducibility hazard. When
referring to a checkpoint as a string identifier anywhere (W&B
artifact names, history file shorthand, paper writeups), the same
"`<wandb-run-name>-step-<N>`" format applies.

**Always save the tokenizer with the model.** When pushing a model
to HuggingFace — whether to the `buckets/open-athena/MarinFold`
bucket or to a public `models` repo — include the tokenizer files
(`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`,
etc.) in the same repo / revision as the model weights. A model
without its tokenizer is unloadable for downstream eval, vLLM
serving, and reproducibility checks. This applies even when the
tokenizer is "well-known" and pinned by URL elsewhere (e.g.
`timodonnell/protein-docs-tokenizer@<sha>`) — co-locate it so
nothing breaks if the source tokenizer URL changes.

### Cross-region data transfers & the shared Iris cluster

Storage and bandwidth are dominant cost drivers for this project,
and the marin Iris pools straddle regions and continents — the v5p
TPU pool spans `{us-central1-a, us-east5-a}`, the CPU pool reaches
`europe-west4`, and the controller lives in `us-central1-a`. There
is no single region that "the cluster" is in, so cross-region I/O is
a default failure mode, not a rare edge case. Treat it as something
to avoid:

- **Don't stream large data across regions or to the open internet.**
  Co-locate a job's GCS I/O with its compute zone (see "GCS bucket"
  below). Stream through `fsspec` rather than copying artifacts to
  local disk, and never pull more than a few MB across regions in a
  hot path.
- **Pin the worker zone** (`--zone`, `ResourceConfig`) so workers
  land in-region with their data, and prefer in-region preemptible
  pools over over-requesting on-demand capacity. Over-requesting
  on-demand workers spills them cross-continent: exp53 asked for a
  large on-demand CPU pool, spilled to `europe-west4`, and those
  workers did slow trans-Atlantic reads/writes and became the
  straggler tail.
- **A cross-region copy larger than 10 GB needs explicit human
  sign-off**, regardless of previous instructions. Under that, mirror
  once into the local region and reference the local copy thereafter
  (mirrors marin's `TransferBudget` default of 10 GB).
- **Never use the GCS storage-transfer-service** to move data between
  regions without explicit user approval — bulk cross-region moves
  are a real cost event.
- **Never stop, restart, or bounce the shared Iris cluster** unless
  the user gives express permission. Other people's jobs run on it.

### GCS bucket: `marin-<region>/protein-structure/MarinFold/` (co-locate with compute)

When an experiment needs to dump large eval/data outputs (raw
distograms, prediction batches, intermediate parquets) to GCS —
typically because it ran on TRC via iris and the worker is
ephemeral — write them under the
`protein-structure/MarinFold/<experiment-name>/` prefix, but in the
`marin-<region>` bucket **co-located with the zone the job's workers
actually run in**, not a fixed region:

```
gs://marin-<region>/protein-structure/MarinFold/<experiment-name>/...
```

- **TPU training / eval** follows the same rule: pin the TPU zone,
  then write to the matching `marin-<region>` bucket. For our
  current v5p-based MarinFold jobs that usually means `us-east5-a`
  and `gs://marin-us-east5/protein-structure/MarinFold/<exp>/`, but
  that is an example, not a global rule.
- **CPU data-gen on the Iris cluster** likewise depends on where the
  workers actually land: us-central1 / us-central2 are common, but
  fallback pools can spill farther. Pin the worker zone and write to
  the matching same-region bucket, e.g.
  `gs://marin-us-central1/protein-structure/MarinFold/<exp>/` for a
  job pinned to `us-central1-a`. Writing that output to a different
  region (for example `marin-us-east5`) means every worker does a
  cross-region PUT; exp5 already moved to `marin-us-central1` for
  exactly this reason.
- If a single canonical location is genuinely needed, do **one bulk
  copy after the job completes**, not thousands of streamed
  cross-region worker PUTs — and respect the > 10 GB sign-off rule
  above.

Keep the `protein-structure/MarinFold/<experiment-name>/` prefix
identical across regions, so MarinFold outputs stay co-located by
convention and the marin team can find them at a glance. Make
`<experiment-name>` informative enough to recognize without reading
the source (e.g.
`exp26/protein-contacts-1_5b-distance-masked-70f8f5-step-49999-foldbench-monomers/`).
Don't sprinkle MarinFold artifacts across `marin-<region>/eval/...`,
`marin-<region>/checkpoints/...`, etc. (those prefixes belong to the
marin protein-experiments convention, not ours).

The same "small artifacts (CSVs, plots) stay in git, large
artifacts go to durable storage" rule from the HF bucket section
applies — GCS is for the big stuff that wouldn't fit in the
experiment dir.

### CoreWeave RNO2A GPU cluster (`cw-rno2a`)

MarinFold can also train on **GPU** via CoreWeave RNO2A (`cw-rno2a`, 512× H100,
an iris cluster; config in marin `lib/iris/config/cw-rno2a.yaml`, runbook
`lib/iris/docs/coreweave.md`). exp108 was the first MarinFold GPU run — its
`experiments/exp108_.../README.md` has the recipe/runbook; the durable,
cluster-general lessons are here:

- **Two easy-to-miss launcher prereqs.** Beyond the kubeconfig
  (`~/.kube/coreweave-iris-rno2a`, context `marin-rn02a_RNO2A`) and the
  object-storage key (`CW_KEY_ID`/`CW_KEY_SECRET`, console → Object Storage):
  `kubectl` must be **on `PATH`** (the launcher shells out to it for the
  controller tunnel — match the cluster k8s version, currently v1.36.x; missing
  it → `Could not connect to controller: … 'kubectl'`), and the experiment venv
  needs **`marin-iris[controller]`** (missing it → `Install iris[controller] to
  use CloudK8sService`).
- **Storage is S3, not GCS.** CoreWeave AI Object Storage, bucket
  `marin-us-east-02a`. From a workstation use `https://cwobject.com` with
  **virtual-hosted addressing** (path-style is rejected — `PathStyleRequestNotAllowed`);
  in-cluster jobs get creds + the LOTA endpoint (`http://cwlota.com`) injected by
  iris. Task pods carry **one** endpoint/credential set — GCS is unreachable from
  jobs, so stage inputs into the CoreWeave bucket first, under a single removable
  top-level prefix (exp108 used `s3://marin-us-east-02a/MarinFold/`).
- **Batch priority does NOT propagate to executor-spawned children.** `iris job
  run --priority batch` sets only the *driver* band; the marin executor /
  `remote()` submit child jobs (tokenize, training gang) with no band →
  interactive. To run training at batch, dispatch it yourself as a
  `fray.JobRequest(priority=3)` (see exp108 `dispatch_train.py`;
  `PRIORITY_BAND_BATCH == 3`, forwarded verbatim by `fray/iris_backend.py`) — the
  sanctioned way, consistent with "Never monkey-patch" above.
- **A driver that submits child gangs must WAIT on them.** Gangs are *children*
  of the driver job; if the driver exits, iris finalizes (kills) them.
- **Multi-node GPU: reliable ceiling ≈ 4 nodes (as of 2026-07).** 1/2/4-node
  gangs bootstrap and train; **8-node fails** — the JAX multi-host coordination
  bootstrap aborts (~5 min in, `CoordinationServiceAgent::SetError`), reproduced
  on a single isolated 8-node gang. Launch multiple concurrent gangs as
  **separate, staggered driver jobs** (~90 s apart) — several gangs from one
  driver collide on coscheduling/coordination. The likely (untried) >4-node fix
  is the `NCCL_*` env grug forwards; exp108's dispatch forwards
  `XLA_FLAGS`/`NCCL_`/`JAX_` from the driver but sets none.
- **GPU env quirks.** `CUDA_ERROR_NOT_PERMITTED` cuMemCreate **FABRIC** warnings
  are **benign** (the container lacks NVIDIA IMEX; XLA falls back — though this
  also drops the NVLS collective fast path, hurting FSDP MFU). Transformer Engine
  is **not** in the `--extra gpu` env, so levanter's default GPU NVTE attention
  silently falls back to the vanilla O(seq²) kernel — set `attn_backend=JAX_FLASH`
  (pallas flash, no TE dependency) for long sequences.
- **Always pin + `uv.lock` marin.** The marin dev line moves daily and
  periodically refactors its API (0.2.38 removed the old `marin.execution`
  executor surface — `ExecutorStep`/`this_output_path`/`executor_main`/
  `default_train`; the modern assembler is `marin.experiment.train.train_lm`).
  An unpinned range that worked last month silently breaks; a committed `uv.lock`
  is what saved exp67/exp85.
- **Monitoring** (no reliable `-f` for scripting): `uv run iris
  --cluster=cw-rno2a job list | grep <name>` and `… job logs <job> --max-lines N`
  (filter `zephyr`/`aiobotocore`/`cuda_vmm` noise; a gang's own state line is
  `…<name> <state>` with a **space** — sub-job lines have a trailing `/`).
  `iris job stop <job>` to kill.

### Run history

**Every W&B-logged run gets a history file under `history/runs/`.**
A "run" here is anything with a W&B link — training, evals, data-gen
pipelines that emit metrics. Multiple processes contributing to the
same W&B `run_id` share one history file.

The file is created right after `wandb.init()` returns (so the W&B
URL is in hand). Use:

```bash
python scripts/history.py new \
    --wandb-url <url> --wandb-name <name> \
    --experiment <exp<N>_<kind>_<name>-or-no_experiment> \
    --kind <models|evals|data|document_structures|other> \
    --short "<one-line description>" \
    --iris-jobs <id1> [<id2> ...]
```

On preemption / restart, append the new iris job ID:

```bash
python scripts/history.py add-iris-job <run-stem-or-wandb-name> <new-iris-job-id>
```

To catch anything that slipped through, `python scripts/history.py sync`
queries the W&B API and creates skeleton files for any runs without
one (needs the `wandb` extra: `uv sync --extra wandb` in `scripts/`).
`python scripts/history.py check` exits non-zero if drift exists —
wire to CI.

Always re-run `python scripts/history.py update-index` after creating or
editing a history file so `history/RUNS.md` stays current.

See `history/README.md` for the schema and the full policy.

### Capture timings for every predictor run

**Always record per-protein (or per-input) wall-time and worker
metadata when you run *any* predictor**, whether it's a MarinFold
model, Protenix, ESMFold, AlphaFold, or anyone else's. **Save
timings to a CSV at evaluation time, not "we can reconstruct it
later from logs"** — Modal's ephemeral-app logs get pruned, iris
job records get garbage-collected, and "we'll grab it from the
output dirs' mtimes" is fragile across re-syncs. Bake it into the
predictor wrapper.

Minimum schema (one row per (input, mode) pair, or per (input,
seed) if the predictor splits work that way):

```
stem, n_residues, n_pairs, mode,
elapsed_seconds,             # pure inference time (matches AF3-paper convention)
model_load_seconds,          # weight-load + runner setup time, reported separately
total_seconds,               # everything: setup + inference + dump (sanity check)
model_nickname,              # e.g. "protenix-v2", "marinfold-1b"
runner_tag,                  # "modal", "iris", "local"
gpu_name, gpu_total_memory_gb, gpu_compute_capability,
hostname, platform, torch_version,
timestamp_utc
```

Plus any per-predictor-specific columns (e.g. `n_seeds`,
`n_samples_per_seed`, `batch_size`).

This schema is concretely realized in
[`experiments/exp12_data_protenix_foldbench_monomers/data/timings.csv`](experiments/exp12_data_protenix_foldbench_monomers/data/timings.csv)
and
[`experiments/exp20_evals_marinfold_1b_foldbench/data/timings.csv`](experiments/exp20_evals_marinfold_1b_foldbench/data/timings.csv) — they share enough columns
that the two CSVs join cleanly on `(stem, n_residues)`. Match it
when adding a new evaluator so cross-experiment timing comparisons
(e.g. MarinFold vs Protenix vs ESMFold scaling curves) just work.
Commit the CSV to git; upload alongside other artifacts when going
to the HF bucket. Plot conventions for the timing-vs-length curve
are in
[`experiments/exp12_data_protenix_foldbench_monomers/_scripts/plot_timings.py`](experiments/exp12_data_protenix_foldbench_monomers/_scripts/plot_timings.py).

For Modal-hosted predictors: capture worker metadata once in
`@modal.enter` (stash in `self.worker_meta`), time the inference in
`predict_one`, and write a `timings.json` per (input, mode) on the
output Volume — then a small post-hoc script aggregates those into
the CSV. Don't try to recover this from Modal logs after the fact;
ephemeral-app logs disappear.

## See also

- `RESOURCES.md` — datasets, tokenizers, prior repos and prior runs.
- `experiments/AGENTS.md` — rules for working under `experiments/`.
- `models/README.md` — the model-training subproject.
