# AGENTS.md

Rules and conventions for AI agents working in this repo. Claude Code,
Codex, Cursor, and similar tools should treat these as overriding
defaults. Layered atop these are per-subdirectory `AGENTS.md` files
(`experiments/AGENTS.md`, `models/AGENTS.md`, …) which add subsystem-
specific rules.

## Project shape

MarinFold trains protein-structure language models on Marin
infrastructure. Five concerns live at the repo root:

- `experiments/` — one dir per GitHub issue tagged `experiment`.
  All new work starts here as `exp<N>_<kind>_<name>/`. Holds prose
  READMEs, launchable `.py` files, and small artifacts (CSVs feeding
  plots, plots themselves). Also holds the `marinfold_experiments`
  package with the PM tools (`marinfold scaffold`,
  `marinfold itemize`, `marinfold graduate`).
- `models/` — library for model-training experiments
  (`marinfold_models.defaults`, `marinfold_models.simple_train_config`,
  …). Plus symlinks to *graduated* model experiments.
- `evals/` — library for eval experiments (production wrappers
  around iris launches). Plus graduated eval symlinks.
- `data/` — library for data-generation experiments
  (production wrappers around zephyr pipelines). Plus graduated
  data symlinks.
- `document_structures/` — interface (`DocumentStructure` Protocol)
  + local-testing CLI (`marinfold-document-structure`) for
  document-structure implementations. Plus graduated symlinks.

Each of these is a self-contained "marin-experiments-style" directory:
its own `pyproject.toml`, its own `.venv`, marin (or stdlib-only deps,
for `experiments/` and `document_structures/`) pulled in via wheels.

Experiments may import from any kind library via path deps in their
own `pyproject.toml`. Libraries DO NOT import from experiments — that
direction is forbidden. If two experiments need the same helper,
promote it to the kind library once a second use case actually exists
(not before).

See `experiments/README.md` for the workflow and graduation flow.

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
- Regenerating index files (`marinfold itemize`,
  `marinfold history update-index`).
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

### Run history

**Every W&B-logged run gets a history file under `history/runs/`.**
A "run" here is anything with a W&B link — training, evals, data-gen
pipelines that emit metrics. Multiple processes contributing to the
same W&B `run_id` share one history file.

The file is created right after `wandb.init()` returns (so the W&B
URL is in hand). Use:

```bash
marinfold history new \
    --wandb-url <url> --wandb-name <name> \
    --experiment <exp<N>_<kind>_<name>-or-no_experiment> \
    --kind <models|evals|data|document_structures|other> \
    --short "<one-line description>" \
    --iris-jobs <id1> [<id2> ...]
```

On preemption / restart, append the new iris job ID:

```bash
marinfold history add-iris-job <run-stem-or-wandb-name> <new-iris-job-id>
```

To catch anything that slipped through, `marinfold history sync`
queries the W&B API and creates skeleton files for any runs without
one (needs the `wandb` extra: `uv sync --extra wandb` in
`experiments/`). `marinfold history check` exits non-zero if drift
exists — wire to CI.

Always re-run `marinfold history update-index` after creating or
editing a history file so `history/RUNS.md` stays current.

See `history/README.md` for the schema and the full policy.

## See also

- `RESOURCES.md` — datasets, tokenizers, prior repos and prior runs.
- `experiments/AGENTS.md` — rules for working under `experiments/`.
- `models/README.md` — the model-training subproject.
