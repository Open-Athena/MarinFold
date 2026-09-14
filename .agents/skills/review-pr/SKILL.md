---
name: review-pr
description: Multi-agent correctness and AGENTS.md-compliance review of a pull request. Run only when explicitly requested or from CI; `--comment` posts findings as inline PR comments.
allowed-tools: Bash(gh issue view:*), Bash(gh search:*), Bash(gh issue list:*), Bash(gh pr comment:*), Bash(gh pr diff:*), Bash(gh pr view:*), Bash(gh pr list:*), mcp__github_inline_comment__create_inline_comment
---

Provide a code review for the given pull request: `/review-pr [--comment] <PR>`.

Adapted from marin-community/marin's `.agents/skills/review-pr/SKILL.md` (via Open-Athena/marin-dna's port). The review pipeline and the high-signal policy are unchanged; the repo-specific parts (what counts as a quotable guidance rule, comment conventions, permalink format) are MarinFold's, derived from the root `AGENTS.md` and the per-directory `AGENTS.md` files. When those change, re-sync the rule lists below.

**Agent assumptions (applies to all agents and subagents):**
- All tools are functional. Do not test tools or make exploratory calls.
- Only call a tool if it is required to complete the task.

Follow these steps precisely:

1. Launch a haiku agent to check if any of the following are true:
   - The PR is closed
   - The PR is a draft
   - The PR does not need code review (e.g. a dependabot bump, a trivial obviously-correct change)
   - Claude has already commented on this PR (check `gh pr view <PR> --comments`) AND a re-review was not explicitly requested. When a maintainer explicitly requests a re-review, always proceed even if a prior review exists.

   If any condition is true, stop. Note: still review agent-authored PRs (`claude/*` and `codex/*` branches, the `agent-generated` label).

2. Launch a haiku agent to return file paths (not contents) for all relevant guidance files:
   - The root `AGENTS.md` (MarinFold has no `CLAUDE.md`)
   - Any `AGENTS.md` or `CLAUDE.md` in directories (and parent directories) containing files modified by the PR — in practice `experiments/AGENTS.md`, `marinfold/AGENTS.md`, `models/AGENTS.md`
   - `.agents/skills/zephyr-pipeline-performance/SKILL.md` when the PR adds or changes an `experiments/exp<N>_data_*/` pipeline (`AGENTS.md` requires reading it before drafting `cli.py`)
   - `history/README.md` when the PR adds or edits files under `history/`

3. Launch an opus agent to view the PR and return a summary of the changes. The same agent checks the PR title, description, and labels against the few PR-level rules `AGENTS.md` and `experiments/AGENTS.md` state, and returns any problems it finds:

   - an issue-closing keyword (`fixes #N`, `closes #N`, `resolves #N`) that targets an experiment issue (label `experiment`) — merging would close it, and `experiments/AGENTS.md` rule 6 makes closing experiment issues a human-only action;
   - a PR from a `claude/*` or `codex/*` branch without the `agent-generated` label (`experiments/AGENTS.md` rule 5: "Agent-opened PRs/issues carry the `agent-generated` label").

   MarinFold has no PR writing-style guide, so flag only those concrete violations. A terse, plain body for a small change is correct — do not flag brevity, structure, or the absence of markdown.

4. Launch 4 agents in parallel to independently review the changes. Each returns a list of issues; each issue includes a description and the reason it was flagged (e.g. "AGENTS.md adherence", "bug"). An empty list is a fine answer.

   Agents 1 + 2: AGENTS.md compliance opus agents. Audit changes for compliance. When evaluating a file, only consider guidance files that share its path or are parents, plus the area skill from step 2. The `AGENTS.md` rules concrete enough to quote, and therefore in scope:

   - **Error handling.** A swallowed exception — `except …: pass`, or a catch that returns a default / logs and continues without re-raising — where the code is not intentionally altering control flow (root: "NEVER EVER SWALLOW EXCEPTIONS unless specifically requested").
   - **No backward compatibility.** New deprecation warnings, fallback paths, compatibility shims, or `hasattr(m, "old_attr")`-style checks instead of updating call sites (root "Deprecation" and "Code style").
   - **Never monkey-patch.** Replacing a function, method, or attribute of an imported module at runtime (root "Never monkey-patch").
   - **Imports.** `from __future__ import …` anywhere; a local (in-function) import that is not breaking a circular dependency or guarding an optional dependency. Lazy imports of `vllm` / `torch` / `mlx` in `marinfold` are required by `marinfold/AGENTS.md` rule 1, not a violation.
   - **Library direction.** Code under `marinfold/` or `models/` importing from `experiments/` (root "Project shape"; `models/AGENTS.md` rule 7).
   - **Light base install** (`marinfold/` only). `import marinfold`, the CLI, the registry, or `document_structures.{core,writers,io}` pulling in `vllm`, `torch`, or `mlx` at import time (`marinfold/AGENTS.md` rule 1). A top-level `import torch` inside a backend module such as `inference/_transformers.py` is fine — that module is itself lazy-imported.
   - **Protein-unaware core** (`marinfold/` only). `marinfold.inference` or `marinfold.document_structures.{core,writers}` gaining protein-specific code: gemmi / biopython imports, contact or distance constants, `<d_X.X>` / `<p_N>` vocabulary (`marinfold/AGENTS.md` rule 4). Format-specific logic belongs in a `document_structures.<name>` subpackage.
   - **Interchangeable backends** (`marinfold/` only). A `Backend` protocol method gaining a parameter that only one concrete backend honours; backend-specific options belong in that backend's `__init__` kwargs (`marinfold/AGENTS.md` rule 2). A generic generation parameter every backend implements is not a violation.
   - **Strict model resolution** (`marinfold/` only). `resolve_model` falling through to treating an unknown string as an HF repo id (`marinfold/AGENTS.md` rule 3).
   - **W&B routing.** `WANDB_PROJECT` / `WANDB_ENTITY` (or `wandb.init(project=…, entity=…)`) set to anything other than `MarinFold` / `open-athena` (root "W&B routing").
   - **Checkpoint paths.** A checkpoint written to or referenced by a path that lacks the W&B run name or the step number, e.g. `…/final/` or `…/latest/` (root "HF bucket"; `models/AGENTS.md` rule 6).
   - **Tokenizers.** A tokenizer name without a `repo@revision` pin in training code (`models/AGENTS.md` rule 4); a model push to HuggingFace without the tokenizer files, e.g. `convert_checkpoint_to_hf_step` called without `tokenizer=` (`models/AGENTS.md` rule 5).
   - **GCS layout.** Large outputs written outside `gs://marin-<region>/protein-structure/MarinFold/<experiment-name>/`, or to a `marin-<region>` bucket whose region visibly differs from the zone the same job pins (root "GCS bucket").
   - **Experiment layout** (`experiments/` only). A new experiment dir not named `exp<N>_<kind>_<name>` with `<kind>` in `models` / `evals` / `data` / `document_structures`, or whose frontmatter `issue:` does not match `<N>` (`experiments/AGENTS.md` rule 1; `exp0_*` is the reserved exception).
   - **Binaries in git.** Model weights, parquets, prediction dumps, or other large artifacts committed instead of pushed to GCS / the HF bucket (`experiments/AGENTS.md` rule 2). Small CSVs under `data/`, plots and their `*.meta.json` sidecars under `plots/`, and notebooks with outputs are required, not violations.
   - **Notebooks need no auth.** A committed `.ipynb` that reads from `gs://` or another private path instead of the public HF bucket (`experiments/AGENTS.md` rule 2).
   - **Run history index.** A PR that adds or edits `history/runs/*.md` without the matching `history/RUNS.md` update (root "Run history": "Always re-run `python scripts/history.py update-index`").
   - **Tests.** A test fixed by relaxing a tolerance or special-casing around the failure (root "Testing").
   - **Data pipelines.** A Zephyr / `map_shard` pipeline that silently drops or skips failed rows instead of failing loudly (`zephyr-pipeline-performance` skill).

   Agents 3 + 4: opus bug agents (parallel). Scan for obvious bugs, security issues, and incorrect logic within the changed code. Focus only on the diff without reading extra context. Flag only significant bugs you can validate from the diff alone; ignore nitpicks and likely false positives. Protein-structure / LM-training bugs that deserve extra suspicion: residue-index off-by-one (0- vs 1-based numbering, `label_seq_id` vs `auth_seq_id`); contact definitions applied inconsistently (Cα vs Cβ, glycine lacking Cβ, sequence-separation thresholds `<` vs `<=`, counting both `(i, j)` and `(j, i)`); Å vs nm; next-token label shift and BOS/EOS off-by-one; padding or prompt tokens leaking into a loss or metric; top-L selection in R-precision using the wrong `L` or sort direction; softmax or argmax over the wrong axis; tokenizer special-token ids assumed rather than looked up; eval proteins leaking into training data (decontamination); silent NaN propagation; reused JAX PRNG keys.

   **CRITICAL: We only want HIGH SIGNAL issues.** Flag issues where:
   - The code will fail to compile or parse (syntax errors, type errors, missing imports, unresolved references)
   - The code will definitely produce wrong results regardless of inputs (clear logic errors)
   - Clear, unambiguous AGENTS.md violations where you can quote the exact rule being broken

   Do NOT flag:
   - Code style or quality concerns
   - Potential issues that depend on specific inputs or state
   - Subjective suggestions or improvements

   If you are not certain an issue is real, do not flag it. False positives erode trust.

   Tell each subagent the PR title and description for author-intent context.

   **MarinFold-specific:** duplicating a helper across experiments is intentional until a second use case exists and the abstraction is stable (`experiments/AGENTS.md` rule 7, `models/AGENTS.md` rule 8) — do not flag copy/paste or suggest promoting code into `marinfold/` or `models/`. Experiment dirs and one-off `_scripts/` are held to correctness, not structure: hard-coded paths, run names, and magic constants there are normal. Commit messages (including attribution trailers) are out of scope. `uv.lock` churn and regenerated `history/` files are expected.

5. For each issue from step 4 — from all four agents, compliance and bug alike — launch a parallel subagent to validate it. Give the subagent the PR title, description, and issue description. It must confirm with high confidence that the issue is real — e.g. for "variable is not defined", verify that in the code; for an AGENTS.md issue, verify the rule is scoped to this file and actually violated. Use opus subagents throughout — for both bugs/logic and AGENTS.md violations.

6. Filter out any issues not validated in step 5. The remainder is the high-signal review list.

7. Output a summary of the review findings to the terminal:
   - If issues were found, list each issue with a brief description.
   - If no issues were found, state: "No issues found. Checked for bugs and AGENTS.md compliance."
   - Separately, report any PR-description problems from step 3.

   If `--comment` argument was NOT provided, stop here. Do not post any GitHub comments.

   If `--comment` IS provided and step 3 found PR-description problems, post **one** top-level comment with `gh pr comment` (prefixed `🤖`, not inline) naming the specific problems and the concrete fix. This is independent of the code review — post it whether or not code issues were found, but skip it when the description is fine.

   If `--comment` IS provided and NO code issues were found, post the no-issues summary comment (format below) using `gh pr comment` and stop.

   If `--comment` IS provided and code issues were found, continue to step 8.

8. Draft the list of comments you plan to leave. For your own review only — do not post it anywhere.

9. Post inline comments for each issue using `mcp__github_inline_comment__create_inline_comment` with `confirmed: true`. For each comment:
   - Begin the body with `🤖` (`experiments/AGENTS.md` rule 5: "Agent comments start with 🤖")
   - Provide a brief description of the issue
   - For small, self-contained fixes, include a committable suggestion block
   - For larger fixes (6+ lines, structural changes, or changes spanning multiple locations), describe the issue and suggested fix without a suggestion block
   - Never post a committable suggestion UNLESS committing the suggestion fixes the issue entirely. If follow-up steps are required, do not leave a committable suggestion.

   **IMPORTANT: Only post ONE comment per unique issue. Do not post duplicate comments.**

Use this list when evaluating issues in steps 4 and 5 (these are false positives, do NOT flag):

- Pre-existing issues
- Something that appears to be a bug but is actually correct
- Pedantic nitpicks that a senior engineer would not flag
- Issues that a linter or type checker will catch — unused imports, formatting, missing annotations (MarinFold has no lint CI; agents type-check locally with `pyrefly` per `AGENTS.md`; do not run them to verify)
- General code quality concerns (e.g. general security issues) unless explicitly required in AGENTS.md
- Issues mentioned in AGENTS.md but explicitly silenced in the code (e.g. via a lint ignore comment)

Notes:

- Use the gh CLI to interact with GitHub (fetch pull requests, create comments). Do not use web fetch.
- Create a todo list before starting.
- You must cite and link each issue in inline comments (e.g. when referring to AGENTS.md, include a permalink to it, ideally with line numbers).
- If no issues are found and `--comment` is provided, post a comment with exactly this format:

---

## 🤖 Code review

No issues found. Checked for bugs and AGENTS.md compliance.

---

- When linking to code in inline comments, follow this format precisely, otherwise the Markdown preview won't render: https://github.com/Open-Athena/MarinFold/blob/<full-40-char-sha>/AGENTS.md#L10-L15
  - Requires the full git sha. Commands like `https://github.com/owner/repo/blob/$(git rev-parse HEAD)/foo/bar` will not work, since your comment is rendered directly as Markdown.
  - Repo name must be `Open-Athena/MarinFold`.
  - `#` after the file name; line range format is `L[start]-L[end]`.
  - Provide at least 1 line of context before and after, centred on the line you are commenting about (commenting on lines 5-6 → link `L4-L7`).
