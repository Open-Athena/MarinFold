# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""``marinfold history`` — manage the persistent audit trail of W&B runs.

Subcommands:

- ``new`` — create a new history file.
- ``add-iris-job`` — append an iris job ID to an existing run file.
- ``sync`` — pull runs from W&B, create skeletons for any missing one (needs `wandb`).
- ``update-index`` — regenerate ``history/RUNS.md``.
- ``check`` — exit non-zero if any W&B run lacks a history file (needs `wandb`).

Each run is one file in ``history/runs/`` with YAML frontmatter +
free-form body. The uniqueness key for a run is its W&B
``run_id`` (immutable), not the filename. See ``history/README.md``
for the schema.
"""

import argparse
import getpass
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from marinfold_experiments import RUN_KINDS
from marinfold_experiments._repo import REPO_ROOT, parse_experiment_dir_name

HISTORY_DIR = REPO_ROOT / "history"
RUNS_DIR = HISTORY_DIR / "runs"
RUNS_MD = HISTORY_DIR / "RUNS.md"

DEFAULT_WANDB_ENTITY = "timodonnell"
DEFAULT_WANDB_PROJECT = "MarinFold"

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


# -- File helpers --------------------------------------------------------


@dataclass(frozen=True)
class RunFile:
    """A parsed run history file."""

    path: Path
    metadata: dict[str, Any]  # the marinfold_run sub-dict
    body: str

    @property
    def run_id(self) -> str | None:
        return (self.metadata.get("wandb") or {}).get("run_id")

    @property
    def launched_at(self) -> str | None:
        return self.metadata.get("launched_at")


def _read_run_file(path: Path) -> RunFile:
    content = path.read_text()
    m = _FRONTMATTER_RE.match(content)
    if not m:
        raise ValueError(f"{path} has no YAML frontmatter block")
    front = yaml.safe_load(m.group(1)) or {}
    run = front.get("marinfold_run")
    if run is None:
        raise ValueError(f"{path} frontmatter has no `marinfold_run:` block")
    return RunFile(path=path, metadata=run, body=m.group(2))


def _write_run_file(rf: RunFile) -> None:
    front_block = yaml.safe_dump(
        {"marinfold_run": rf.metadata},
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    ).rstrip()
    text = f"---\n{front_block}\n---\n{rf.body}"
    rf.path.write_text(text)


def _existing_run_files() -> list[RunFile]:
    if not RUNS_DIR.is_dir():
        return []
    out = []
    for p in sorted(RUNS_DIR.glob("*.md")):
        try:
            out.append(_read_run_file(p))
        except (ValueError, yaml.YAMLError) as exc:
            print(f"[marinfold history] WARN: skipping unparseable {p}: {exc}", file=sys.stderr)
    return out


def _runs_by_id(files: list[RunFile]) -> dict[str, RunFile]:
    return {rf.run_id: rf for rf in files if rf.run_id}


# -- Filename derivation -------------------------------------------------


def _slugify_for_filename(s: str) -> str:
    """Lowercase, replace non [a-z0-9_] with underscores, collapse repeats."""
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_")
    return re.sub(r"_+", "_", s) or "unnamed"


def _filename_for(*, launched_at: str, experiment: str, wandb_run_name: str) -> str:
    date = launched_at[:10].replace("-", "")  # YYYY-MM-DD → YYYYMMDD
    return f"{date}_{experiment}_{_slugify_for_filename(wandb_run_name)}.md"


# -- W&B URL parsing -----------------------------------------------------


_WANDB_URL_RE = re.compile(
    r"https?://wandb\.ai/(?P<entity>[^/]+)/(?P<project>[^/]+)/runs/(?P<run_id>[^/?#]+)"
)


def _parse_wandb_url(url: str) -> dict[str, str]:
    """Extract entity/project/run_id from a wandb.ai run URL.

    The URL's last path segment is the W&B run_id (the immutable
    internal ID). The display run name is NOT in the URL; the caller
    must pass it separately or look it up via the W&B API.
    """
    m = _WANDB_URL_RE.match(url.strip())
    if not m:
        raise ValueError(
            f"Could not parse W&B URL: {url!r}. "
            "Expected the form https://wandb.ai/<entity>/<project>/runs/<run_id>."
        )
    return m.groupdict()


# -- Git --------------------------------------------------------------


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return None


# -- new -------------------------------------------------------------------


def _cmd_new(args: argparse.Namespace) -> int:
    wandb_parts = _parse_wandb_url(args.wandb_url)
    run_id = wandb_parts["run_id"]
    entity = wandb_parts["entity"]
    project = wandb_parts["project"]

    if args.wandb_name is None:
        print(
            "ERROR: --wandb-name is required (the display name printed by wandb.init), "
            "or use `marinfold history sync` which fetches it from the API.",
            file=sys.stderr,
        )
        return 2

    if args.kind not in RUN_KINDS:
        print(f"ERROR: --kind must be one of {RUN_KINDS}; got {args.kind!r}", file=sys.stderr)
        return 2

    launched_at = args.launched_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    existing = _runs_by_id(_existing_run_files())
    if run_id in existing and not args.force:
        print(
            f"Run {run_id} already exists at {existing[run_id].path.relative_to(REPO_ROOT)}. "
            "Use --force to overwrite, or `marinfold history add-iris-job` to append a job ID.",
            file=sys.stderr,
        )
        return 1

    metadata: dict[str, Any] = {
        "user": args.user or getpass.getuser(),
        "launched_at": launched_at,
        "experiment": args.experiment,
        "kind": args.kind,
        "short_description": args.short or "",
        "wandb": {
            "url": args.wandb_url,
            "entity": entity,
            "project": project,
            "run_id": run_id,
            "run_name": args.wandb_name,
        },
        "git_sha": args.git_sha or _git_sha() or "unknown",
        "iris_job_ids": list(args.iris_jobs or []),
    }

    fname = _filename_for(
        launched_at=launched_at, experiment=args.experiment, wandb_run_name=args.wandb_name,
    )
    path = RUNS_DIR / fname
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    body = _default_body(metadata)
    _write_run_file(RunFile(path=path, metadata=metadata, body=body))
    print(f"Wrote {path.relative_to(REPO_ROOT)}")
    return 0


def _default_body(metadata: dict[str, Any]) -> str:
    wandb = metadata["wandb"]
    date_short = metadata["launched_at"][:10]
    return (
        f"\n# {date_short} · {metadata['experiment']} · {wandb['run_name']}\n\n"
        f"**Launched:** {metadata['launched_at']} by {metadata['user']}  \n"
        f"**Kind:** {metadata['kind']}  \n"
        f"**Experiment:** {metadata['experiment']}  \n"
        f"**W&B:** [{wandb['run_name']}]({wandb['url']})  \n"
        f"**Git:** `{metadata['git_sha'][:8] if metadata['git_sha'] != 'unknown' else 'unknown'}`  \n\n"
        f"## Description\n\n"
        f"{metadata['short_description'] or '_(short description here)_'}\n\n"
        f"## Detailed plan\n\n"
        f"_(Why we ran this, what we expect to see, unusual parameters.)_\n\n"
        f"## Changes from previous runs\n\n"
        f"_(Bullet list of differences from the last run of this kind.)_\n\n"
        f"## Notes\n\n"
        f"_(Anything else worth remembering — preemptions, midway tweaks, etc.)_\n"
    )


# -- add-iris-job -----------------------------------------------------------


def _resolve_run_file_arg(arg: str) -> Path:
    """Accept a filename stem, full filename, path, or wandb run_id/name."""
    if "/" in arg or arg.endswith(".md"):
        p = Path(arg)
        if p.exists():
            return p.resolve()
        candidate = RUNS_DIR / Path(arg).name
        if candidate.exists():
            return candidate

    # Bare stem — try matching by filename stem first, then by W&B name / run_id.
    cands = list(RUNS_DIR.glob(f"{arg}.md"))
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        names = ", ".join(p.name for p in cands)
        raise SystemExit(f"Ambiguous run reference {arg!r}; matches: {names}")

    for rf in _existing_run_files():
        wandb = rf.metadata.get("wandb") or {}
        if arg in (wandb.get("run_id"), wandb.get("run_name")):
            return rf.path
    raise SystemExit(
        f"Could not find a run file matching {arg!r}. "
        "Pass the filename stem, the W&B run_id, or the W&B run name."
    )


def _cmd_add_iris_job(args: argparse.Namespace) -> int:
    path = _resolve_run_file_arg(args.run)
    rf = _read_run_file(path)
    jobs = list(rf.metadata.get("iris_job_ids") or [])
    if args.iris_job in jobs:
        print(f"{args.iris_job} already recorded in {path.relative_to(REPO_ROOT)}", file=sys.stderr)
        return 0
    jobs.append(args.iris_job)
    rf.metadata["iris_job_ids"] = jobs
    _write_run_file(rf)
    print(f"Appended iris job {args.iris_job} to {path.relative_to(REPO_ROOT)}")
    return 0


# -- update-index ---------------------------------------------------------


def _render_runs_md(files: list[RunFile]) -> str:
    rows: list[str] = []
    files_sorted = sorted(
        files, key=lambda rf: rf.launched_at or "", reverse=True,
    )
    for rf in files_sorted:
        m = rf.metadata
        wandb = m.get("wandb") or {}
        launched = (m.get("launched_at") or "")[:10] or "?"
        experiment = m.get("experiment") or "?"
        run_name = wandb.get("run_name") or "?"
        kind = m.get("kind") or "?"
        user = m.get("user") or "?"
        short = (m.get("short_description") or "").replace("|", "&#124;").replace("\n", " ")
        wandb_url = wandb.get("url") or ""
        wandb_cell = f"[{run_name}]({wandb_url})" if wandb_url else run_name
        details_rel = rf.path.relative_to(HISTORY_DIR).as_posix()
        details_cell = f"[md]({details_rel})"
        rows.append(
            f"| {launched} | `{experiment}` | {wandb_cell} | `{kind}` | {user} | {short} | {details_cell} |"
        )

    lines = [
        "# Runs",
        "",
        "Every W&B-logged run, sorted by launch date (newest first).",
        "",
        "_This page is regenerated by `marinfold history update-index`._",
        "_See [`README.md`](README.md) for the file schema + policy._",
        "",
    ]
    if not rows:
        lines.append("_(No runs recorded yet.)_")
    else:
        lines.append("| Date | Experiment | W&B run | Kind | User | Description | Details |")
        lines.append("|---|---|---|---|---|---|---|")
        lines.extend(rows)
    lines.append("")
    return "\n".join(lines)


def _cmd_update_index(args: argparse.Namespace) -> int:
    files = _existing_run_files()
    content = _render_runs_md(files)

    if args.check:
        current = RUNS_MD.read_text() if RUNS_MD.exists() else ""
        if current.strip() != content.strip():
            print(f"STALE: {RUNS_MD.relative_to(REPO_ROOT)}", file=sys.stderr)
            print("Regenerate with: marinfold history update-index", file=sys.stderr)
            return 1
        print(f"up to date: {RUNS_MD.relative_to(REPO_ROOT)}")
        return 0

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_MD.write_text(content)
    print(f"wrote {RUNS_MD.relative_to(REPO_ROOT)}")
    return 0


# -- sync (W&B-backed) ----------------------------------------------------


def _require_wandb() -> Any:
    try:
        import wandb  # type: ignore
    except ImportError:
        print(
            "wandb is not installed. From the experiments/ subproject, run:\n"
            "  uv sync --extra wandb",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return wandb


def _wandb_iter_runs(entity: str, project: str, limit: int | None, since_iso: str | None):
    wandb = _require_wandb()
    api = wandb.Api()
    filters: dict[str, Any] = {}
    if since_iso:
        filters["created_at"] = {"$gte": since_iso}
    runs = api.runs(f"{entity}/{project}", filters=filters or None, per_page=50)
    for i, run in enumerate(runs):
        if limit is not None and i >= limit:
            break
        yield run


def _experiment_from_wandb_run(run) -> str:
    """Best-effort: pull the experiment dir name out of a wandb run's metadata.

    We look at config + tags + group. If none looks like ``exp<N>_<kind>_<slug>``,
    fall back to ``no_experiment``.
    """
    exp_re = re.compile(r"^exp\d+_[a-z_]+$")
    config = dict(run.config) if hasattr(run, "config") else {}
    candidates = []
    candidates.append(config.get("experiment"))
    candidates.append(config.get("marinfold_experiment"))
    candidates.append(getattr(run, "group", None))
    candidates.extend(getattr(run, "tags", None) or [])
    for c in candidates:
        if isinstance(c, str) and exp_re.match(c):
            return c
    return "no_experiment"


def _kind_from_wandb_run(run, fallback_experiment: str) -> str:
    """Best-effort: prefer explicit ``config['kind']``, else derive from the experiment name.

    Since experiment kind and run kind now share the same taxonomy, a
    run that lives in ``exp<N>_<kind>_<slug>`` defaults to ``<kind>``.
    Runs not tied to an experiment default to ``other``.
    """
    config = dict(run.config) if hasattr(run, "config") else {}
    k = config.get("kind")
    if isinstance(k, str) and k in RUN_KINDS:
        return k
    parsed = parse_experiment_dir_name(fallback_experiment)
    return parsed[1] if parsed is not None else "other"


def _cmd_sync(args: argparse.Namespace) -> int:
    entity = args.entity
    project = args.project

    existing = _runs_by_id(_existing_run_files())
    print(
        f"[marinfold history] {len(existing)} existing run files; fetching W&B runs from "
        f"{entity}/{project}...",
        file=sys.stderr,
    )

    created = 0
    seen = 0
    for run in _wandb_iter_runs(entity, project, args.limit, args.since):
        seen += 1
        if run.id in existing:
            continue
        launched_at = getattr(run, "created_at", None) or datetime.now(timezone.utc).isoformat()
        if not launched_at.endswith("Z") and "+" not in launched_at[10:]:
            launched_at = launched_at + "Z"
        experiment = _experiment_from_wandb_run(run)
        kind = _kind_from_wandb_run(run, experiment)
        user = (getattr(run, "user", None) and run.user.username) or DEFAULT_WANDB_ENTITY
        config = dict(run.config) if hasattr(run, "config") else {}

        metadata: dict[str, Any] = {
            "user": user,
            "launched_at": launched_at,
            "experiment": experiment,
            "kind": kind,
            "short_description": config.get("notes") or run.notes or "",
            "wandb": {
                "url": run.url,
                "entity": entity,
                "project": project,
                "run_id": run.id,
                "run_name": run.name,
            },
            "git_sha": config.get("git_sha") or (run.commit if hasattr(run, "commit") else None) or "unknown",
            "iris_job_ids": [],
        }
        fname = _filename_for(
            launched_at=launched_at, experiment=experiment, wandb_run_name=run.name,
        )
        path = RUNS_DIR / fname
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        body = _default_body(metadata)
        _write_run_file(RunFile(path=path, metadata=metadata, body=body))
        created += 1
        print(f"  created {path.relative_to(REPO_ROOT)}", file=sys.stderr)

    print(
        f"[marinfold history] sync done. saw {seen} W&B runs, created {created} new history files.",
        file=sys.stderr,
    )
    return 0


# -- check ----------------------------------------------------------------


def _cmd_check(args: argparse.Namespace) -> int:
    entity = args.entity
    project = args.project
    existing = _runs_by_id(_existing_run_files())

    missing: list[tuple[str, str]] = []
    for run in _wandb_iter_runs(entity, project, args.limit, args.since):
        if run.id not in existing:
            missing.append((run.id, run.name))

    if not missing:
        print(f"[marinfold history] OK — all {len(existing)} W&B runs have history files.")
        return 0

    print(
        f"[marinfold history] DRIFT — {len(missing)} W&B run(s) lack a history file:",
        file=sys.stderr,
    )
    for run_id, run_name in missing:
        print(f"  - {run_name} ({run_id})", file=sys.stderr)
    print("Run `marinfold history sync` to create skeletons.", file=sys.stderr)
    return 1


# -- argparse -------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="marinfold history",
        description="Manage the MarinFold run history (history/runs/*.md).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_new = sub.add_parser("new", help="Create a new history file for a W&B run.")
    p_new.add_argument(
        "--wandb-url", required=True,
        help="Full https://wandb.ai/<entity>/<project>/runs/<run_id> URL.",
    )
    p_new.add_argument(
        "--wandb-name", required=True,
        help="W&B display run name (e.g. 'fuzzy_cloth'). Printed by wandb.init().",
    )
    p_new.add_argument(
        "--experiment", required=True,
        help="Experiment dir name (exp<N>_<kind>_<slug>), or 'no_experiment'.",
    )
    p_new.add_argument(
        "--kind", required=True, choices=sorted(RUN_KINDS),
        help="What this run does. Must match the experiment kind for in-experiment runs.",
    )
    p_new.add_argument(
        "--short", default=None,
        help="One-line description of the run.",
    )
    p_new.add_argument(
        "--iris-jobs", nargs="*", default=None,
        help="Iris job ID(s) at launch time. More can be added later via add-iris-job.",
    )
    p_new.add_argument(
        "--user", default=None,
        help="User who launched the run. Defaults to $USER.",
    )
    p_new.add_argument(
        "--launched-at", default=None,
        help="ISO-8601 UTC launch time. Defaults to now.",
    )
    p_new.add_argument(
        "--git-sha", default=None,
        help="Git SHA of the launching commit. Defaults to `git rev-parse HEAD`.",
    )
    p_new.add_argument("--force", action="store_true", help="Overwrite an existing entry for the same run_id.")
    p_new.set_defaults(func=_cmd_new)

    p_aij = sub.add_parser("add-iris-job", help="Append an iris job ID to an existing run.")
    p_aij.add_argument("run", help="Run filename stem, W&B run_id, or W&B run name.")
    p_aij.add_argument("iris_job", help="Iris job ID to append.")
    p_aij.set_defaults(func=_cmd_add_iris_job)

    p_idx = sub.add_parser("update-index", help="Regenerate history/RUNS.md.")
    p_idx.add_argument(
        "--check", action="store_true",
        help="Exit non-zero if the file is out of date; don't write.",
    )
    p_idx.set_defaults(func=_cmd_update_index)

    for name, helptxt, fn in [
        ("sync", "Pull runs from W&B, create skeletons for any without a history file.", _cmd_sync),
        ("check", "Exit non-zero if any W&B run lacks a history file.", _cmd_check),
    ]:
        sp = sub.add_parser(name, help=helptxt)
        sp.add_argument("--entity", default=DEFAULT_WANDB_ENTITY)
        sp.add_argument("--project", default=DEFAULT_WANDB_PROJECT)
        sp.add_argument("--limit", type=int, default=200, help="Max W&B runs to scan.")
        sp.add_argument("--since", default=None, help="ISO-8601 lower bound on created_at.")
        sp.set_defaults(func=fn)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
