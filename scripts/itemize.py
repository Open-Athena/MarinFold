# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""``scripts/itemize.py`` — regenerate experiments/index.md from gh + frontmatter.

Source of truth for WHICH experiments exist is
``gh issue list --label experiment``. The index groups experiments by
**kind**. An experiment's kind comes from its
``experiments/exp<N>_<kind>_<name>/`` dir (README ``marinfold_experiment:``
frontmatter, else the dir name) when one exists; otherwise from the
issue's ``kind/<kind>`` label. Dir-less issues with no ``kind/<kind>``
label land under "Unclassified". Per-experiment metadata (title, branch)
comes from the frontmatter when present, falling back to the issue title.

If a dir-backed issue carries a ``kind/<kind>`` label that disagrees with
its dir, the dir wins and a warning is printed (fix the stale label).

Experiment dirs without a matching `experiment`-labeled issue are listed
under "Orphans" so we notice and fix the naming.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    KIND_DESCRIPTIONS,
    KINDS,
    REPO_ROOT,
    github_repo,
    kind_from_labels,
    parse_experiment_dir_name,
    read_frontmatter,
)


def list_experiment_issues(repo: str) -> list[dict]:
    out = subprocess.check_output(
        [
            "gh", "issue", "list",
            "--repo", repo,
            "--label", "experiment",
            "--state", "all",
            "--limit", "200",
            "--json", "number,title,state,url,createdAt,closedAt,labels",
        ],
        text=True,
    )
    return json.loads(out)


def scan_experiment_dirs() -> dict[int, Path]:
    """Map issue number -> experiments/exp<N>_<kind>_<name>/ directory."""
    out: dict[int, Path] = {}
    exp_root = REPO_ROOT / "experiments"
    if not exp_root.is_dir():
        return out
    # Sort so the winner on a collision is deterministic (not filesystem order).
    for p in sorted(exp_root.iterdir()):
        if not p.is_dir():
            continue
        parsed = parse_experiment_dir_name(p.name)
        if parsed is None:
            continue
        n, _kind, _name = parsed
        if n in out:
            # Two dirs parse to the same issue number. The index renders one
            # dir per issue and the orphan pass can't see the loser (its issue
            # IS matched), so a silent overwrite would hide the drift this tool
            # exists to surface. Warn and keep the first (sorted) dir.
            print(
                f"WARNING: multiple experiment dirs for issue #{n}: "
                f"{out[n].name} and {p.name}; keeping {out[n].name}. "
                f"Rename or remove the duplicate.",
                file=sys.stderr,
            )
            continue
        out[n] = p
    return out


def _dir_kind(exp_dir: Path) -> str | None:
    """Kind of a dir-backed experiment: README frontmatter, else dir name."""
    fm = read_frontmatter(exp_dir / "README.md") or {}
    kind = fm.get("kind")
    if kind:
        return kind
    parsed = parse_experiment_dir_name(exp_dir.name)
    return parsed[1] if parsed else None


def _render_row(issue: dict, exp_dir: Path | None, repo: str) -> str:
    number = issue["number"]
    issue_url = issue["url"]
    title = issue["title"].replace("|", "&#124;")
    state = issue["state"].lower()
    if exp_dir is not None:
        dir_name = exp_dir.name
        nb_url = f"https://github.com/{repo}/blob/main/experiments/{dir_name}/README.md"
        fm = read_frontmatter(exp_dir / "README.md") or {}
        branch = fm.get("branch") or "?"
        fm_title = fm.get("title")
        display_title = fm_title or title
        dir_cell = f"[`{dir_name}`]({nb_url})"
    else:
        display_title = title
        branch = "—"
        dir_cell = "_no dir yet_"
    return f"| [#{number}]({issue_url}) | {state} | {display_title} | `{branch}` | {dir_cell} |"


def _render_orphan_row(exp_dir: Path, repo: str) -> str:
    dir_name = exp_dir.name
    fm = read_frontmatter(exp_dir / "README.md") or {}
    title = fm.get("title", dir_name)
    issue_num = fm.get("issue")
    branch = fm.get("branch", "?")
    kind = fm.get("kind", "?")
    nb_url = f"https://github.com/{repo}/blob/main/experiments/{dir_name}/README.md"
    issue_cell = f"#{issue_num}" if issue_num else "—"
    return f"| {issue_cell} | {title} | `{kind}` | `{branch}` | [`{dir_name}`]({nb_url}) |"


KIND_TABLE_HEADER = "| Issue | State | Title | Branch | Directory |\n|---|---|---|---|---|"
ORPHAN_TABLE_HEADER = "| Issue | Title | Kind | Branch | Directory |\n|---|---|---|---|---|"


def _resolve_kind(issue: dict, exp_dir: Path | None) -> tuple[str | None, str | None]:
    """Resolve an experiment's kind, returning ``(kind, warning)``.

    A dir-backed experiment's kind comes from its dir (frontmatter / dir
    name) — authoritative, since the dir name physically encodes it. A
    dir-less experiment's kind comes from its ``kind/<kind>`` label. If a
    dir-backed issue *also* carries a label that disagrees with the dir,
    the dir wins and we surface a warning so the stale label gets fixed.
    """
    label_kind = kind_from_labels([lbl["name"] for lbl in issue.get("labels", [])])
    if exp_dir is not None:
        dir_kind = _dir_kind(exp_dir)
        warning = None
        if label_kind and dir_kind and label_kind != dir_kind:
            warning = (
                f"#{issue['number']}: kind/{label_kind} label disagrees with "
                f"dir kind `{dir_kind}` ({exp_dir.name}); using the dir"
            )
        return dir_kind, warning
    return label_kind, None


def render_index(
    issues: list[dict], exp_dirs: dict[int, Path], repo: str
) -> tuple[str, list[str]]:
    by_kind: dict[str, list[str]] = {k: [] for k in KINDS}
    unclassified_rows: list[str] = []
    warnings: list[str] = []
    matched: set[int] = set()
    for issue in sorted(issues, key=lambda i: -int(i["number"])):
        matched.add(issue["number"])
        exp_dir = exp_dirs.get(issue["number"])
        kind, warning = _resolve_kind(issue, exp_dir)
        if warning:
            warnings.append(warning)
        row = _render_row(issue, exp_dir, repo)
        if kind in by_kind:
            by_kind[kind].append(row)
        else:
            unclassified_rows.append(row)
    orphan_rows = [
        _render_orphan_row(p, repo)
        for n, p in sorted(exp_dirs.items())
        if n not in matched
    ]

    total = sum(len(v) for v in by_kind.values()) + len(unclassified_rows)
    lines = [
        "# Experiments",
        "",
        "Each row is a GitHub issue tagged `experiment`, grouped by **kind**.",
        "An experiment's kind comes from its `experiments/exp<N>_<kind>_<name>/`",
        "dir when one exists, otherwise from the issue's `kind/<kind>` label.",
        f"Kinds: {', '.join(f'`{k}`' for k in KINDS)}.",
        "",
        f"_{total} experiments. This page is regenerated by `python scripts/itemize.py`._",
        "",
    ]
    for kind in KINDS:
        rows = by_kind[kind]
        lines.append(f"## `{kind}` — {KIND_DESCRIPTIONS[kind]} ({len(rows)})")
        lines.append("")
        if rows:
            lines.append(KIND_TABLE_HEADER)
            lines.extend(rows)
        else:
            lines.append("_(none yet.)_")
        lines.append("")
    if unclassified_rows:
        lines.extend([
            "## Unclassified",
            "",
            "Dir-less issues with no `kind/<kind>` label. Add one so they land",
            "under a kind above.",
            "",
            KIND_TABLE_HEADER,
        ])
        lines.extend(unclassified_rows)
        lines.append("")
    if orphan_rows:
        lines.extend([
            "## Orphans",
            "",
            "Dirs that don't match any issue labelled `experiment`.",
            "Either file the issue or fix the dir name.",
            "",
            ORPHAN_TABLE_HEADER,
        ])
        lines.extend(orphan_rows)
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n", warnings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python scripts/itemize.py")
    ap.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "experiments" / "index.md",
        help="Output path",
    )
    ap.add_argument(
        "--repo", default=None,
        help="GitHub repo (owner/name). Defaults to the origin remote, then Open-Athena/MarinFold.",
    )
    ap.add_argument(
        "--check", action="store_true",
        help="Exit non-zero if the existing file is out of date; don't write.",
    )
    args = ap.parse_args(argv)

    repo = args.repo or github_repo()
    issues = list_experiment_issues(repo)
    exp_dirs = scan_experiment_dirs()
    content, warnings = render_index(issues, exp_dirs, repo)
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)

    if args.check:
        current = args.output.read_text() if args.output.exists() else ""
        if current.strip() != content.strip():
            print(f"STALE: {args.output}", file=sys.stderr)
            print("Regenerate with: python scripts/itemize.py", file=sys.stderr)
            return 1
        print(f"up to date: {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
