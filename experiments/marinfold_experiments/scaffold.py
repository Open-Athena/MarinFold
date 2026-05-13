# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""``marinfold scaffold`` — create a new experiment directory from a GitHub issue.

Usage::

    marinfold scaffold --issue 7 --kind models
    marinfold scaffold --issue 7 --kind document_structures --name my_name

Reads the issue via ``gh api`` (must be authenticated), derives a name
from the title, and creates ``experiments/exp<N>_<kind>_<name>/`` with
a README.md prefilled from the issue's prose.

The ``--kind`` argument is required if the issue doesn't declare one
(see the issue template's "Kind" field). Recognised kinds:
``models``, ``evals``, ``data``, ``document_structures``.

Does NOT overwrite an existing directory unless ``--force`` is passed.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from marinfold_experiments import KINDS
from marinfold_experiments._repo import REPO_ROOT, github_repo


def fetch_issue(number: int, github_repo: str) -> dict:
    out = subprocess.check_output(
        ["gh", "api", f"/repos/{github_repo}/issues/{number}"], text=True,
    )
    return json.loads(out)


def title_to_name(title: str) -> str:
    t = title.strip()
    for prefix in ("exp:", "experiment:"):
        if t.lower().startswith(prefix):
            t = t[len(prefix):].strip()
            break
    t = re.sub(r"[^\w\s-]", "", t)
    t = re.sub(r"[\s-]+", "_", t)
    t = t.strip("_").lower()
    parts = t.split("_")
    return "_".join(parts[:6]) or "experiment"


def extract_section(body: str, header_names: list[str]) -> str | None:
    """Return the body under the first matching ``## <name>`` header."""
    lines = body.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("## "):
            continue
        label = stripped[3:].strip().lower()
        if any(label.startswith(h.lower()) for h in header_names):
            out: list[str] = []
            for j in range(i + 1, len(lines)):
                nxt = lines[j].strip()
                if nxt.startswith("## ") or nxt.startswith("# "):
                    break
                out.append(lines[j])
            return _strip_template_placeholders("\n".join(out).strip())
    return None


def _strip_template_placeholders(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()


def _kind_from_issue_body(body: str) -> str | None:
    """Extract the ``Kind`` field from an issue body if present.

    Issue template renders it as ``- Kind: models`` or similar. Accept
    several casings.
    """
    m = re.search(r"(?im)^\s*[-*]?\s*kind\s*[:=]\s*([a-z_-]+)\s*$", body)
    if not m:
        return None
    raw = m.group(1).replace("-", "_").lower()
    return raw if raw in KINDS else None


def render_readme(*, issue: dict, kind: str, name: str, branch: str) -> str:
    body = issue.get("body") or ""

    question = extract_section(body, ["Question"]) or "_(Copy from the issue.)_"
    hypothesis = extract_section(body, ["Hypothesis"]) or "_(Copy from the issue.)_"
    background = extract_section(body, ["Background"]) or ""
    approach = extract_section(body, ["Approach"]) or "_(Outline what the experiment will do.)_"
    success = extract_section(body, ["Success criteria"]) or "_(Concrete metrics + thresholds.)_"

    safe_title = issue["title"].replace('"', '\\"')

    out: list[str] = []
    out.append("---\n")
    out.append("marinfold_experiment:\n")
    out.append(f"  issue: {issue['number']}\n")
    out.append(f"  title: \"{safe_title}\"\n")
    out.append(f"  kind: {kind}\n")
    out.append(f"  branch: {branch}\n")
    out.append("---\n\n")
    out.append(f"# {issue['title']}\n\n")
    out.append(f"**Issue:** [#{issue['number']}]({issue['html_url']}) · **Kind:** `{kind}` · **Branch:** `{branch}`\n\n")
    out.append("## Question\n\n" + question + "\n\n")
    out.append("## Hypothesis\n\n" + hypothesis + "\n\n")
    if background:
        out.append("## Background\n\n" + background + "\n\n")
    out.append("## Approach\n\n" + approach + "\n\n")
    out.append("## Success criteria\n\n" + success + "\n\n")
    out.append("## Results\n\n_(Fill in after the run completes.)_\n\n")
    out.append("## Conclusion\n\n_(Fill in after results are in.)_\n")
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="marinfold scaffold")
    ap.add_argument("--issue", type=int, required=True, help="GitHub issue number")
    ap.add_argument(
        "--kind", default=None,
        choices=sorted(KINDS),
        help=(
            "Experiment kind. If not passed, attempts to read 'Kind:' from "
            "the issue body."
        ),
    )
    ap.add_argument("--name", default=None, help="Override the auto-derived name")
    ap.add_argument(
        "--branch", default="main",
        help="Branch the experiment lives on (default: main; use exp/<N>-<name> for speculative work)",
    )
    ap.add_argument(
        "--repo", default=None,
        help="GitHub repo (owner/name). Defaults to the origin remote, then Open-Athena/MarinFold.",
    )
    ap.add_argument("--force", action="store_true", help="Clobber an existing README.md")
    args = ap.parse_args(argv)

    repo = args.repo or github_repo()
    issue = fetch_issue(args.issue, repo)

    kind = args.kind or _kind_from_issue_body(issue.get("body") or "")
    if not kind:
        print(
            "Could not determine experiment kind. Pass --kind, or add "
            "'Kind: <models|evals|data|document_structures>' to the issue body.",
            file=sys.stderr,
        )
        return 1

    name = args.name or title_to_name(issue["title"])

    existing = sorted(
        p for p in (REPO_ROOT / "experiments").glob(f"exp{args.issue}_*")
        if p.is_dir()
    )
    if existing and not args.force:
        existing_names = ", ".join(p.name for p in existing)
        print(f"Experiment for issue #{args.issue} already exists: {existing_names}", file=sys.stderr)
        print("Re-run with --force to clobber README.md, or edit the existing dir.", file=sys.stderr)
        return 1

    exp_dir = REPO_ROOT / "experiments" / f"exp{args.issue}_{kind}_{name}"
    readme = exp_dir / "README.md"

    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "data").mkdir(exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)

    readme.write_text(render_readme(issue=issue, kind=kind, name=name, branch=args.branch))

    rel = readme.relative_to(REPO_ROOT)
    print(f"Scaffolded {rel}")
    print("Next steps:")
    print(f"  1. Edit {rel}: fill in the approach and success criteria")
    print("  2. Add launchable .py files in the experiment dir (and a pyproject.toml if it needs marin deps)")
    print( "  3. When results land, fill in Results + Conclusion, then 'marinfold graduate' if appropriate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
