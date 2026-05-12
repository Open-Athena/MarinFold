# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""``marinfold graduate`` — promote an experiment by symlinking it into its kind dir.

Usage::

    marinfold graduate exp42_models_protein_1b_distance_masked
    marinfold graduate experiments/exp42_models_protein_1b_distance_masked
    marinfold graduate --slug protein_1b_distance_masked_v2 exp42_models_protein_1b_distance_masked

Resolves the experiment's kind from its directory name
(``exp<N>_<kind>_<slug>``) and creates a symlink under the
corresponding top-level dir. The symlink's name defaults to the slug
(dropping the ``exp<N>_<kind>_`` prefix); ``--slug`` overrides it.

Symlink target is the relative path
``../experiments/exp<N>_<kind>_<slug>``. The experiment dir itself is
NOT moved or modified.

Idempotent: re-running with the same arguments either confirms the
existing symlink or refuses (use ``--force`` to repoint).
"""

import argparse
import os
import sys
from pathlib import Path

from marinfold_experiments._repo import REPO_ROOT, parse_experiment_dir_name


def _normalise_experiment_arg(arg: str) -> Path:
    """Accept either a bare dir name or a path; return the absolute experiment dir."""
    p = Path(arg)
    if p.is_absolute() or p.exists():
        return p.resolve()
    # Bare name → look under experiments/
    return (REPO_ROOT / "experiments" / arg).resolve()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="marinfold graduate")
    ap.add_argument(
        "experiment",
        help="Experiment dir name (e.g. exp42_models_foo) or path.",
    )
    ap.add_argument(
        "--slug", default=None,
        help="Override the symlink name in the kind dir. Defaults to "
             "the slug portion of the experiment dir name.",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Replace an existing symlink at the target name.",
    )
    args = ap.parse_args(argv)

    exp_path = _normalise_experiment_arg(args.experiment)
    if not exp_path.is_dir():
        print(f"Not a directory: {exp_path}", file=sys.stderr)
        return 1

    if exp_path.parent.resolve() != (REPO_ROOT / "experiments").resolve():
        print(
            f"Experiment must live directly under experiments/. Got: {exp_path}",
            file=sys.stderr,
        )
        return 1

    parsed = parse_experiment_dir_name(exp_path.name)
    if parsed is None:
        print(
            f"Could not parse experiment dir name {exp_path.name!r} — "
            "expected exp<N>_<kind>_<slug>.",
            file=sys.stderr,
        )
        return 1
    _n, kind, slug = parsed

    target_dir = REPO_ROOT / kind
    if not target_dir.is_dir():
        print(
            f"Kind dir does not exist: {target_dir}. "
            "Initialise it (with its pyproject.toml) before graduating.",
            file=sys.stderr,
        )
        return 1

    link_name = args.slug or slug
    link_path = target_dir / link_name
    relative_target = Path("..") / "experiments" / exp_path.name

    if link_path.exists() or link_path.is_symlink():
        if not args.force:
            existing_target = (
                os.readlink(link_path) if link_path.is_symlink() else "(non-symlink)"
            )
            print(
                f"Already exists: {link_path} -> {existing_target}\n"
                "Re-run with --force to repoint.",
                file=sys.stderr,
            )
            return 1
        link_path.unlink()

    link_path.symlink_to(relative_target, target_is_directory=True)
    print(f"Graduated {exp_path.name} -> {link_path.relative_to(REPO_ROOT)}")
    print(f"  (symlink target: {relative_target})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
