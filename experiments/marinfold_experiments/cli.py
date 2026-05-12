# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""``marinfold`` — single dispatcher for the experiment-management CLIs.

Each subcommand delegates to its module's ``main(argv)`` so the
modules stay independently runnable (e.g. via ``python -m
marinfold_experiments.scaffold --issue 7``) and the dispatcher
itself is a tiny lookup table.

Subcommands:

- ``scaffold``  — create an experiment dir from a GitHub issue
- ``itemize``   — regenerate experiments/index.md
- ``graduate``  — symlink an experiment into its kind dir
- ``history``   — manage the run-history audit trail (itself a subcommand)
"""

import sys

from marinfold_experiments import graduate, history, itemize, scaffold


_COMMANDS = {
    "scaffold": scaffold.main,
    "itemize": itemize.main,
    "graduate": graduate.main,
    "history": history.main,
}

_USAGE = f"""usage: marinfold <command> [<args>...]

Commands:
  scaffold   Create an experiment dir from a GitHub issue.
  itemize    Regenerate experiments/index.md from gh + frontmatter.
  graduate   Symlink an experiment into its kind dir.
  history    Manage the run-history audit trail (subcommand).

Pass `--help` to any command for details, e.g. `marinfold scaffold --help`.
""".strip()


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(_USAGE)
        return 0
    cmd, *rest = argv
    func = _COMMANDS.get(cmd)
    if func is None:
        print(f"unknown command: {cmd!r}\n\n{_USAGE}", file=sys.stderr)
        return 2
    return func(rest)


if __name__ == "__main__":
    sys.exit(main())
