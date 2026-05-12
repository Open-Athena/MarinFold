# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""PM tooling for the MarinFold experiment system.

Single CLI ``marinfold`` with subcommands:

- ``scaffold`` — create an experiment dir from a GitHub issue.
- ``itemize`` — regenerate ``experiments/index.md``.
- ``graduate`` — symlink an experiment into its kind dir.
- ``history {new, add-iris-job, update-index, sync, check}`` — manage the run-history audit trail.

The four experiment **kinds** are shared across the experiment
directory naming (``exp<N>_<kind>_<slug>``) and the run-history
``kind:`` field. ``other`` is a fifth kind used only for runs that
don't belong to any of the first four (e.g. ad-hoc scratch runs).
"""

KINDS = ("models", "evals", "data", "document_structures")
"""Recognised experiment kinds. Match the top-level dir names exactly."""

RUN_KINDS = KINDS + ("other",)
"""Kinds valid in a run-history file. Adds ``other`` for runs outside
any experiment."""
