# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Test config: import the *local* marinfold (this branch) and this exp's modules.

The experiment's ``pyproject.toml`` resolves ``marinfold`` from git (so the same
env reproduces on remote Iris workers), but the reusable-contacts helper lives
on this branch and isn't published yet. For local unit tests we put the
in-repo ``marinfold`` package and this experiment dir on ``sys.path`` so the
worker imports resolve without a full ``uv sync``.
"""

import sys
from pathlib import Path

_EXP_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _EXP_DIR.parent.parent          # experiments/expNN/../.. == repo root
_MARINFOLD = _REPO_ROOT / "marinfold"        # the package dir (has marinfold/marinfold)

for p in (str(_EXP_DIR), str(_MARINFOLD)):
    if p not in sys.path:
        sys.path.insert(0, p)
