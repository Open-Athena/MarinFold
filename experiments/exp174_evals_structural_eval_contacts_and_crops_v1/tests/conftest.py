# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Put the experiment dir on ``sys.path`` so tests import its modules directly.

Same idiom as exp53 / exp139. ``marinfold`` itself comes from the venv (an
editable install of the in-repo package, see ``pyproject.toml``).
"""

import sys
from pathlib import Path

_EXP_DIR = Path(__file__).resolve().parent.parent

if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))
