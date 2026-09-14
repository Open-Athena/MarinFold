# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Local, deterministic fixtures; no remote corpora or credentials needed."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "marinfold"))

from experiments.exp279_models_exact_soft_contact_targets.targets import (  # noqa: E402
    Vocabulary,
)


@pytest.fixture
def vocab():
    return Vocabulary(size=2845, position_ids=tuple(range(143, 2143)))


def make_document(edges, *, residues=None):
    if residues is None:
        residues = sorted({v for edge in edges for v in edge}) or [143, 144]
    tokens = [2, 8]
    for residue in residues:
        tokens.extend([residue, 86])
    tokens.extend([3, 2141, 4, 2142, 9])
    for a, b in edges:
        tokens.extend([5, a, b])
    return np.asarray(tokens + [10, 1], dtype=np.int32)
