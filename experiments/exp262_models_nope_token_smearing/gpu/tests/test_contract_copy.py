# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp262's copy of the exp232 contract must not drift from the original.

The copy exists for a packaging reason (Iris bundles the working directory, and
this project must be submittable alone), not a scientific one. exp262's entire
claim is that only the architecture changed, so the copy has to stay identical
to the file it was taken from — everywhere except the docstring that says so.
"""

import ast
from pathlib import Path

GPU = Path(__file__).resolve().parents[1]
ORIGINAL = GPU.parents[1] / "exp232_sweep_cv1_decontam" / "training_contract.py"
COPY = GPU / "exp232_contract.py"


def _body_without_docstring(path: Path) -> str:
    """The module's source with its leading docstring removed."""
    source = path.read_text()
    tree = ast.parse(source)
    first = tree.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        lines = source.splitlines(keepends=True)
        return "".join(lines[first.end_lineno :])
    return source


def test_copy_is_byte_identical_to_exp232s_contract():
    assert ORIGINAL.is_file(), f"exp232's contract is missing at {ORIGINAL}"
    assert _body_without_docstring(COPY) == _body_without_docstring(ORIGINAL), (
        "exp262's copy of the exp232 contract has drifted from the original. "
        "Re-copy it rather than editing here — the comparison is only clean while "
        "the two are the same file."
    )


def test_the_copy_says_it_is_a_copy():
    assert "VERBATIM COPY" in COPY.read_text().split('"""')[1]
