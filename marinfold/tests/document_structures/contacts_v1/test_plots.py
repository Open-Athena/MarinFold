# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the contacts-v1 ``plots.py`` PDF writers.

No model is run: tiny synthetic predict / evaluate outputs are built by hand
and the writers are asserted to produce a non-empty PDF with the expected
page count (one per structure).
"""

from pathlib import Path

import pytest

# matplotlib is a base marinfold dependency, but importorskip keeps the
# module green in a stripped environment.
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless backend

from marinfold import EvalResult  # noqa: E402
from marinfold.document_structures.contacts_v1 import (  # noqa: E402
    plot_evaluate_pdf,
    plot_infer_pdf,
)


def _read_pdf_page_count(pdf_path: Path) -> int:
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(pdf_path)).pages)
    except ImportError:
        blob = pdf_path.read_bytes()
        return blob.count(b"/Type /Page") - blob.count(b"/Type /Pages")


def _infer_record(entry_id: str, n: int) -> dict:
    pairs = [[i, j] for i in range(1, n + 1) for j in range(i + 6, n + 1)]
    return {
        "entry_id": entry_id,
        "n_residues": n,
        "min_seq_separation": 6,
        "ensemble_k": 1,
        "pairs": pairs,
        "p_contact": [0.1 * ((i + j) % 5) for (i, j) in pairs],
    }


def test_plot_infer_pdf_writes_one_page_per_record(tmp_path: Path) -> None:
    records = [_infer_record("AF-A-F1", 14), _infer_record("AF-B-F1", 16)]
    out_pdf = tmp_path / "infer.pdf"
    plot_infer_pdf(out_pdf, records)
    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 0
    assert _read_pdf_page_count(out_pdf) == 2


def test_plot_evaluate_pdf_writes_one_page_per_structure(tmp_path: Path) -> None:
    per_example = []
    n_by_entry = {}
    for entry_id, n in [("AF-A-F1", 14), ("AF-B-F1", 16)]:
        n_by_entry[entry_id] = n
        for i in range(1, n + 1):
            for j in range(i + 6, n + 1):
                per_example.append({
                    "entry_id": entry_id,
                    "i": i,
                    "j": j,
                    "p_contact": 0.1 * ((i + j) % 5),
                    "gt": int((j - i) == 6),
                })
    result = EvalResult(
        metrics={"auc_long": 0.9},
        per_example=per_example,
        extras={
            "structure": "contacts-v1",
            "model": "dummy",
            "per_structure_n_residues": n_by_entry,
        },
    )
    out_pdf = tmp_path / "eval.pdf"
    plot_evaluate_pdf(out_pdf, result)
    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 0
    assert _read_pdf_page_count(out_pdf) == 2
