# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the impl's ``plots.py`` PDF writers.

These tests don't run a model. They build tiny synthetic predict /
evaluate outputs by hand and assert that the corresponding PDF writer
produces a non-empty file with the expected page count.
"""

from pathlib import Path

import pytest

# Skip the whole module cleanly when matplotlib isn't installed (i.e.
# the ``[contacts-and-distances-v1]`` extra wasn't synced). CI that
# installs the extra will exercise the assertions.
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless backend

from marinfold import EvalResult  # noqa: E402
from marinfold.document_structures.contacts_and_distances_v1 import (  # noqa: E402
    plot_evaluate_pdf,
    plot_infer_pdf,
)


def _read_pdf_page_count(pdf_path: Path) -> int:
    """Page count via pypdf if installed, else count ``/Type /Page`` markers.

    The marker count is good enough for a smoke test — matplotlib's
    PdfPages writes one page object per page, and there's no other
    source of ``/Type /Page`` in these PDFs.
    """
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(pdf_path)).pages)
    except ImportError:
        blob = pdf_path.read_bytes()
        return blob.count(b"/Type /Page") - blob.count(b"/Type /Pages")


def test_plot_infer_pdf_writes_one_page_per_record(tmp_path: Path) -> None:
    records = [
        {
            "entry_id": "AF-ABC123-F1",
            "n_residues": 5,
            "n_seeded": 0,
            "query_atom": "CA",
            "pairs": [(1, 2), (1, 3), (2, 4), (3, 5)],
            "expected_distances": [4.0, 7.5, 10.2, 15.1],
        },
        {
            "entry_id": "AF-ABC123-F1",
            "n_residues": 5,
            "n_seeded": 5,
            "query_atom": "CA",
            "pairs": [(1, 2), (1, 3), (2, 4), (3, 5)],
            "expected_distances": [4.1, 7.6, 10.0, 14.8],
        },
    ]
    out_pdf = tmp_path / "infer.pdf"
    plot_infer_pdf(out_pdf, records)
    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 0
    assert _read_pdf_page_count(out_pdf) == 2


def test_plot_evaluate_pdf_writes_one_page_per_structure_n_seeded(
    tmp_path: Path,
) -> None:
    # Two structures, two seed-N values: expect 4 pages.
    per_example = []
    for entry_id, n in [("AF-A-F1", 5), ("AF-B-F1", 6)]:
        for n_seeded in (0, 5):
            for i in range(1, n):
                for j in range(i + 1, n + 1):
                    gt = float(j - i)
                    pred = float(j - i) + 0.5
                    per_example.append({
                        "entry_id": entry_id,
                        "n_seeded": n_seeded,
                        "i": i,
                        "j": j,
                        "gt_angstrom": gt,
                        "expected_angstrom": pred,
                        "abs_err_angstrom": abs(pred - gt),
                    })
    result = EvalResult(
        metrics={"mae_at_n0_angstrom": 0.5, "mae_at_n5_angstrom": 0.5},
        per_example=per_example,
        extras={
            "structure": "contacts-and-distances-v1",
            "model": "dummy",
            "backend": "transformers",
            "query_atom": "CA",
            "seed_n_values": [0, 5],
            "n_structures": 2,
            "per_structure_mae": {
                0: {"AF-A-F1": 0.5, "AF-B-F1": 0.5},
                5: {"AF-A-F1": 0.5, "AF-B-F1": 0.5},
            },
            "per_structure_n_pairs": {
                0: {"AF-A-F1": 10, "AF-B-F1": 15},
                5: {"AF-A-F1": 10, "AF-B-F1": 15},
            },
            "per_structure_n_residues": {"AF-A-F1": 5, "AF-B-F1": 6},
        },
    )
    out_pdf = tmp_path / "eval.pdf"
    plot_evaluate_pdf(out_pdf, result)
    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 0
    assert _read_pdf_page_count(out_pdf) == 4


def test_plot_evaluate_pdf_tolerates_string_mae_keys(tmp_path: Path) -> None:
    """``extras`` round-tripped through JSON gets int keys stringified —
    the writer should still find the MAE entry and produce a page."""
    per_example = [
        {"entry_id": "X", "n_seeded": 0, "i": 1, "j": 3,
         "gt_angstrom": 8.0, "expected_angstrom": 8.1, "abs_err_angstrom": 0.1},
    ]
    result = EvalResult(
        metrics={"mae_at_n0_angstrom": 0.1},
        per_example=per_example,
        extras={
            "query_atom": "CA",
            "seed_n_values": [0],
            "per_structure_mae": {"0": {"X": 0.1}},  # stringified key
            "per_structure_n_pairs": {"0": {"X": 1}},
            "per_structure_n_residues": {"X": 3},
        },
    )
    out_pdf = tmp_path / "eval_stringkeys.pdf"
    plot_evaluate_pdf(out_pdf, result)
    assert _read_pdf_page_count(out_pdf) == 1
