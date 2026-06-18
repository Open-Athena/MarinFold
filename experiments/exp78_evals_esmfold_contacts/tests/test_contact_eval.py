# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the exp78 scoring + combine + plot pipeline.

The pyconfind / structure-extraction path is shared verbatim with exp74
(``pyconfind_contacts.py``) and exercised there; here we test the bits new
to exp78: the precision metric matrices, the model-agnostic row shape, and
that ``combine_scores`` + ``plot`` run end-to-end on a synthetic table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import contact_eval as ce  # noqa: E402


def test_true_matrix_applies_degree_and_separation():
    # (i, j, degree): only (0,6,...) and (2,10,...) clear both degree>=0.001
    # and sep>=6; (0,3) is too close, (1,8) is below degree.
    contacts = [(0, 6, 0.5), (0, 3, 0.9), (1, 8, 0.0005), (2, 10, 0.01)]
    m = ce._true_matrix(L=12, contacts=contacts)
    assert m[0, 6] and m[2, 10]
    assert not m[0, 3]   # sep 3 < 6
    assert not m[1, 8]   # degree 0.0005 < 0.001
    assert int(m.sum()) == 2


def test_degree_matrix_keeps_max_degree():
    m = ce._degree_matrix(L=10, contacts=[(0, 5, 0.2), (0, 5, 0.7), (1, 9, 0.3)])
    assert m[0, 5] == 0.7
    assert m[1, 9] == 0.3


def test_precision_rows_perfect_ranker():
    # L=10, true contacts at the highest-scored pairs -> precision 1.0 at the
    # cuts that fit, and R-precision exactly 1.0 (ceiling) in the aggregate.
    L = 10
    resolved = np.arange(L)
    pair_i, pair_j, pair_sep = ce._resolved_pairs(resolved)
    true_mat = np.zeros((L, L), dtype=bool)
    true_pairs = [(0, 6), (1, 8), (2, 9)]
    for i, j in true_pairs:
        true_mat[i, j] = True
    score = np.zeros((L, L))
    for rank, (i, j) in enumerate(true_pairs):
        score[i, j] = 100 - rank  # true pairs are the top-scored
    rows = ce._precision_rows(score=score, true_mat=true_mat,
                              pair_i=pair_i, pair_j=pair_j, pair_sep=pair_sep, L=L)
    agg = {r["cut"]: r for r in rows if r["range"] == "all"}
    assert agg["R"]["precision"] == pytest.approx(1.0)   # R-precision ceiling
    assert agg["R"]["n_true"] == 3
    # All 3 true contacts are long-range (sep>=24? no: sep 6/7/7) -> short/med.
    assert agg["L/5"]["precision"] == pytest.approx(1.0)  # top-2 are true


def test_evaluate_protein_row_shape(tmp_path, monkeypatch):
    """evaluate_protein stamps model/mode/predictor and skips absent models."""
    # Stub compute_contacts so we don't need real structures: GT has two
    # true long-range contacts; the 'esmfold' prediction recovers one.
    from pyconfind_contacts import ContactResult

    def fake_compute(structure, input_seq, *, stem, prefer_chain=None):
        if "gt" in str(structure):
            contacts = ((0, 30, 0.9), (5, 40, 0.9))
        else:
            contacts = ((0, 30, 0.8),)  # esmfold recovers one
        return ContactResult(
            stem=stem, chain="A", n_input_residues=len(input_seq),
            n_resolved_residues=len(input_seq), n_mapped_residues=len(input_seq),
            alignment_identity=1.0, resolved_positions=tuple(range(len(input_seq))),
            contacts=contacts,
        )

    monkeypatch.setattr(ce, "compute_contacts", fake_compute)

    seq = "A" * 50
    gt = tmp_path / "gt.cif"
    gt.write_text("x")
    pred_root = tmp_path / "pred"
    (pred_root / "esmfold" / "p1").mkdir(parents=True)
    (pred_root / "esmfold" / "p1" / "structure.cif").write_text("x")
    # esmfold2 prediction absent -> should be skipped, present flag 0.

    rows, raw, meta = ce.evaluate_protein(
        stem="p1", input_seq=seq, gt_cif=gt, gt_chain="A",
        pred_root=pred_root, models=("esmfold", "esmfold2"),
    )
    assert meta["esmfold_present"] == 1
    assert meta["esmfold2_present"] == 0
    assert meta["n_true_contacts"] == 2
    assert {r["model"] for r in rows} == {"esmfold"}
    assert all(r["predictor"] == "structure" and r["mode"] == "single_seq" for r in rows)
    # GT raw rows carry model="na"; pred raw rows carry the model name.
    assert any(r["role"] == "gt" and r["model"] == "na" for r in raw)
    assert any(r["role"] == "pred" and r["model"] == "esmfold" for r in raw)


def _synthetic_precision_csv(path: Path, *, model: str, dataset: str) -> None:
    rows = []
    rng = np.random.default_rng(0)
    for stem_i in range(4):
        for rng_name in ce.RANGES:
            for cut, _ in ce.CUTS:
                rows.append(dict(
                    dataset=dataset, stem=f"{dataset}_{stem_i}",
                    neff_tier="deep", fold_verdict="same_fold",
                    model=model, mode="single_seq", predictor="structure",
                    range=rng_name, cut=cut, precision=float(rng.random()),
                    n_candidate=100, n_true=10, n_top=10,
                ))
    pd.DataFrame(rows).to_csv(path, index=False)


def test_combine_and_plot_end_to_end(tmp_path):
    import combine_scores
    import plot

    d_fold = tmp_path / "fold"; d_fold.mkdir()
    d_e65 = tmp_path / "e65"; d_e65.mkdir()
    _synthetic_precision_csv(d_fold / "contact_precision.csv", model="esmfold", dataset="foldbench100")
    _synthetic_precision_csv(d_e65 / "contact_precision.csv", model="esmfold2", dataset="denovo_pdb")
    # minimal meta so combine's _concat_csv("contact_eval_meta.csv") succeeds
    for d in (d_fold, d_e65):
        pd.DataFrame([dict(dataset="x", stem="x_0", L=50)]).to_csv(d / "contact_eval_meta.csv", index=False)

    # a fake exp74 protenix precision table (no `model` column -> auto-stamped)
    prot = tmp_path / "protenix.csv"
    pdf = pd.read_csv(d_fold / "contact_precision.csv").drop(columns=["model"])
    pdf["mode"] = "msa"
    pdf.to_csv(prot, index=False)

    out = tmp_path / "out"
    combine_scores.combine([d_fold, d_e65], out, prot)
    combined = pd.read_csv(out / "contact_precision_all.csv")
    assert set(combined["model"].unique()) == {"esmfold", "esmfold2", "protenix-v2"}

    plots = tmp_path / "plots"
    plot.main(out / "contact_precision_all.csv", plots)
    assert (plots / "contacts_at_R_by_model_and_range.png").exists()
    assert (plots / "contacts_at_L_by_model_and_range.png").exists()
