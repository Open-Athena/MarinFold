"""End-to-end invariants for the saved confidence-ranking experiment."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"


def test_all_archived_structures_and_contact_degrees_reproduce() -> None:
    frame = pd.read_csv(DATA/"contact_confidence_validation.csv")
    assert len(frame) == 333*7
    assert not frame.duplicated(["stem","method"]).any()
    assert (frame.contact_degree_max_difference == 0).all()
    assert (frame.contact_support_difference == 0).all()


def test_natural_degree_control_and_fixed_denominator() -> None:
    metrics = pd.read_csv(DATA/"contact_confidence_metrics.csv")
    targets = pd.read_csv(DATA/"targets.csv")
    frame = metrics.merge(targets[["stem","designed"]],on="stem",validate="many_to_one")
    natural = frame[(frame.designed==0)&(frame.ranking=="degree")]
    assert len(natural)==314*7*2
    np.testing.assert_allclose(natural.precision,natural.original_precision,atol=1e-12,rtol=0)
    np.testing.assert_allclose(frame.precision,frame.n_correct/frame.k,atol=1e-12,rtol=0)
    assert (frame.k == np.maximum(1, frame.L//5)).all()
    assert (frame.n_selected == np.minimum(frame.k,frame.n_positive)).all()


def test_all_predictors_matched_and_marinfold_knn_unchanged() -> None:
    rows = pd.read_csv(DATA/"contact_confidence_figure_rows.csv",float_precision="round_trip")
    old = pd.read_csv(DATA/"neff_figure_rows.csv",float_precision="round_trip")
    old = old[old.figure=="04_contacts_pl5_neff"]
    for (_, _), group in rows.groupby(["figure","metric"]):
        assert group.groupby("method").stem.nunique().to_dict()=={m:333 for m in old.method.unique()}
        control = group[group.method.isin(["marinfold","knn"])].merge(old,on=["stem","method","metric"],suffixes=("_new","_old"))
        np.testing.assert_array_equal(control.value_new,control.value_old)
