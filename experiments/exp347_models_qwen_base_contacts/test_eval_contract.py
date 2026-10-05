"""Protect the resolved-pair universe and coordinate mapping used for R-precision."""

import json
from pathlib import Path

import numpy as np

from eval_contract import (
    contact_votes,
    load_metric_reference,
    rollout_prompt,
    score_votes,
)

HERE = Path(__file__).resolve().parent


def test_live_contacts_wrap_retract_and_vote_once() -> None:
    text = (
        "<contact> <p1998> <p5> <contact> <p5> <p1998> "
        "<contact> <p1998> <p6> <retract> <p6> <p1998> "
        "<contact> <p1998> <p4> <contact> <p1998> <p2005> <end>"
    )
    votes = contact_votes(
        [text, "<contact> <p12> <p19> <end>"], [1998, 12], 8, "contacts_v1"
    )
    assert votes[0, 7] == votes[7, 0] == 2
    assert votes[0, 6] == 1
    assert votes.sum() == 6


def test_prompted_coordinates_and_duplicate_filter() -> None:
    votes = contact_votes(["1 8\n8 1\n0 7\n1 7\n1 6\n1 9\nEND\n"], [0], 8, "prompted")
    assert votes[0, 7] == votes[0, 6] == 1
    assert votes.sum() == 4


def test_metric_excludes_unresolved_residues() -> None:
    reference = load_metric_reference(
        HERE.parent / "exp89_evals_contacts_v1_model_on_eval_set/compute_metrics.py"
    )
    record = {"L": 10, "resolved": [0, 1, 7, 9], "contacts": [[0, 7, 0.1]]}
    votes = np.zeros((10, 10), dtype=int)
    votes[0, 8] = votes[8, 0] = 100  # Highest score is outside the frozen universe.
    votes[0, 7] = votes[7, 0] = 1
    rows = score_votes(votes, record, reference)
    all_r = next(r for r in rows if r["range"] == "all" and r["cut"] == "R")
    assert all_r["precision"] == 1.0
    assert all_r["n_candidate"] == 4
    assert len(rows) == 20


def test_frozen_eval_val_and_fresh_realizations() -> None:
    records = [
        json.loads(line)
        for line in (HERE / "data/eval_val.jsonl").read_text().splitlines()
    ]
    assert len(records) == len({r["stem"] for r in records}) == 97
    assert {r["eval_set"] for r in records} == {"eval-val"}
    record = records[0]
    a = rollout_prompt(record, "contacts_v1", 0)
    b = rollout_prompt(record, "contacts_v1", 1)
    assert a != b
    assert a == rollout_prompt(record, "contacts_v1", 0)
    assert rollout_prompt(record, "prompted", 0) == rollout_prompt(
        record, "prompted", 1
    )
