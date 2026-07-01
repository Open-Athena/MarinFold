# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the marinfold-free rollout parsing/scoring (no model needed)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rollout_metrics import gt_by_band, parse_pred, score_rollout  # noqa: E402


# A realization where sequence index i sits at position 100+i (L=12).
POS_TO_SEQ = {100 + i: i for i in range(12)}


def test_parse_dedup_and_minsep():
    text = ("<contact> <p100> <p106> "      # (0,6)  sep 6  ok
            "<contact> <p106> <p100> "      # (0,6) again, reversed -> deduped
            "<contact> <p101> <p109> "      # (1,9)  sep 8  ok
            "<contact> <p100> <p102> "      # (0,2)  sep 2  -> dropped (< MIN_SEP)
            "<contact> <p200> <p100> "      # unknown position -> dropped
            "<end>")
    pred = parse_pred(text, POS_TO_SEQ)
    assert pred == {(0, 6), (1, 9)}


def test_score_precision_recall_f1():
    gt = {(0, 6), (2, 8)}                    # both sep >= 6
    gtb = gt_by_band(gt)
    pred = {(0, 6), (1, 9)}                  # 1 correct of 2 predicted
    sc = score_rollout(pred, gtb)
    assert sc["all_npred"] == 2
    assert sc["all_tp"] == 1
    assert sc["all_prec"] == 0.5
    assert sc["all_rec"] == 0.5
    assert abs(sc["all_f1"] - 0.5) < 1e-9


def test_empty_prediction():
    gt = {(0, 6), (2, 8)}
    sc = score_rollout(set(), gt_by_band(gt))
    assert sc["all_npred"] == 0
    assert sc["all_prec"] == 0.0
    assert sc["all_rec"] == 0.0
    assert sc["all_f1"] == 0.0


def test_band_split():
    # (0,8) short/med? sep 8 -> short[6,11]; (0,15) -> med[12,23]; (0,40) -> long[>=24]
    gt = {(0, 8), (0, 15), (0, 40)}
    gtb = gt_by_band(gt)
    assert len(gtb["short"]) == 1 and len(gtb["med"]) == 1 and len(gtb["long"]) == 1
    pred = {(0, 8), (0, 40)}                 # right short + long, miss med
    sc = score_rollout(pred, gtb)
    assert sc["short_rec"] == 1.0
    assert sc["long_rec"] == 1.0
    assert sc["med_rec"] == 0.0
    assert sc["all_tp"] == 2 and sc["all_npred"] == 2


def test_recall_nan_when_no_gt_in_band():
    gt = {(0, 8)}                            # only a short contact
    sc = score_rollout({(0, 8)}, gt_by_band(gt))
    assert math.isnan(sc["long_rec"])
    assert math.isnan(sc["long_f1"])
    assert sc["all_rec"] == 1.0
