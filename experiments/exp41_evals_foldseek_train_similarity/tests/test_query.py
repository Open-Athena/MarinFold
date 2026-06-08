# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the foldseek train-similarity query tool.

Unit tests (no network, no foldseek) drive the parse -> join -> verdict
path off an inline ``.m8`` fixture. The real foldseek path is exercised by
the actual FoldBench-100 run (see ``data/foldbench_vs_full_reps_similarity.csv``),
so it is not re-run here.

Run via ``uv run --extra test pytest tests/``.
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

import query_similarity as qs  # noqa: E402
from query_similarity import (  # noqa: E402
    DEFAULT_MAX_SEQS,
    FORMAT_FIELDS,
    easy_search,
    normalize_name,
    parse_m8,
    summarize,
    verdict_for,
)

# Fixture: one .m8 row per (query, target) hit. Columns match FORMAT_FIELDS:
# query target alntmscore qtmscore ttmscore lddt fident alnlen evalue
_FIXTURE_M8 = "\n".join(
    [
        # cand_red: best train hit qtm 0.97 -> redundant
        "cand_red\tr_train_a\t0.98\t0.97\t0.95\t0.9\t0.80\t120\t1e-20",
        "cand_red\tr_train_b\t0.40\t0.38\t0.40\t0.5\t0.20\t60\t1e-3",
        # cand_fold: best train hit qtm 0.60 -> same_fold
        "cand_fold\tr_train_b\t0.62\t0.60\t0.58\t0.6\t0.25\t90\t1e-8",
        # cand_novel: best train hit qtm 0.20 -> novel_fold
        "cand_novel\tr_train_a\t0.22\t0.20\t0.19\t0.3\t0.10\t40\t1e-1",
        # cand_heldout: only hits a val rep (qtm 0.80) -> split=val, no train hit
        "cand_heldout\tr_val_a\t0.82\t0.80\t0.79\t0.8\t0.55\t100\t1e-12",
    ]
)

# representative_id -> split. cand_nohit has no rows at all.
_FIXTURE_MANIFEST = pd.DataFrame(
    {
        "representative_id": ["r_train_a", "r_train_b", "r_val_a", "r_test_a"],
        "split": ["train", "train", "val", "test"],
    }
)

_STEMS = {"cand_red", "cand_fold", "cand_novel", "cand_heldout", "cand_nohit"}
_N_RES = {s: 100 for s in _STEMS}


def _summary() -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as td:
        m8 = Path(td) / "aln.m8"
        m8.write_text(_FIXTURE_M8 + "\n")
        hits = parse_m8(m8)
    assert list(hits.columns) == list(FORMAT_FIELDS)
    return summarize(
        hits, _FIXTURE_MANIFEST, _N_RES, _STEMS,
        foldseek_version="testver", db_snapshot_tag="testtag",
    )


def test_output_schema():
    df = _summary()
    expected = {
        "stem", "n_residues", "best_target_rep", "best_target_split",
        "best_alntmscore", "best_qtmscore", "best_train_target_rep",
        "best_train_alntmscore", "best_train_qtmscore", "best_train_fident",
        "tm_field", "n_hits_tm_ge_fold", "n_train_hits_tm_ge_fold",
        "verdict", "fold_tm", "redundant_tm", "foldseek_version", "db_snapshot_tag",
    }
    assert set(df.columns) == expected
    assert len(df) == len(_STEMS)  # one row per candidate, including the no-hit one


def test_verdict_bins():
    df = _summary().set_index("stem")
    assert df.loc["cand_red", "verdict"] == "redundant"
    assert df.loc["cand_fold", "verdict"] == "same_fold"
    assert df.loc["cand_novel", "verdict"] == "novel_fold"
    # Numbers carried through correctly.
    assert df.loc["cand_red", "best_train_qtmscore"] == pytest.approx(0.97)
    assert df.loc["cand_red", "best_train_fident"] == pytest.approx(0.80)


def test_heldout_candidate_has_split_but_no_train_match():
    """A candidate whose only hit is a val/test rep: non-null split, empty train, novel."""
    row = _summary().set_index("stem").loc["cand_heldout"]
    assert row["best_target_split"] == "val"
    assert pd.isna(row["best_train_target_rep"])
    assert pd.isna(row["best_train_qtmscore"])
    assert row["n_train_hits_tm_ge_fold"] == 0
    assert row["verdict"] == "novel_fold"


def test_no_hit_candidate_is_novel():
    row = _summary().set_index("stem").loc["cand_nohit"]
    assert pd.isna(row["best_target_rep"])
    assert row["n_hits_tm_ge_fold"] == 0
    assert row["verdict"] == "novel_fold"


def test_hit_counts():
    """n_hits counts targets at/above the fold boundary; train subset too."""
    df = _summary().set_index("stem")
    # cand_red: qtm 0.97 (>=0.5) and 0.38 (<0.5) -> 1 hit at fold level, 1 train hit
    assert df.loc["cand_red", "n_hits_tm_ge_fold"] == 1
    assert df.loc["cand_red", "n_train_hits_tm_ge_fold"] == 1


def test_verdict_for_boundaries():
    assert verdict_for(None, 0.5, 0.9) == "novel_fold"
    assert verdict_for(0.49, 0.5, 0.9) == "novel_fold"
    assert verdict_for(0.5, 0.5, 0.9) == "same_fold"
    assert verdict_for(0.89, 0.5, 0.9) == "same_fold"
    assert verdict_for(0.9, 0.5, 0.9) == "redundant"


def test_normalize_name():
    known = {"5sbj_A", "K7TTU0"}
    # Exact filename-stem match (the case foldseek actually emits here).
    assert normalize_name("5sbj_A", known) == "5sbj_A"
    assert normalize_name("5sbj_A.cif", known) == "5sbj_A"
    assert normalize_name("K7TTU0", known) == "K7TTU0"
    # A trailing chain suffix is stripped to reach a known id.
    assert normalize_name("K7TTU0_A", known) == "K7TTU0"
    # Unknown name falls through to the stripped base (won't join).
    assert normalize_name("unknown_thing.cif", known) == "unknown_thing"


def test_easy_search_forwards_max_seqs(monkeypatch, tmp_path: Path):
    captured: dict[str, list[str]] = {}

    def _fake_run_foldseek(args: list[str]):
        captured["args"] = args
        return None

    monkeypatch.setattr(qs, "run_foldseek", _fake_run_foldseek)

    easy_search(
        tmp_path / "candidates",
        tmp_path / "db" / "targetDB",
        tmp_path / "out" / "aln.m8",
        tmp_path / "tmp",
        max_seqs=4096,
    )
    args = captured["args"]
    assert args[0] == "easy-search"
    assert args[args.index("--max-seqs") + 1] == "4096"


def test_easy_search_default_max_seqs_matches_foldseek_default(monkeypatch, tmp_path: Path):
    captured: dict[str, list[str]] = {}

    def _fake_run_foldseek(args: list[str]):
        captured["args"] = args
        return None

    monkeypatch.setattr(qs, "run_foldseek", _fake_run_foldseek)

    easy_search(
        tmp_path / "candidates",
        tmp_path / "db" / "targetDB",
        tmp_path / "out" / "aln.m8",
        tmp_path / "tmp",
    )
    args = captured["args"]
    assert args[args.index("--max-seqs") + 1] == str(DEFAULT_MAX_SEQS)
