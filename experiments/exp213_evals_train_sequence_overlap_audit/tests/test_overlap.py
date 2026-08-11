# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the exp213 overlap pipeline's pure logic (issue #213).

No network, no mmseqs, no corpus — these pin the three places a silent error
would change the answer: the arm round trip through the FASTA header, the
alignment reduction (which hit wins, and which are ignored), and the stratum
ladder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from overlap_lib import (  # noqa: E402
    ARM_AFDB,
    ARM_ESM,
    HOMOLOGY_EVALUE,
    MIN_QCOV,
    STRATUM_NO_HIT,
    STRATUM_REMOTE,
    arm_of,
    fasta_header,
    identity_stratum,
    is_designed,
)
from search_overlap import read_manifest_meta, reduce_alignments  # noqa: E402
from stratify_and_compare import (  # noqa: E402
    MARINFOLD,
    load_predictors,
    paired_bootstrap,
)


# ---------------------------------------------------------------------------
# FASTA header <-> arm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", [ARM_AFDB, ARM_ESM])
def test_header_round_trip(arm):
    assert arm_of(fasta_header(arm, "00001_7_AF-Q9X0-F1")) == arm


def test_header_rejects_unknown_arm():
    with pytest.raises(ValueError, match="unknown arm"):
        fasta_header("uniref", "x")


@pytest.mark.parametrize("bad", ["has space", "has|pipe"])
def test_header_rejects_ambiguous_local_id(bad):
    # A '|' would make arm_of ambiguous; whitespace would truncate the id mmseqs
    # reports, silently merging distinct training sequences.
    with pytest.raises(ValueError, match="must not contain"):
        fasta_header(ARM_AFDB, bad)


def test_arm_of_rejects_unprefixed_target():
    with pytest.raises(ValueError, match="no recognised arm prefix"):
        arm_of("A0A123")


# ---------------------------------------------------------------------------
# The stratum ladder
# ---------------------------------------------------------------------------


def test_no_significant_hits_is_the_homology_free_stratum():
    assert identity_stratum(0, None) == STRATUM_NO_HIT
    # Even a high identity can't rescue it: with zero significant hits there is
    # nothing to be identical to.
    assert identity_stratum(0, 0.9) == STRATUM_NO_HIT


def test_hits_with_no_covered_alignment_are_remote():
    assert identity_stratum(5, None) == STRATUM_REMOTE
    assert identity_stratum(5, 0.19) == STRATUM_REMOTE


@pytest.mark.parametrize("identity,expected", [
    (0.20, "id_20_30"), (0.299, "id_20_30"),
    (0.30, "id_30_50"), (0.499, "id_30_50"),
    (0.50, "id_50_70"), (0.699, "id_50_70"),
    (0.70, "id_70_100"), (1.0, "id_70_100"),
])
def test_identity_ladder_edges(identity, expected):
    assert identity_stratum(1, identity) == expected


def test_designed_flag():
    assert is_designed("denovo_pdb")
    assert not is_designed("casp_fm")
    assert not is_designed("foldbench100")


# ---------------------------------------------------------------------------
# Alignment reduction
# ---------------------------------------------------------------------------


def _m8(tmp_path: Path, rows: list[tuple]) -> Path:
    """Write an mmseqs convertalis table: query,target,fident,alnlen,qcov,tcov,evalue,bits."""
    path = tmp_path / "aln.m8"
    path.write_text("".join("\t".join(str(v) for v in row) + "\n" for row in rows))
    return path


META = {
    "foldbench100__aaa_A": {"dataset": "foldbench100", "stem": "aaa_A", "query_len": 100,
                            "neff_tier": "", "fold_verdict": "", "seq_leakage": "",
                            "msa_neff": "", "length": ""},
    "denovo_pdb__bbb_A": {"dataset": "denovo_pdb", "stem": "bbb_A", "query_len": 60,
                          "neff_tier": "", "fold_verdict": "", "seq_leakage": "",
                          "msa_neff": "", "length": ""},
}


def test_reduce_picks_best_bitscore_and_gates_identity_on_coverage(tmp_path):
    rows = [
        # A short, near-perfect match: high identity, but covers 10% of the query.
        ("foldbench100__aaa_A", "afdb|s_1_A", 0.95, 10, 0.10, 0.10, 1e-5, 40.0),
        # A long, weaker match: this is the one that should set the identity axis.
        ("foldbench100__aaa_A", "esm_atlas|s_2_H", 0.42, 90, 0.90, 0.88, 1e-30, 200.0),
    ]
    (out,) = [r for r in reduce_alignments(_m8(tmp_path, rows), META, 2000)
              if r["stem"] == "aaa_A"]
    assert out["best_identity_covered"] == pytest.approx(0.42)
    assert out["best_identity_any"] == pytest.approx(0.95)
    assert out["best_bitscore"] == pytest.approx(200.0)
    assert out["best_arm"] == ARM_ESM
    assert out["n_hits_significant"] == 2
    assert out["stratum"] == "id_30_50"


def test_reduce_ignores_hits_above_the_significance_threshold(tmp_path):
    rows = [
        ("foldbench100__aaa_A", "afdb|s_1_A", 0.80, 90, 0.90, 0.90,
         HOMOLOGY_EVALUE * 10, 20.0),
    ]
    (out,) = [r for r in reduce_alignments(_m8(tmp_path, rows), META, 2000)
              if r["stem"] == "aaa_A"]
    assert out["n_hits"] == 1              # reported by mmseqs ...
    assert out["n_hits_significant"] == 0  # ... but not evidence of a relative
    assert out["best_identity_covered"] is None
    assert out["stratum"] == STRATUM_NO_HIT


def test_reduce_attributes_hits_per_arm(tmp_path):
    rows = [
        ("foldbench100__aaa_A", "afdb|s_1_A", 0.35, 90, 0.90, 0.90, 1e-10, 100.0),
        ("foldbench100__aaa_A", "esm_atlas|s_2_H", 0.80, 90, 0.90, 0.90, 1e-40, 300.0),
        ("foldbench100__aaa_A", "esm_atlas|s_3_H", 0.60, 90, 0.90, 0.90, 1e-20, 150.0),
    ]
    (out,) = [r for r in reduce_alignments(_m8(tmp_path, rows), META, 2000)
              if r["stem"] == "aaa_A"]
    assert out["afdb_n_hits_significant"] == 1
    assert out["esm_atlas_n_hits_significant"] == 2
    assert out["afdb_best_identity_covered"] == pytest.approx(0.35)
    assert out["esm_atlas_best_identity_covered"] == pytest.approx(0.80)
    assert out["best_identity_covered"] == pytest.approx(0.80)


def test_reduce_emits_a_row_for_every_query_including_hitless_ones(tmp_path):
    rows = [("foldbench100__aaa_A", "afdb|s_1_A", 0.9, 90, 0.9, 0.9, 1e-30, 300.0)]
    out = reduce_alignments(_m8(tmp_path, rows), META, 2000)
    assert {r["stem"] for r in out} == {"aaa_A", "bbb_A"}
    (hitless,) = [r for r in out if r["stem"] == "bbb_A"]
    assert hitless["n_hits"] == 0
    assert hitless["stratum"] == STRATUM_NO_HIT
    assert hitless["designed"] == 1


def test_reduce_flags_censored_hit_counts(tmp_path):
    rows = [("foldbench100__aaa_A", f"afdb|s_{i}_A", 0.5, 90, 0.9, 0.9, 1e-10, 100.0)
            for i in range(3)]
    (out,) = [r for r in reduce_alignments(_m8(tmp_path, rows), META, 3)
              if r["stem"] == "aaa_A"]
    assert out["hits_censored"] == 1


def test_coverage_gate_is_inclusive_at_the_threshold(tmp_path):
    rows = [("foldbench100__aaa_A", "afdb|s_1_A", 0.55, 50, MIN_QCOV, 0.9, 1e-10, 100.0)]
    (out,) = [r for r in reduce_alignments(_m8(tmp_path, rows), META, 2000)
              if r["stem"] == "aaa_A"]
    assert out["best_identity_covered"] == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# Manifest strata
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict]) -> Path:
    import csv as _csv

    with path.open("w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_exp41_verdicts_fill_foldbench_but_never_overwrite_exp65(tmp_path):
    """FoldBench's manifest has no fold_verdict; exp65's does and wins."""
    foldbench = _write_csv(tmp_path / "fb.csv", [
        {"dataset": "foldbench100", "stem": "aaa_A", "input_seq": "MKV"},
    ])
    exp65 = _write_csv(tmp_path / "e65.csv", [
        {"dataset": "denovo_pdb", "stem": "bbb_A", "input_seq": "MKVA",
         "neff_tier": "orphan", "fold_verdict": "novel_fold", "seq_leakage": "",
         "msa_neff": "1.0", "length": "4"},
    ])
    exp41 = _write_csv(tmp_path / "e41.csv", [
        {"stem": "aaa_A", "verdict": "same_fold"},
        # A stem that only exists in the exp65 half must not be clobbered, and
        # one that exists nowhere must not create a phantom row.
        {"stem": "bbb_A", "verdict": "redundant"},
        {"stem": "zzz_A", "verdict": "redundant"},
    ])
    meta = read_manifest_meta([foldbench, exp65], exp41)
    assert set(meta) == {"foldbench100__aaa_A", "denovo_pdb__bbb_A"}
    assert meta["foldbench100__aaa_A"]["fold_verdict"] == "same_fold"
    assert meta["denovo_pdb__bbb_A"]["fold_verdict"] == "novel_fold"
    assert meta["foldbench100__aaa_A"]["query_len"] == 3


def test_manifest_meta_without_exp41_leaves_verdicts_empty(tmp_path):
    foldbench = _write_csv(tmp_path / "fb.csv", [
        {"dataset": "foldbench100", "stem": "aaa_A", "input_seq": "MKV"},
    ])
    meta = read_manifest_meta([foldbench], None)
    assert meta["foldbench100__aaa_A"]["fold_verdict"] == ""


# ---------------------------------------------------------------------------
# Predictor selection
# ---------------------------------------------------------------------------


def test_distogram_rows_are_not_pooled_into_the_structure_baseline(tmp_path):
    """exp89's table carries Protenix twice; only the `structure` rows count.

    Pooling the two moved Protenix-single-seq's R-precision by 0.22 during
    development, with no error raised — hence this regression test.
    """
    import pandas as pd

    base = pd.DataFrame([
        {"dataset": "foldbench100", "stem": "aaa_A", "model": "protenix-v2",
         "mode": "single_seq", "predictor": "structure", "range": "all",
         "cut": "R", "precision": 0.9},
        {"dataset": "foldbench100", "stem": "aaa_A", "model": "protenix-v2",
         "mode": "single_seq", "predictor": "distogram", "range": "all",
         "cut": "R", "precision": 0.1},
    ])
    baselines_csv = tmp_path / "baselines.csv"
    base.to_csv(baselines_csv, index=False)
    marinfold_csv = tmp_path / "mf.csv"
    pd.DataFrame([{"dataset": "foldbench100", "stem": "aaa_A", "range": "all",
                   "cut": "R", "precision": 0.5}]).to_csv(marinfold_csv, index=False)

    tidy = load_predictors(marinfold_csv, baselines_csv, None)
    protenix = tidy[tidy["predictor"] == "Protenix-v2 single-seq"]
    assert len(protenix) == 1
    assert protenix["precision"].iloc[0] == pytest.approx(0.9)
    assert set(tidy["predictor"]) == {MARINFOLD, "Protenix-v2 single-seq"}


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------


def test_paired_bootstrap_recovers_a_constant_offset():
    a = np.linspace(0.1, 0.9, 200)
    mean, lo, hi = paired_bootstrap(a + 0.05, a)
    assert mean == pytest.approx(0.05)
    # Every pair has the identical difference, so every resample does too.
    assert lo == pytest.approx(0.05) and hi == pytest.approx(0.05)


def test_paired_bootstrap_drops_pairs_with_a_nan_on_either_side():
    a = np.array([0.5, 0.6, np.nan, 0.8])
    b = np.array([0.4, np.nan, 0.2, 0.7])
    mean, lo, hi = paired_bootstrap(a, b)
    assert mean == pytest.approx(0.1)  # only pairs 0 and 3 survive
    assert np.isfinite(lo) and np.isfinite(hi)


def test_paired_bootstrap_with_no_usable_pairs_is_nan():
    mean, lo, hi = paired_bootstrap(np.array([np.nan]), np.array([0.5]))
    assert all(np.isnan(v) for v in (mean, lo, hi))


def test_paired_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.6, 0.2, 300), rng.normal(0.5, 0.2, 300)
    mean, lo, hi = paired_bootstrap(a, b)
    assert lo < mean < hi
