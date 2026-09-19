"""Guard the whole-corpus selection rules against silent drift from the audit."""

import csv
from pathlib import Path

from select_corpus import (
    CLUSTER_CAP,
    apply_cap,
    sequence_excluded,
    structure_excluded,
)

HERE = Path(__file__).resolve().parent
SEQUENCE_FIELDS = "query,target,fident,qcov,tcov,qlen,tlen,evalue".split(",")


def hit(**overrides) -> dict:
    row = {
        "query": "q",
        "target": "t",
        "fident": "0.10",
        "qcov": "0.10",
        "tcov": "0.10",
        "qlen": "100",
        "tlen": "100",
        "evalue": "1.0",
    }
    return {**row, **overrides}


def test_identity_rule_needs_both_identity_and_shorter_coverage() -> None:
    assert sequence_excluded(hit(fident="0.35", qcov="0.60"))
    assert not sequence_excluded(hit(fident="0.35", qcov="0.40"))
    assert not sequence_excluded(hit(fident="0.25", qcov="0.90"))


def test_coverage_is_measured_on_the_shorter_sequence() -> None:
    # query is longer, so the target's coverage is the one that counts
    assert sequence_excluded(
        hit(fident="0.9", qlen="400", tlen="100", qcov="0.10", tcov="0.80")
    )
    assert not sequence_excluded(
        hit(fident="0.9", qlen="400", tlen="100", qcov="0.80", tcov="0.10")
    )


def test_strong_homology_excludes_without_any_identity_threshold() -> None:
    assert sequence_excluded(hit(fident="0.05", qcov="0.01", evalue="1e-9"))


def test_structure_rule_requires_both_normalizations_and_both_coverages() -> None:
    assert structure_excluded(
        {
            "qtmscore": "0.9",
            "ttmscore": "0.85",
            "qcov": "0.9",
            "tcov": "0.85",
        }
    )
    assert not structure_excluded(
        {"qtmscore": "0.9", "ttmscore": "0.7", "qcov": "0.9", "tcov": "0.9"}
    )
    assert not structure_excluded(
        {"qtmscore": "0.9", "ttmscore": "0.9", "qcov": "0.9", "tcov": "0.5"}
    )


def test_cap_keeps_the_most_confident_members_and_marks_the_rest() -> None:
    index = {f"s{i}": {"plddt": 70 + i, "scrmsd": 1.0} for i in range(8)}
    selected, rows = apply_cap({"rep": list(index)}, index)
    assert len(selected) == CLUSTER_CAP
    assert selected == {"s7", "s6", "s5", "s4", "s3"}
    assert len(rows) == 8
    assert sum(row["selected"] for row in rows) == CLUSTER_CAP


def test_cap_breaks_confidence_ties_by_self_consistency_then_stem() -> None:
    index = {
        "b": {"plddt": 90, "scrmsd": 0.5},
        "a": {"plddt": 90, "scrmsd": 0.5},
        "c": {"plddt": 90, "scrmsd": 0.1},
    }
    _, rows = apply_cap({"rep": ["b", "a", "c"]}, index, cap=1)
    order = [row["stem"] for row in sorted(rows, key=lambda r: r["rank"])]
    assert order == ["c", "a", "b"]


def test_rule_reproduces_the_audit_exclusion_set_exactly() -> None:
    """The 18-hour audit's own hits must yield the stems it recorded as excluded."""
    hits = Path("/data/exp278/scale-review-18h/sequence-hits.tsv")
    retention = HERE / "data/scale-20260909/review-18h/retention.csv"
    if not hits.exists() or not retention.exists():
        return
    csv.field_size_limit(10**9)
    rows = list(csv.DictReader(retention.open()))
    quality = {r["stem"] for r in rows if r["quality_pass"] == "True"}
    audit = {
        r["stem"]
        for r in rows
        if r["quality_pass"] == "True" and r["sequence_excluded"] == "True"
    }
    mine = set()
    with hits.open() as handle:
        for row in csv.DictReader(handle, fieldnames=SEQUENCE_FIELDS, delimiter="\t"):
            if sequence_excluded(row):
                mine.add(row["query"])
    assert mine & quality == audit
