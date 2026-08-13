# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure unit tests for exp226's logic — no network, no mmseqs, no data files.

Three things here can be wrong in a way the pipeline would not notice:

* **Chain resolution.** FoldBench's ``chain_id`` is sometimes a label asym id,
  so :func:`select_entity` tries auth first and label second. Silently picking
  the wrong entity would put a *different protein's* sequence into the query
  set, and every identity number downstream would be about that protein.
* **The survival predicate.** A protein with no covered hit has an empty
  identity cell and must survive every filter; getting that backwards would
  quietly delete the most novel proteins from the answer.
* **Fisher's exact test**, checked against textbook tables, because the
  newer-vs-older verdict rests on it and the counts are small.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyze_survival import fisher_exact_two_sided, survives  # noqa: E402
from build_query_set import (  # noqa: E402
    MonomerTarget,
    N_FOLDBENCH_MONOMERS,
    ResolvedTarget,
    parse_targets,
    select_entity,
)


# --- helpers ----------------------------------------------------------------


def entity(rcsb_id: str, seq: str, auth: list[str], label: list[str],
           kind: str = "polypeptide(L)", taxids: list[int] | None = None) -> dict:
    return {
        "rcsb_id": rcsb_id,
        "entity_poly": {"type": kind, "pdbx_seq_one_letter_code_can": seq},
        "rcsb_polymer_entity_container_identifiers": {
            "auth_asym_ids": auth, "asym_ids": label,
        },
        "rcsb_entity_source_organism": [
            {"ncbi_taxonomy_id": t, "ncbi_scientific_name": f"tax{t}"}
            for t in (taxids or [9606])
        ],
        "rcsb_polymer_entity": {"pdbx_description": "a protein"},
    }


def entry(rcsb_id: str, entities: list[dict]) -> dict:
    return {"rcsb_id": rcsb_id, "struct": {"title": "t"}, "polymer_entities": entities}


def resolved(taxids: tuple[str, ...]) -> ResolvedTarget:
    return ResolvedTarget(
        target=MonomerTarget("1abc", "A"), entity_id="1ABC_1", sequence="AAA",
        auth_asym_ids=("A",), asym_ids=("A",), chain_match="auth",
        source_taxids=taxids, source_names=(), title="", description="",
    )


# --- chain resolution -------------------------------------------------------


def test_auth_chain_is_preferred_over_label():
    """The common case: FoldBench's chain is the auth id."""
    e = entry("1ABC", [entity("1ABC_1", "AAA", ["A"], ["X"]),
                       entity("1ABC_2", "CCC", ["B"], ["A"])])
    got = select_entity(e, MonomerTarget("1abc", "A"))
    assert (got.entity_id, got.sequence, got.chain_match) == ("1ABC_1", "AAA", "auth")


def test_label_chain_is_the_fallback():
    """8ork_A / 5sbj_A: FoldBench stored the label asym id, auth is AAA / C."""
    e = entry("8ORK", [entity("8ORK_1", "MGET", ["AAA"], ["A"])])
    got = select_entity(e, MonomerTarget("8ork", "A"))
    assert (got.entity_id, got.sequence, got.chain_match) == ("8ORK_1", "MGET", "label")


def test_ambiguous_chain_raises_rather_than_guessing():
    e = entry("1ABC", [entity("1ABC_1", "AAA", ["A"], ["A"]),
                       entity("1ABC_2", "CCC", ["A"], ["B"])])
    with pytest.raises(ValueError, match="matches 2 protein entities"):
        select_entity(e, MonomerTarget("1abc", "A"))


def test_unmatched_chain_raises():
    e = entry("1ABC", [entity("1ABC_1", "AAA", ["B"], ["B"])])
    with pytest.raises(ValueError, match="matches no protein entity"):
        select_entity(e, MonomerTarget("1abc", "A"))


def test_non_protein_entities_are_ignored():
    """A DNA/RNA entity sharing the chain letter must not be selected."""
    e = entry("1ABC", [entity("1ABC_1", "AUGC", ["A"], ["A"], kind="polyribonucleotide"),
                       entity("1ABC_2", "MKV", ["A"], ["B"])])
    assert select_entity(e, MonomerTarget("1abc", "A")).sequence == "MKV"


def test_entry_with_no_protein_entity_raises():
    e = entry("1ABC", [entity("1ABC_1", "AUGC", ["A"], ["A"], kind="polyribonucleotide")])
    with pytest.raises(ValueError, match="no protein entity"):
        select_entity(e, MonomerTarget("1abc", "A"))


def test_sequence_is_uppercased_and_stripped():
    e = entry("1ABC", [entity("1ABC_1", " mkv \n", ["A"], ["A"])])
    assert select_entity(e, MonomerTarget("1abc", "A")).sequence == "MKV"


# --- the FoldBench target list ----------------------------------------------


def test_parse_targets_strips_the_assembly_suffix():
    text = "pdb_id,chain_id\n" + "".join(
        f"{i:04x}-assembly1,A\n" for i in range(N_FOLDBENCH_MONOMERS)
    )
    targets = parse_targets(text)
    assert len(targets) == N_FOLDBENCH_MONOMERS
    assert targets[0].pdb_id == "0000" and targets[0].stem == "0000_A"


def test_parse_targets_rejects_a_wrong_row_count():
    with pytest.raises(SystemExit, match="expected 334"):
        parse_targets("pdb_id,chain_id\n5sbj-assembly1,A\n")


def test_parse_targets_rejects_duplicate_rows():
    rows = [f"{i:04x}-assembly1,A" for i in range(N_FOLDBENCH_MONOMERS - 1)]
    text = "pdb_id,chain_id\n" + "\n".join(rows + [rows[0]]) + "\n"
    with pytest.raises(SystemExit, match="duplicate"):
        parse_targets(text)


# --- the designed-protein proxy ---------------------------------------------


def test_synthetic_construct_taxon_is_designed():
    assert resolved(("32630",)).synthetic


def test_missing_source_organism_counts_as_designed():
    assert resolved(()).synthetic


def test_a_natural_source_is_not_designed():
    assert not resolved(("9606",)).synthetic


def test_a_chimera_with_any_natural_source_is_not_designed():
    assert not resolved(("32630", "9606")).synthetic


# --- the survival predicate -------------------------------------------------


def test_no_covered_hit_survives_every_filter():
    assert survives({"best_identity_covered": ""}, 0.40)
    assert survives({"best_identity_covered": ""}, 0.30)


def test_the_threshold_is_strict():
    """A protein at exactly 40 % identity is filtered out by a <40 % filter."""
    assert not survives({"best_identity_covered": "0.40"}, 0.40)
    assert survives({"best_identity_covered": "0.399"}, 0.40)


def test_survives_reads_the_requested_column():
    row = {"best_identity_covered": "0.9", "best_identity_any": "0.1"}
    assert not survives(row, 0.40)
    assert survives(row, 0.40, "best_identity_any")


# --- Fisher's exact test ----------------------------------------------------


def test_fisher_matches_the_tea_tasting_table():
    assert fisher_exact_two_sided(3, 1, 1, 3) == pytest.approx(0.4857142857)


def test_fisher_matches_a_significant_table():
    assert fisher_exact_two_sided(1, 9, 11, 3) == pytest.approx(0.0027594, abs=1e-7)


def test_fisher_is_symmetric_in_its_rows():
    assert fisher_exact_two_sided(15, 85, 23, 199) == pytest.approx(
        fisher_exact_two_sided(23, 199, 15, 85))


def test_fisher_is_one_for_identical_proportions():
    assert fisher_exact_two_sided(5, 5, 5, 5) == pytest.approx(1.0)


def test_fisher_never_exceeds_one():
    for table in [(1, 1, 1, 1), (0, 10, 0, 10), (2, 3, 4, 5), (7, 0, 0, 7)]:
        assert 0.0 <= fisher_exact_two_sided(*table) <= 1.0
