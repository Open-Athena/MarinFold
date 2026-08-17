# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the exp241 audit. No network, no /data, no mmseqs."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze  # noqa: E402
import annotate_rcsb as A  # noqa: E402
import upstream as U  # noqa: E402


# --- the containment test the chain-resolution control turns on -------------

def test_is_subsequence_exact_and_gapped():
    assert A.is_subsequence("ACDEF", "ACDEF")
    # unmodelled loop in the middle: the query is the observed residues
    assert A.is_subsequence("ACDEF", "ACXXDEXF".replace("X", "G"))
    assert A.is_subsequence("", "ANYTHING")


def test_is_subsequence_rejects_reordering():
    """Order matters — a permuted chain is not the same chain."""
    assert not A.is_subsequence("FEDCA", "ACDEF")
    assert not A.is_subsequence("ACDEFG", "ACDEF")


# --- kingdom assignment ------------------------------------------------------

def test_kingdom_virus_wins_over_cellular_clades():
    """A virus lineage can carry a host clade; the virus test must come first."""
    assert A.kingdom_of({"Viruses", "Eukaryota"}, "Vaccinia virus") == "virus"


def test_kingdom_synthetic_and_fallbacks():
    assert A.kingdom_of({"artificial sequences"}, "synthetic construct") == "synthetic"
    assert A.kingdom_of({"Bacteria"}, "Escherichia coli") == "bacteria"
    assert A.kingdom_of(set(), "metagenome") == "unclassified"
    assert A.kingdom_of(set(), "") == "unknown"


# --- the mechanism ladder ----------------------------------------------------

def _reach(**overrides):
    row = {"designed_signal": "0", "in_afdb_arm": "0", "in_afdb_full": "0",
           "uniprot_accessions": ""}
    row.update({k: str(v) for k, v in overrides.items()})
    return row


def test_ladder_charges_designed_first():
    """A designed protein is charged as designed even with every other flag set."""
    assert analyze.classify(_reach(
        designed_signal=1, uniprot_accessions="P12345", in_afdb_full=1,
    )) == "designed_not_natural"


def test_ladder_orders_the_natural_rungs():
    assert analyze.classify(_reach()) == "not_in_uniprot"
    assert analyze.classify(
        _reach(uniprot_accessions="P12345")) == "afdb_absent"
    assert analyze.classify(
        _reach(uniprot_accessions="P12345", in_afdb_full=1)) == "unsampled_corpus"


def test_ladder_flags_a_search_miss_above_everything_natural():
    """In the arm but reported below 40 % identity is a defect, not a mechanism."""
    assert analyze.classify(_reach(
        uniprot_accessions="P12345", in_afdb_full=1, in_afdb_arm=1,
    )) == "search_miss"


def test_every_mechanism_is_named():
    for row in analyze.read_csv(analyze.DATA / "mechanism_counts.csv"):
        assert row["escape_mechanism"] in analyze.MECHANISMS


# --- entity selection --------------------------------------------------------

def _entity(rcsb_id, chains, seq):
    return {"rcsb_id": rcsb_id,
            "entity_poly": {"pdbx_seq_one_letter_code_can": seq},
            "rcsb_polymer_entity_container_identifiers": {
                "auth_asym_ids": chains, "asym_ids": chains}}


def test_select_entity_prefers_the_named_chain():
    entry = {"polymer_entities": [_entity("X_1", ["A"], "AAAAAAAA"),
                                  _entity("X_2", ["B"], "CDEFGHIK")]}
    assert A.select_entity(entry, "B", "CDEFGHIK")["rcsb_id"] == "X_2"


def test_select_entity_falls_back_to_the_sequence_when_no_chain():
    """A CASP stem names no chain; the domain must still find its own entity."""
    entry = {"polymer_entities": [_entity("X_1", ["A"], "M" * 400),
                                  _entity("X_2", ["B"], "PPCDEFGHIKPP")]}
    assert A.select_entity(entry, None, "CDEFGHIK")["rcsb_id"] == "X_2"


def test_select_entity_matches_a_gapped_query():
    entry = {"polymer_entities": [_entity("X_1", ["A"], "M" * 40),
                                  _entity("X_2", ["B"], "CDXXEFXXGHIK")]}
    assert A.select_entity(entry, None, "CDEFGHIK")["rcsb_id"] == "X_2"


# --- the upstream seam -------------------------------------------------------

def test_afdb_header_regex_matches_the_arm_grammar():
    m = U.AFDB_HEADER_RE.match(">afdb|01351_1105_AF-A0A1S3FT64-F1\n")
    assert m and m.group(1) == "A0A1S3FT64"
    assert U.AFDB_HEADER_RE.match(">esm_atlas|02452_8247_bc164314b66c") is None


def test_eval2_threshold_is_forty_percent_not_thirty():
    """The constant this experiment reports is checked against eval2 itself.

    ``read_proteins`` re-derives membership from ``best_identity_covered <
    EVAL2_THRESHOLD`` and raises if it disagrees with the published manifest, so
    this call passing *is* the assertion that eval2 cuts at 40 %.
    """
    assert U.EVAL2_THRESHOLD == 0.40
    proteins = U.read_proteins()
    assert len(proteins) == U.EXPECTED_IDENTITY_TABLE_N
    assert len(U.eval2_natural(proteins)) == U.EXPECTED_EVAL2_NATURAL_N


def test_query_fasta_agrees_with_the_identity_table():
    sequences = U.read_query_sequences()
    identity = U.read_identity_table()
    assert set(sequences) == set(identity)
    for key, row in identity.items():
        assert len(sequences[key]) == int(row["length"] or row["query_len"])


def test_casp_map_skips_unavailable_rows():
    mapping = U.read_casp_pdb_map()
    assert mapping, "exp65's CASP fallback map should not be empty"
    assert all(pdb and pdb != "unavailable" for pdb, _ in mapping.values())


@pytest.mark.parametrize("name", [
    "provenance_of_the_78.csv", "mechanism_table.csv", "mechanism_counts.csv",
    "label_audit.csv", "kingdom_by_arm.csv", "arm_identity_histogram.csv",
])
def test_result_tables_exist_and_are_non_empty(name):
    rows = analyze.read_csv(analyze.DATA / name)
    assert rows, name


def test_mechanism_table_covers_every_one_of_the_78():
    rows = analyze.read_csv(analyze.DATA / "mechanism_table.csv")
    assert len(rows) == U.EXPECTED_EVAL2_NATURAL_N
    counts = analyze.read_csv(analyze.DATA / "mechanism_counts.csv")
    assert sum(int(r["n"]) for r in counts) == len(rows)


# --- the applied correction --------------------------------------------------

def test_manifest_v2_is_a_drop_in_replacement():
    """Every exp226 column survives, so downstream joins keep working."""
    v2 = {r["dataset"] + "/" + r["stem"]: r
          for r in analyze.read_csv(analyze.DATA / "eval2_manifest_v2.csv")}
    original = {r["dataset"] + "/" + r["stem"]: r for r in U.read_eval2()}
    assert set(v2) == set(original)
    for key, row in original.items():
        for column, value in row.items():
            if column == "designed_any":   # the one column this corrects
                continue
            assert v2[key][column] == value, f"{key}.{column} changed"


def test_manifest_v2_corrects_exactly_the_audited_designs():
    v2 = analyze.read_csv(analyze.DATA / "eval2_manifest_v2.csv")
    flipped = [r for r in v2
               if r["designed_any"] == "1" and r["designed_any_exp226"] == "0"]
    assert len(flipped) == 15
    # The flag only ever moves natural -> designed; nothing is un-designed.
    assert not [r for r in v2
                if r["designed_any"] == "0" and r["designed_any_exp226"] == "1"]
    assert all(r["designed_source"].startswith("exp241_") for r in flipped)
    assert sum(1 for r in v2 if r["designed_any"] == "0") == 63


def test_manifest_v2_viral_flag_matches_the_kingdom_column():
    v2 = analyze.read_csv(analyze.DATA / "eval2_manifest_v2.csv")
    for row in v2:
        assert row["is_viral"] == ("1" if row["kingdom"] == "virus" else "0")
    natural_viral = sum(1 for r in v2
                        if r["designed_any"] == "0" and r["is_viral"] == "1")
    assert natural_viral == 27


def test_rescored_headline_uses_the_audited_n():
    rows = analyze.read_csv(analyze.DATA / "eval2_headline_v2.csv")
    natural = [r for r in rows if r["subset"] == "eval2 natural (audited)"]
    assert natural and all(int(r["n"]) == 63 for r in natural)
    halves = {r["subset"]: int(r["n"]) for r in rows
              if r["subset"].startswith("eval2 natural, ")}
    assert halves["eval2 natural, viral"] + halves["eval2 natural, non-viral"] == 63
