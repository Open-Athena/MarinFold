"""Regression checks against selecting isolated chains from multichain assemblies."""

from screen_monomer_errors import monomer_eligibility


def example_entry(assembly_sizes: list[int]) -> dict:
    """Represent two crystallographic copies of one protein sequence."""
    return {
        "struct": {"title": "Protein crystal structure"},
        "rcsb_accession_info": {"initial_release_date": "2025-01-01"},
        "rcsb_entry_container_identifiers": {"assembly_ids": [str(i + 1) for i in range(len(assembly_sizes))]},
        "assemblies": [{"rcsb_id": f"TEST-{i + 1}",
                        "rcsb_assembly_info": {"polymer_entity_instance_count": size, "polymer_entity_instance_count_protein": size},
                        "pdbx_struct_assembly_gen": [{"asym_id_list": ["A", "B"] if size > 1 else ["AB"[i % 2]]}]}
                       for i, size in enumerate(assembly_sizes)],
        "polymer_entities": [{"entity_poly": {"type": "polypeptide(L)"},
                              "rcsb_polymer_entity": {"pdbx_description": "Protein"},
                              "polymer_entity_instances": [
                                  {"rcsb_polymer_entity_instance_container_identifiers": {"asym_id": label, "auth_asym_id": author}}
                                  for label, author in (("A", "1h"), ("B", "9A"))]}],
    }


def test_single_extracted_chain_does_not_make_dimer_a_monomer() -> None:
    assert monomer_eligibility(example_entry([2]), "1h")["monomer_status"] == "nonmonomer_or_ambiguous_assemblies"


def test_single_chain_assembly_does_not_override_conflicting_dimer_annotation() -> None:
    assert monomer_eligibility(example_entry([1, 2]), "1h")["monomer_status"] == "nonmonomer_or_ambiguous_assemblies"


def test_multiple_crystal_copies_can_have_separate_monomer_assemblies() -> None:
    result = monomer_eligibility(example_entry([1, 1]), "9A")
    assert result["monomer_status"] == "passes_assembly_metadata"
    assert result["label_chain"] == "B"
    assert result["assembly_ids"] == '["TEST-2"]'


def test_protein_nucleic_acid_assembly_is_excluded() -> None:
    entry = example_entry([2])
    entry["assemblies"][0]["rcsb_assembly_info"]["polymer_entity_instance_count_protein"] = 1
    assert monomer_eligibility(entry, "1h")["monomer_status"] == "nonmonomer_or_ambiguous_assemblies"
