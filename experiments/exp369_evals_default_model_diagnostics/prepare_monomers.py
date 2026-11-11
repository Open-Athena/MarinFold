"""Freeze context-audited monomer comparisons, including every NMR conformer.

These cases have moderate structural disagreements, not catastrophic fold
failures. Selection precedes MarinFold inference. The full NMR ensemble is
retained so that preference cannot depend on choosing a favorable conformer.
"""

import argparse
import hashlib
import json
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import requests
from tmtools import tm_align

from confirm_training_errors import ca_coordinates, compute_contacts
from screen_monomer_errors import monomer_eligibility

HERE = Path(__file__).resolve().parent
CASES = {
    "P86888": ("Pru p 7", "Solution NMR monomer; moderate disagreement; score all 20 conformers",
               "https://www.rcsb.org/structure/8X67;https://doi.org/10.3390/biom15020232"),
    "Q2FXT9": ("RecJ", "Isolated monomer with Mg2+; moderate disagreement, not a wrong-fold example",
               "https://www.rcsb.org/structure/8Y30;https://doi.org/10.1002/pro.70484"),
    "O34558": ("MrpR / YopR", "Apo monomeric repressor; moderate disagreement, not a wrong-fold example",
               "https://www.rcsb.org/structure/8BJ6;https://doi.org/10.1093/nar/gkad675;https://doi.org/10.1016/j.chom.2023.11.003"),
}


def nmr_ensemble(case: dict, work: Path) -> list[dict]:
    """Recompute canonical contacts and backbone similarity for all 20 models."""
    path = work / "monomers/8x67_ensemble.cif"
    if not path.exists():
        response = requests.get("https://files.rcsb.org/download/8X67.cif", timeout=90)
        response.raise_for_status()
        path.write_bytes(response.content)
    structure = gemmi.read_structure(str(path))
    if len(structure) != 20:
        raise ValueError("Expected the deposited 20-conformer Pru p 7 ensemble")
    sequence = case["sequence"]
    afpath = work / "AF-P86888-F1-model_v4.cif"
    afca = ca_coordinates(afpath, "A", sequence)
    afpairs = {tuple(pair) for pair in case["afdb_pairs"]}
    records = []
    for index, model in enumerate(structure):
        single = structure.clone()
        del single[:]
        single.add_model(model)
        model_path = work / f"monomers/8x67_model{index + 1:02d}.cif"
        single.make_mmcif_document().write_file(str(model_path))
        contacts = compute_contacts(model_path, sequence, stem=f"8x67_model{index + 1}", prefer_chain="A")
        if list(contacts.resolved_positions) != case["resolved"] or contacts.alignment_identity != 1:
            raise ValueError("NMR conformers must share the identical complete scoring universe")
        pairs = {(i, j) for i, j, degree in contacts.contacts if degree >= .001 and j - i >= 6}
        ca = ca_coordinates(model_path, "A", sequence)
        indices = case["resolved"]
        tm = tm_align(np.array([afca[i] for i in indices]), np.array([ca[i] for i in indices]), sequence, sequence)
        records.append({"model": index + 1, "pdb_pairs": sorted(pairs), "afdb_only": sorted(afpairs - pairs),
                        "pdb_only": sorted(pairs - afpairs), "n_pdb_contacts": len(pairs),
                        "contact_f1": 2 * len(afpairs & pairs) / (len(afpairs) + len(pairs)),
                        "tm_score_common_residues": tm.tm_norm_chain1,
                        "ensemble_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if set(records[0]["pdb_pairs"]) != {tuple(pair) for pair in case["pdb_pairs"]}:
        raise ValueError("Full-deposition model 1 disagrees with the frozen ModelServer contacts")
    return records


def main() -> None:
    """Verify context and provenance before writing a sequence-only inference plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("/data/exp369"))
    args = parser.parse_args()
    cases = {row["uniprot"]: row for row in json.loads((args.work / "monomers/confirmed_cases.json").read_text())}
    assemblies = json.loads((args.work / "monomers/entry_assemblies.json").read_text())
    inputs = {(row["PDB"], row["CHAIN"], row["SP_PRIMARY"]): (row, training)
              for row, training in json.loads((args.work / "monomers/screen_inputs.json").read_text())}
    targets, rows, membership = [], [], []
    for accession, (name, interpretation, sources) in CASES.items():
        case = cases[accession]
        metadata = monomer_eligibility(assemblies[case["pdb"]], case["chain"])
        if metadata["monomer_status"] != "passes_assembly_metadata":
            raise ValueError(f"Ineligible monomer assembly: {accession}")
        if case["alignment_identity"] != 1 or case["n_common"] / case["L"] < .90 or not case["training_contacts_reproduced"]:
            raise ValueError(f"Ineligible construct or source provenance: {accession}")
        if pd.Timestamp(case["release_date"]) <= pd.Timestamp(case["afdb_object_created"]):
            raise ValueError("Experimental release does not postdate the AFDB training source")
        source, training = inputs[case["pdb"], case["chain"], accession]
        if hashlib.sha256(training["document"].encode()).hexdigest() != case["training_document_sha256"]:
            raise ValueError("Training membership hash mismatch")
        membership.append({"source_index": source, "training_row": training})
        case = case | {"protein_name": name, "interpretation": interpretation, "context_sources": sources,
                       "assembly_ids": metadata["assembly_ids"], "label_chain": metadata["label_chain"],
                       "monomer_context_audited": True, "catastrophic_fold_error": False}
        rows.append({key: value for key, value in case.items()
                     if key not in ("sequence", "resolved", "afdb_pairs", "pdb_pairs", "afdb_only", "pdb_only")})
        if accession == "P86888":
            case["pdb_ensemble"] = nmr_ensemble(case, args.work)
            pd.DataFrame([{key: value for key, value in item.items() if key not in ("pdb_pairs", "afdb_only", "pdb_only")}
                          for item in case["pdb_ensemble"]]).to_csv(HERE / "data/monomer_nmr_structures.csv", index=False)
        targets.append(case | {"stem": f"{accession}__{case['pdb']}_{case['chain']}", "dataset": "afdb_training_monomer",
                               "truth": case["pdb_pairs"], "orders": {f"{policy}:{repeat}": [] for policy in ("random", "sse_balanced") for repeat in range(2)},
                               "doses_by_repeat": {str(repeat): [{"axis": "count", "dose": 0, "k": 0}] for repeat in range(2)}})
    plan = {"schema": 1, "model": "contacts-v1-exp277-m2-p06-full-epoch-1.5B", "step": 266344,
            "eval_set": "afdb-training-monomers", "n_rollouts": 100, "repeats": 2,
            "temperature": 1., "top_p": .95, "top_k": -1, "targets": targets}
    (HERE / "data/monomer_inference_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (HERE / "data/monomer_training_membership.json").write_text(json.dumps(membership, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(HERE / "data/monomer_selected_cases.csv", index=False)
    print(pd.DataFrame(rows)[["uniprot", "protein_name", "pdb", "chain", "contact_f1", "tm_score_common_residues"]].to_string(index=False))


if __name__ == "__main__":
    main()
