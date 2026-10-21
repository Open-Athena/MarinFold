"""Freeze full-complex contact truth and inputs for the four-model comparison."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from freeze_foldbench_eval import FOLDBENCH_GT, atom_site_maps, selected_structure
from marinfold.document_structures.contacts_v1 import analyze_structure

HERE = Path(__file__).resolve().parent


def full_truth(target: dict, directory: Path) -> dict:
    """Recompute both contact classes and verify the frozen inter-chain truth."""
    path = directory / f"{target['stem']}.cif"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == target["ground_truth_cif_sha256"]
    )
    auth, residue_map = atom_site_maps(path, target["chain_ids"])
    analysis = analyze_structure(
        selected_structure(path, auth), entry_id=target["stem"], max_chains=2
    )
    offsets = dict(zip(auth, target["chain_offsets"], strict=True))
    mapping = {
        r.seq_index: offsets[r.chain] + residue_map[(r.chain, r.resnum)]
        for r in analysis.residues
    }
    assert sorted(mapping.values()) == target["resolved_positions"]
    boundary = target["chain_lengths"][0]
    contacts = []
    for contact in analysis.contacts:
        i, j = sorted((mapping[contact.seq_i], mapping[contact.seq_j]))
        inter = i < boundary <= j
        if contact.degree >= 0.001 and (inter or j - i >= 6):
            contacts.append([i, j, contact.degree])
    contacts.sort()
    assert {(i, j) for i, j, _ in contacts if i < boundary <= j} == {
        tuple(p) for p in target["gt_contacts"]
    }
    return {**target, "all_contacts": contacts}


def main() -> None:
    """Save immutable truth and native two-chain predictor inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=FOLDBENCH_GT)
    parser.add_argument("--output", type=Path, default=HERE / "data/four_model_v1")
    args = parser.parse_args()
    targets = pq.read_table(
        HERE / "data/foldbench_complex_contact_eval_targets.parquet"
    ).to_pylist()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for target in targets:
        rows.append(full_truth(target, args.ground_truth))
        print(target["stem"], len(rows[-1]["all_contacts"]), flush=True)
    (args.output / "targets.json").write_text(json.dumps(rows, separators=(",", ":")) + "\n")
    inputs = [
        {
            "stem": t["stem"],
            "chain_sequences": t["chain_sequences"],
            "L": t["L"],
            "split": t["split"],
        }
        for t in targets
    ]
    (args.output / "predictor_inputs.json").write_text(
        json.dumps(inputs, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
