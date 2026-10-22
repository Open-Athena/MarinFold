"""Verify shortlisted training/PDB disagreements against the original AFDB v4."""

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from tmtools import tm_align

from marinfold.document_structures.contacts_v1.read import live_contacts, sequence_from_document

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp89_evals_contacts_v1_model_on_eval_set"))
from pyconfind_contacts import align_obs_to_ref, compute_contacts


def ca_coordinates(path: Path, chain: str, sequence: str) -> dict[int, np.ndarray]:
    """Map observed CA coordinates into the exact training sequence."""
    structure = gemmi.read_structure(str(path))
    residues = [r for r in structure[0][chain] if r.het_flag == "A" and r.find_atom("CA", "*")]
    observed = "".join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper() for r in residues)
    mapping = align_obs_to_ref(observed, sequence)
    return {index: np.array(list(residues[obs].find_atom("CA", "*").pos))
            for obs, index in enumerate(mapping) if index is not None and observed[obs] == sequence[index]}


def main() -> None:
    """Recompute experimental and original training-structure contacts exactly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("/data/exp369"))
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--screen", type=Path, default=HERE / "data/afdb_candidate_screen.csv")
    parser.add_argument("--inputs", type=Path, default=Path("/data/exp369/screen_inputs.json"))
    parser.add_argument("--output-csv", type=Path, default=HERE / "data/afdb_confirmed_disagreements.csv")
    parser.add_argument("--output-cases", type=Path, default=Path("/data/exp369/afdb_cases.json"))
    parser.add_argument("--unique-proteins", action="store_true")
    parser.add_argument("--exact-identity", action="store_true")
    args = parser.parse_args()
    candidates = pd.read_csv(args.screen).query("status == 'screened'").sort_values(
        ["fraction_over_12A", "PDB", "CHAIN"], ascending=[False, True, True])
    if args.exact_identity:
        candidates = candidates.loc[candidates.identity == 1]
    if args.unique_proteins:
        candidates = candidates.drop_duplicates("SP_PRIMARY")
    candidates = candidates.head(args.top)
    inputs = {(row["PDB"], row["CHAIN"], row["SP_PRIMARY"]): (row, train)
              for row, train in json.loads(args.inputs.read_text())}
    confirmed, targets = [], []
    for candidate in candidates.itertuples(index=False):
        row, training = inputs[candidate.PDB, candidate.CHAIN, candidate.SP_PRIMARY]
        source_manifest = Path(f"/home/bizon/exp53_scratch/selection_manifest/train/shard_{row['shard']:05d}.parquet")
        manifest = pq.read_table(source_manifest).to_pandas()
        source = manifest.loc[manifest.entry_id == row["entry_id"]].iloc[0]
        afpath = args.work / f"{row['entry_id']}-model_v4.cif"
        if not afpath.exists() or not afpath.stat().st_size:
            with afpath.open("wb") as stream:
                subprocess.run(["gcloud", "storage", "cat", source.gcs_uri], stdout=stream, check=True)
        # The source bucket stores gzip bytes behind a .cif object name, as in
        # the original exp53 reader. Gemmi detects gzip by suffix, so decode it.
        payload = afpath.read_bytes()
        if payload.startswith(b"\x1f\x8b"):
            afpath.write_bytes(gzip.decompress(payload))
        source_info_path = args.work / f"{row['entry_id']}-source-info.json"
        if not source_info_path.exists():
            output = subprocess.run(["gcloud", "storage", "objects", "describe", source.gcs_uri, "--format=json"], capture_output=True, text=True, check=True)
            source_info_path.write_text(output.stdout)
        source_info = json.loads(source_info_path.read_text())
        entry_path = args.work / "metadata" / f"{row['PDB']}.json"
        if not entry_path.exists():
            response = requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{row['PDB']}", timeout=30)
            response.raise_for_status()
            entry_path.parent.mkdir(exist_ok=True)
            entry_path.write_text(json.dumps(response.json()))
        entry = json.loads(entry_path.read_text())
        pdbpath = args.work / "pdb_chains" / f"{row['PDB']}_{row['CHAIN']}.cif"
        sequence = sequence_from_document(training["document"], training["seq_len"], training["n_term_index"])
        gt = compute_contacts(pdbpath, sequence, stem=f"{row['PDB']}_{row['CHAIN']}", prefer_chain=row["CHAIN"])
        af = compute_contacts(afpath, sequence, stem=row["entry_id"], prefer_chain="A")
        common = sorted(set(gt.resolved_positions) & set(af.resolved_positions))
        valid = set(common)
        pdb_pairs = {(i, j) for i, j, degree in gt.contacts if degree >= .001 and j - i >= 6 and i in valid and j in valid}
        af_pairs = {(i, j) for i, j, degree in af.contacts if degree >= .001 and j - i >= 6 and i in valid and j in valid}
        training_pairs = {tuple(sorted(((i - training["n_term_index"]) % 2000, (j - training["n_term_index"]) % 2000)))
                          for i, j in live_contacts(training["document"])}
        training_pairs = {(i, j) for i, j in training_pairs if i in valid and j in valid and j - i >= 6}
        if af_pairs != training_pairs:
            raise ValueError(f"Original AFDB coordinates disagree with the actual training document: {row['entry_id']}")
        afca = ca_coordinates(afpath, "A", sequence)
        pdbca = ca_coordinates(pdbpath, row["CHAIN"], sequence)
        ca_common = sorted(set(afca) & set(pdbca))
        ac = np.array([afca[i] for i in ca_common])
        pc = np.array([pdbca[i] for i in ca_common])
        ca_sequence = "".join(sequence[i] for i in ca_common)
        tm = tm_align(ac, pc, ca_sequence, ca_sequence)
        shared = len(af_pairs & pdb_pairs)
        release = entry["rcsb_accession_info"]["initial_release_date"]
        source_created = source_info.get("creation_time", source_info.get("timeCreated"))
        result = {"uniprot": row["SP_PRIMARY"], "entry_id": row["entry_id"], "pdb": row["PDB"], "chain": row["CHAIN"],
                  "title": entry["struct"]["title"], "release_date": release, "afdb_object_created": source_created,
                  "method": ";".join(e["method"] for e in entry["exptl"]),
                  "resolution": min(entry["rcsb_entry_info"].get("resolution_combined", [float("nan")])),
                  "L": len(sequence), "n_common": len(common), "alignment_identity": gt.alignment_identity,
                  "afdb_global_plddt": training["global_plddt"], "afdb_contacts": len(af_pairs), "pdb_contacts": len(pdb_pairs),
                  "shared_contacts": shared, "contact_f1": 2 * shared / (len(af_pairs) + len(pdb_pairs)),
                  "afdb_contact_precision": shared / len(af_pairs), "afdb_contact_recall": shared / len(pdb_pairs) if pdb_pairs else None,
                  "tm_score_common_residues": tm.tm_norm_chain1, "tm_aligned_rmsd": tm.rmsd,
                  "training_contacts_reproduced": True, "source_gcs_uri": source.gcs_uri,
                  "source_cif_sha256": hashlib.sha256(afpath.read_bytes()).hexdigest(),
                  "pdb_chain_sha256": hashlib.sha256(pdbpath.read_bytes()).hexdigest(),
                  "training_document_sha256": hashlib.sha256(training["document"].encode()).hexdigest()}
        confirmed.append(result)
        targets.append(result | {"sequence": sequence, "resolved": common, "afdb_pairs": sorted(af_pairs), "pdb_pairs": sorted(pdb_pairs),
                                 "afdb_only": sorted(af_pairs - pdb_pairs), "pdb_only": sorted(pdb_pairs - af_pairs)})
        pd.DataFrame(confirmed).to_csv(args.output_csv, index=False)
        args.output_cases.write_text(json.dumps(targets, indent=2) + "\n")
        print(json.dumps({key: result[key] for key in ("uniprot", "pdb", "chain", "contact_f1", "tm_score_common_residues", "afdb_object_created", "release_date")}), flush=True)


if __name__ == "__main__":
    main()
