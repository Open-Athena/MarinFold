#!/usr/bin/env python
"""Prepare post-score Helico reconstruction cases and verify index mapping."""

import csv
import difflib
import gzip
import json
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

from search_policy import canonical

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "exp301_evals_fold_switching_proteins" / "data"
MIRROR = Path("/data/tim/af3-db/mmcif_files")
OUT = HERE / "_cache" / "helico" / "data"
CA_CHECK_LIMIT = 14.0
CA_CHECK_MAX_FRACTION = 0.05
CASES = (
    ("1qs8b_1miqb", "branch10"),
    ("3g0ha_3ewsb", "branch5"),
    ("3j7wb_3j7vg", "branch10"),
    ("4y0mj_4xwsd", "iid"),
)


def one_letter(residue: gemmi.Residue) -> str:
    """Canonical residue letter for alignment to the union input sequence."""
    info = gemmi.find_tabulated_residue(residue.name)
    return info.one_letter_code.upper() if info and info.is_amino_acid() else "X"


def align_observed(observed: str, reference: str) -> list[int | None]:
    """Map observed polymer ranks to exp301 union-sequence positions."""
    mapping: list[int | None] = [None] * len(observed)
    matcher = difflib.SequenceMatcher(a=observed, b=reference, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"equal", "replace"}:
            for offset in range(min(i2 - i1, j2 - j1)):
                mapping[i1 + offset] = j1 + offset
    return mapping


def prepare_structure(fold: str, sequence: str) -> tuple[bytes, dict[int, int], dict[int, np.ndarray], dict]:
    """Isolate one PDB chain, create Helico CIF, and map union to token ranks."""
    pdb, chain_id = fold[:4].lower(), fold[4]
    source = MIRROR / f"{pdb}.cif"
    if not source.exists():
        raise FileNotFoundError(source)
    structure = gemmi.read_structure(str(source))
    structure.setup_entities()
    while len(structure) > 1:
        del structure[1]
    model = structure[0]
    chain = next((item for item in model if item.name == chain_id), None)
    if chain is None:
        raise ValueError(f"{fold}: chain missing")
    for other in [item.name for item in model if item.name != chain_id]:
        model.remove_chain(other)
    structure.remove_ligands_and_waters()
    structure.remove_empty_chains()
    chain = structure[0][0]
    polymer = list(chain.get_polymer())
    observed = "".join(one_letter(residue) for residue in polymer)
    mapping = align_observed(observed, sequence)
    aligned = [(rank, position) for rank, position in enumerate(mapping) if position is not None]
    identity = sum(observed[rank] == sequence[position] for rank, position in aligned) / len(aligned)
    if identity < 0.98:
        raise ValueError(f"{fold}: observed/union identity {identity:.3f}")
    union_to_rank = {position: rank for rank, position in aligned}
    if len(union_to_rank) != len(aligned):
        raise ValueError(f"{fold}: duplicate union position mapping")
    union_ca = {}
    for rank, position in aligned:
        atom = polymer[rank].find_atom("CA", "*")
        if atom is not None:
            union_ca[position] = np.array([atom.pos.x, atom.pos.y, atom.pos.z])

    chain.name = "A"
    for residue in chain:
        residue.subchain = "A"
    structure.setup_entities()
    structure.assign_label_seq_id(True)
    for residue in structure[0][0]:
        residue.subchain = "A"
    document = structure.make_mmcif_document().as_string()
    atom_rows = [line.split() for line in document.splitlines()
                 if line.startswith(("ATOM", "HETATM"))]
    labels = {row[8] for row in atom_rows}
    if "." in labels or len(labels) != len(polymer):
        raise ValueError(f"{fold}: invalid CIF label_seq_id assignment: "
                         f"{len(labels)} labels for {len(polymer)} polymer residues, "
                         f"unset={'.' in labels}")
    return gzip.compress(document.encode()), union_to_rank, union_ca, {
        "fold": fold, "observed_residues": len(polymer), "mapped_residues": len(aligned),
        "sequence_identity": identity,
    }


def verify_contacts(fold: str, pairs: set[tuple[int, int]], coordinates: dict[int, np.ndarray]) -> dict:
    """Assert exp301 true contacts still land near each other in this CIF."""
    distances = [float(np.linalg.norm(coordinates[i] - coordinates[j]))
                 for i, j in pairs if i in coordinates and j in coordinates]
    if not distances:
        raise ValueError(f"{fold}: no true contacts mapped")
    far = sum(distance > CA_CHECK_LIMIT for distance in distances)
    fraction = far / len(distances)
    if fraction > CA_CHECK_MAX_FRACTION:
        raise ValueError(f"{fold}: {far}/{len(distances)} true contacts beyond {CA_CHECK_LIMIT} A")
    return {"true_contacts_ca_checked": len(distances),
            "true_contacts_ca_far": far, "true_contacts_ca_far_fraction": fraction}


def main() -> None:
    truth = pd.read_parquet(SOURCE / "eval_targets.parquet").set_index("pair_id")
    scored = pd.read_csv(HERE / "data" / "scored_shortlists.csv")
    raw = pd.concat([pd.read_parquet(path) for path in sorted(
        (HERE / "_cache" / "raw").glob("shard-*.parquet"))], ignore_index=True
    ).set_index(["pair_id", "candidate_id"])
    (OUT / "gt").mkdir(parents=True, exist_ok=True)
    (OUT / "arms").mkdir(exist_ok=True)
    arms, targets, manifest = {}, [], []
    for pair_id, method in CASES:
        target = truth.loc[pair_id]
        subset = scored[(scored.pair_id == pair_id) & (scored.method == method)
                        & scored.finished & (scored.n_pred_fs >= 10)]
        a = subset.sort_values("phi_fs", ascending=False).iloc[0]
        b = subset.sort_values("phi_fs").iloc[0]
        if a.recall_a_fs < 0.25 or a.phi_fs < 0.10 or b.recall_b_fs < 0.25 or b.phi_fs > -0.10:
            raise ValueError(f"{pair_id}: chosen shortlist lacks both contact-level hits")
        a_raw = raw.loc[(pair_id, a.candidate_id)]
        b_raw = raw.loc[(pair_id, b.candidate_id)]
        contact_sets = {
            "true_fold1": canonical(target.contacts_fold1),
            "true_fold2": canonical(target.contacts_fold2),
            "blind_fold1": canonical(a_raw.contacts) | canonical(a_raw.given),
            "blind_fold2": canonical(b_raw.contacts) | canonical(b_raw.given),
        }
        candidate_meta = {"blind_fold1": a, "blind_fold2": b}
        if pair_id == "3j7wb_3j7vg":
            control = scored[(scored.pair_id == pair_id) & (scored.method == "iid")
                             & scored.finished & (scored.n_pred_fs >= 10)]
            iid_a = control.sort_values("phi_fs", ascending=False).iloc[0]
            iid_b = control.sort_values("phi_fs").iloc[0]
            for variant, selected in (("iid_fold1", iid_a), ("iid_fold2", iid_b)):
                record = raw.loc[(pair_id, selected.candidate_id)]
                contact_sets[variant] = canonical(record.contacts) | canonical(record.given)
                candidate_meta[variant] = selected
        for fold_number in (1, 2):
            fold = str(target[f"fold{fold_number}"])
            cif, position, coordinates, meta = prepare_structure(fold, str(target.sequence))
            check = verify_contacts(fold, contact_sets[f"true_fold{fold_number}"], coordinates)
            for variant, pairs in contact_sets.items():
                name = f"{pair_id}__gt{fold_number}__{variant}"
                mapped = sorted({(min(position[i], position[j]), max(position[i], position[j]))
                                 for i, j in pairs if i in position and j in position})
                if len(mapped) < 10:
                    raise ValueError(f"{name}: only {len(mapped)} contacts map to Helico tokens")
                (OUT / "gt" / f"{name}.cif.gz").write_bytes(cif)
                arms[name] = [list(pair) for pair in mapped]
                targets.append({"target_id": name, "dataset": "exp304_foldswitch", "stem": pair_id})
                candidate = candidate_meta.get(variant)
                manifest.append({"target_id": name, "pair_id": pair_id,
                                 "input_fold": fold_number, "variant": variant,
                                 "candidate_id": candidate.candidate_id if candidate is not None else "",
                                 "shortlist_rank": int(candidate["rank"]) if candidate is not None else 0,
                                 "n_contacts_mapped": len(mapped), **meta, **check})
    with (OUT / "targets.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target_id", "dataset", "stem"])
        writer.writeheader()
        writer.writerows(targets)
    (OUT / "arms" / "exp304.json").write_text(json.dumps(arms, indent=2))
    pd.DataFrame(manifest).to_csv(HERE / "data" / "helico_target_manifest.csv", index=False)
    print(f"prepared {len(targets)} Helico cases under {OUT}")


if __name__ == "__main__":
    main()
