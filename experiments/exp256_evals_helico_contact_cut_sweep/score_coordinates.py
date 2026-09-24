# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Independently reproduce saved Helico lDDT from exported coordinates on CPU.

The original runner matches heavy atoms by chain, sequential residue position,
and atom name, then averages the 0.5/1/2/4-A distance-threshold indicators over
all matched atom pairs with reference distance in (0.01, 15) A. It includes
within-residue pairs. This implementation reads the published mmCIF/PDB files
directly and uses a sparse neighbor search instead of Helico's dense distance
matrices. PDB coordinates were rounded to 0.001 A; exact score identity is not
expected. No Helico imports, model loading, or inference are required.
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from scipy.spatial import cKDTree

from analyze import EVAL_SET, load_arm, paired

DEFAULT_ARMS = ("mf_L", "mf_1p5L", "mf_union")


def ground_truth(path: Path) -> tuple[dict, dict]:
    """Read the original runner's mmCIF heavy-atom/residue matching universe."""
    with gzip.open(path, "rt") as handle:
        cif = MMCIF2Dict(handle)
    count = len(cif["_atom_site.id"])
    columns = {name: cif[f"_atom_site.{name}"] for name in (
        "label_atom_id", "type_symbol", "Cartn_x", "Cartn_y", "Cartn_z",
        "label_comp_id", "label_asym_id", "label_seq_id")}
    models = cif.get("_atom_site.pdbx_PDB_model_num", ["1"] * count)
    alternates = cif.get("_atom_site.label_alt_id", ["."] * count)
    entity_types = dict(zip(cif["_entity.id"], cif["_entity.type"]))
    chain_types = {chain: entity_types[entity] for chain, entity in zip(cif["_struct_asym.id"], cif["_struct_asym.entity_id"])}
    residue_positions = {}
    atoms = {}
    for index in range(count):
        if models[index] != "1" or alternates[index] not in (".", "A", "?"):
            continue
        if columns["type_symbol"][index] == "H" or columns["label_comp_id"][index] == "HOH":
            continue
        chain = columns["label_asym_id"][index]
        residue = (columns["label_comp_id"][index], columns["label_seq_id"][index])
        chain_positions = residue_positions.setdefault(chain, {})
        if residue not in chain_positions:
            chain_positions[residue] = len(chain_positions)
        key = (chain, chain_positions[residue], columns["label_atom_id"][index])
        atoms[key] = np.array([float(columns[f"Cartn_{axis}"][index]) for axis in "xyz"], dtype=np.float32)
    for chain, positions in residue_positions.items():
        if chain_types[chain] == "non-polymer" and len(positions) != 1:
            raise ValueError(f"{path}: ambiguous multi-residue ligand chain {chain}")
    return atoms, {chain: chain_types[chain] for chain in residue_positions}


def matched_coordinates(path: Path, truth: dict, chain_types: dict) -> tuple[np.ndarray, np.ndarray]:
    """Match exported PDB atoms to mmCIF by the original sequential convention."""
    pred, gt = [], []
    seen = set()
    # Helico's PDB exporter preserves multi-character chain names, expanding
    # the fixed-width layout by this offset instead of truncating the name.
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if not line.startswith("ATOM  "):
                continue
            chains = [chain for chain in chain_types if line[21:].startswith(chain)]
            if not chains:
                raise ValueError(f"{path}: unexpected exported chain")
            chain = max(chains, key=len)
            extra = len(chain) - 1
            # A ligand has one token per atom in the PDB export, but the
            # runner groups these tokens by one shared residue UID.
            position = 0 if chain_types[chain] == "non-polymer" else int(line[22 + extra:26 + extra]) - 1
            key = (chain, position, line[12:16].strip())
            if key not in truth:
                continue
            if key in seen:
                raise ValueError(f"{path}: duplicate predicted atom {key}")
            seen.add(key)
            pred.append([float(line[start + extra:start + extra + 8]) for start in (30, 38, 46)])
            gt.append(truth[key])
    return np.asarray(pred, dtype=np.float32), np.asarray(gt, dtype=np.float32)


def sparse_lddt(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Helico's pair-weighted all-atom lDDT without a dense N² array."""
    pairs = cKDTree(gt).query_pairs(15.0, output_type="ndarray")
    reference = np.linalg.norm(gt[pairs[:, 0]] - gt[pairs[:, 1]], axis=1)
    keep = (reference > 0.01) & (reference < 15.0)
    pairs, reference = pairs[keep], reference[keep]
    if not len(pairs):
        raise ValueError("lDDT requires nonempty reference neighbor pairs")
    predicted = np.linalg.norm(pred[pairs[:, 0]] - pred[pairs[:, 1]], axis=1)
    error = np.abs(predicted - reference)
    return float(np.mean([np.mean(error < threshold) for threshold in (0.5, 1.0, 2.0, 4.0)]))


def main() -> None:
    """Rescore selected arms and save per-target and paired validation tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    args = parser.parse_args()
    if "mf_L" not in args.arms or len(set(args.arms)) != len(args.arms):
        raise ValueError("coordinate audit needs unique arms including the mf_L baseline")
    targets = pd.read_csv(args.inputs / "targets.csv")
    if not targets.eval_set.eq(EVAL_SET).all():
        raise ValueError("coordinate audit only accepts eval-val targets")
    ids = set(targets.target_id)
    rows = []
    for tag in args.arms:
        results = load_arm(args.inputs / "results", tag, ids)
        for row in results[results.status.eq("ok")].itertuples():
            truth, chain = ground_truth(args.inputs / "gt" / f"{row.target_id}.cif.gz")
            pred, gt = matched_coordinates(args.inputs / "predictions" / tag / f"{row.target_id}.pdb.gz", truth, chain)
            if len(pred) != row.n_matched_atoms:
                raise ValueError(f"{tag}/{row.target_id}: matched {len(pred)} atoms, expected {row.n_matched_atoms}")
            score = sparse_lddt(pred, gt)
            rows.append(dict(target_id=row.target_id, arm=tag, n_matched_atoms=len(pred),
                             saved_lddt=row.lddt, coordinate_lddt=score, difference=score - row.lddt))
        print(f"Rescored {tag}: {sum(row['arm'] == tag for row in rows)} targets", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "coordinate_validation.csv", index=False)
    wide = frame.pivot(index="target_id", columns="arm", values="coordinate_lddt").dropna()
    comparisons = [dict(arm=tag, n=len(wide), **paired(wide, tag, "mf_L"))
                   for tag in args.arms if tag != "mf_L"]
    summary = dict(n_rows=len(frame), max_absolute_lddt_difference=float(frame.difference.abs().max()),
                   mean_absolute_lddt_difference=float(frame.difference.abs().mean()),
                   paired_comparisons=comparisons)
    (args.out / "coordinate_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
