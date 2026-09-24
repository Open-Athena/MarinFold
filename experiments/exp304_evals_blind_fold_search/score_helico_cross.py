#!/usr/bin/env python
"""Score each Helico prediction against both experimental fold references."""

import gzip
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import tmtools

from build_helico_targets import SOURCE, align_observed, one_letter, prepare_structure

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PREDICTIONS = HERE / "_cache" / "helico" / "results" / "predictions" / "exp304-validation"


def prediction_coords(path: Path, sequence: str) -> dict[int, np.ndarray]:
    """Map Helico's predicted Cα atoms back to exp301 union positions."""
    structure = gemmi.read_pdb_string(gzip.decompress(path.read_bytes()).decode())
    structure.setup_entities()
    polymer = list(structure[0][0].get_polymer())
    observed = "".join(one_letter(residue) for residue in polymer)
    mapping = align_observed(observed, sequence)
    aligned = [(rank, position) for rank, position in enumerate(mapping) if position is not None]
    identity = sum(observed[rank] == sequence[position] for rank, position in aligned) / len(aligned)
    if identity < 0.98:
        raise ValueError(f"{path.name}: predicted/union identity {identity:.3f}")
    result = {}
    for rank, position in aligned:
        atom = polymer[rank].find_atom("CA", "*")
        if atom is not None:
            result[position] = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
    return result


def tm_score(pred: np.ndarray, reference: np.ndarray) -> float:
    """TM-score normalized to the same matched reference positions."""
    length = len(pred)
    if length < 3:
        return float("nan")
    result = tmtools.tm_align(pred.astype(np.float64), reference.astype(np.float64),
                              "A" * length, "A" * length)
    return float(result.tm_norm_chain2)


def lddt(pred: np.ndarray, reference: np.ndarray, region_mask: np.ndarray | None = None) -> float:
    """Cα lDDT; optionally keep local pairs touching the switching region."""
    pred_dist = np.linalg.norm(pred[:, None] - pred[None, :], axis=-1)
    ref_dist = np.linalg.norm(reference[:, None] - reference[None, :], axis=-1)
    mask = (ref_dist < 15) & (ref_dist > 0.01)
    if region_mask is not None:
        mask &= region_mask[:, None] | region_mask[None, :]
    if not mask.any():
        return float("nan")
    difference = np.abs(pred_dist[mask] - ref_dist[mask])
    return float(np.mean([(difference < threshold).mean() for threshold in (0.5, 1, 2, 4)]))


def main() -> None:
    manifest = pd.read_csv(DATA / "helico_target_manifest.csv")
    truth = pd.read_parquet(SOURCE / "eval_targets.parquet").set_index("pair_id")
    own = pd.read_csv(HERE / "_cache" / "helico" / "results" / "exp304-validation.csv")
    if len(own) != len(manifest) or not (own.status == "ok").all():
        raise ValueError("Helico predictions incomplete")
    rows = []
    for pair_id, group in manifest.groupby("pair_id"):
        target = truth.loc[pair_id]
        sequence = str(target.sequence)
        references = {}
        for fold_number in (1, 2):
            _, _, coordinates, _ = prepare_structure(str(target[f"fold{fold_number}"]), sequence)
            references[fold_number] = coordinates
        common = {int(position) for position in target.common_positions}
        lo, hi = int(target.fs_lo), int(target.fs_hi)
        for record in group.itertuples():
            predicted = prediction_coords(PREDICTIONS / f"{record.target_id}.pdb.gz", sequence)
            positions = sorted(common & predicted.keys() & references[1].keys() & references[2].keys())
            if len(positions) < 20:
                raise ValueError(f"{record.target_id}: only {len(positions)} common positions")
            pred = np.array([predicted[position] for position in positions])
            region = np.array([lo <= position < hi for position in positions])
            for reference_fold in (1, 2):
                ref = np.array([references[reference_fold][position] for position in positions])
                rows.append({"target_id": record.target_id, "pair_id": pair_id,
                             "input_fold": record.input_fold, "variant": record.variant,
                             "reference_fold": reference_fold,
                             "n_common_ca": len(positions), "n_region_ca": int(region.sum()),
                             "tm_common": tm_score(pred, ref),
                             "tm_region": tm_score(pred[region], ref[region]),
                             "lddt_common": lddt(pred, ref),
                             "lddt_region_touch": lddt(pred, ref, region)})
    frame = pd.DataFrame(rows).sort_values(["pair_id", "input_fold", "variant", "reference_fold"])
    frame.to_csv(DATA / "helico_cross_reference.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
