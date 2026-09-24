"""Extract predictor and oracle contacts using the identical Helico geometry.

Run in the pinned Helico CPU environment. Full positive/negative maps share
the oracle's eligible-pair mask; contact counts may differ. Every map keeps
its ESMFold2 seed and structure digest. Rendering never repeats this work.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from helico.bench import oracle_contact_state, structure_to_chains
from helico.contacts import load_rotamer_library
from helico.data import parse_ccd, parse_mmcif, tokenize_sequences

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "scratch/helico/confidence"
DESTINATION = ROOT / "scratch/helico/structured"
PREDICTIONS = ROOT / "scratch/structured_decoys/esmfold2"
HELICO_SHA = "b10385d736673c81b10e70d1099962af6f2573c0"


def main() -> None:
    """Fail on missing coordinates or identity before freezing 505 contact maps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    prediction_root = PREDICTIONS.parent / "smoke/esmfold2" if args.smoke else PREDICTIONS
    if subprocess.check_output(["git", "-C", "/home/bizon/git/helico", "rev-parse", "HEAD"], text=True).strip() != HELICO_SHA:
        raise ValueError("Helico source differs from pinned revision")
    low = pd.read_csv(ROOT / "data/confidence_targets.csv")
    low = low[low.msa_depth < 10]
    targets = pd.read_csv(SOURCE / "targets.csv")
    targets = targets[targets.target_id.isin(low.stem)].sort_values("target_id")
    if len(targets) != 5:
        raise ValueError("Expected exactly five low-depth natural proteins")
    if args.smoke:
        targets = targets[targets.target_id.isin([p.name for p in prediction_root.iterdir()])]
    destination = DESTINATION / "smoke" if args.smoke else DESTINATION
    (destination / "maps").mkdir(parents=True, exist_ok=True)
    (destination / "gt").mkdir(exist_ok=True)
    ccd, rotamers = parse_ccd(), load_rotamer_library()
    records = []
    for target in targets.to_dict("records"):
        stem = target["target_id"]
        gt_path = SOURCE / "gt" / f"{stem}.cif.gz"
        gt = parse_mmcif(gt_path, max_resolution=float("inf"))
        if gt is None:
            raise ValueError(f"Cannot parse {gt_path}")
        chains = structure_to_chains(gt)
        protein = [c for c in chains if c["type"] == "protein"]
        if len(protein) != 1 or protein[0]["sequence"] != target["input_seq"]:
            raise ValueError(f"Ground-truth sequence mismatch: {stem}")
        tokenized = tokenize_sequences(chains, ccd)
        oracle_tensor = oracle_contact_state(gt, tokenized, rotamers)
        if oracle_tensor is None:
            raise ValueError(f"Oracle extraction failed: {stem}")
        oracle = oracle_tensor.numpy()
        known = oracle != 0
        maps = {"oracle-0": oracle}
        paths = sorted((prediction_root / stem).glob("*/structure.cif"), key=lambda p: int(p.parent.name))
        expected = 2 if args.smoke else 100
        if len(paths) != expected or {int(p.parent.name) for p in paths} != set(range(expected)):
            raise ValueError(f"Missing seeded predictions: {stem}")
        for path in paths:
            seed = int(path.parent.name)
            metadata = json.loads(path.with_name("result.json").read_text())
            if not np.isfinite([metadata["ptm"], metadata["mean_plddt"]]).all():
                raise ValueError(f"Non-finite ESMFold2 confidence: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != metadata["structure_sha256"] or metadata["sequence"] != target["input_seq"]:
                raise ValueError(f"Input or output identity mismatch: {path}")
            prediction = parse_mmcif(path, max_resolution=float("inf"))
            if prediction is None:
                raise ValueError(f"Cannot parse {path}")
            predicted_chains = structure_to_chains(prediction)
            if len(predicted_chains) != 1 or predicted_chains[0]["sequence"] != target["input_seq"]:
                raise ValueError(f"Prediction sequence differs: {path}")
            state_tensor = oracle_contact_state(prediction, tokenized, rotamers)
            if state_tensor is None:
                raise ValueError(f"Predicted contact extraction failed: {path}")
            state = state_tensor.numpy()
            if np.any((state == 0) & known):
                raise ValueError(f"Predicted structure lacks oracle-eligible pairs: {path}")
            state = np.where(known, state, 0).astype(oracle.dtype)
            maps[f"esmfold2-{seed}"] = state
            records.append({"stem": stem, "arm": "esmfold2", "map_seed": seed,
                            "structure_sha256": digest, "input_structure": str(path.relative_to(ROOT)),
                            "esmfold2_ptm": metadata["ptm"], "esmfold2_mean_plddt": metadata["mean_plddt"]})
        records.append({"stem": stem, "arm": "oracle", "map_seed": 0,
                        "structure_sha256": hashlib.sha256(gt_path.read_bytes()).hexdigest(),
                        "input_structure": str(gt_path.relative_to(ROOT)),
                        "esmfold2_ptm": None, "esmfold2_mean_plddt": None})
        for record in records:
            if record["stem"] != stem:
                continue
            state = maps[f"{record['arm']}-{record['map_seed']}"]
            positive, truth = np.triu(state == 2, 1), np.triu(oracle == 2, 1)
            tp = int((positive & truth).sum())
            union = int((positive | truth).sum())
            record.update({"map_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                           "n_present": int(positive.sum()), "n_absent": int(np.triu(state == 1, 1).sum()),
                           "n_unknown": int(np.triu(state == 0, 1).sum()),
                           "contact_precision": tp / positive.sum() if positive.any() else 0.0,
                           "contact_recall": tp / truth.sum(), "contact_jaccard": tp / union,
                           "msa_depth": int(low.set_index("stem").loc[stem, "msa_depth"])})
        np.savez_compressed(destination / "maps" / f"{stem}.npz", **maps)
        shutil.copyfile(gt_path, destination / "gt" / gt_path.name)
        print(f"{stem}: {len(maps)} maps, {len(set(hashlib.sha256(m.tobytes()).hexdigest() for k,m in maps.items() if k != 'oracle-0'))} distinct decoys", flush=True)
    targets.to_csv(destination / "targets.csv", index=False)
    (destination / "ranked_pairs.json").write_text("{}")
    table = pd.DataFrame(records).sort_values(["stem", "arm", "map_seed"])
    table.to_csv(destination / "map_metadata.csv", index=False)
    if not args.smoke:
        table.to_csv(ROOT / "data/structured_decoy_maps.csv", index=False)
        manifest = {"helico_source_sha": HELICO_SHA, "n_targets": 5, "n_maps": len(records),
                    "contact_definition": "Helico oracle_contact_state, pyconfind 0.6.0 native_only; contact_distance=3A, min_degree=.001, minimum sequence separation=6",
                    "mask": "oracle eligible-pair mask shared by all maps; absent and present counts may vary",
                    "files_sha256": {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.rglob("*")) if p.is_file() and "smoke" not in p.parts}}
        (ROOT / "data/helico_structured_inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
