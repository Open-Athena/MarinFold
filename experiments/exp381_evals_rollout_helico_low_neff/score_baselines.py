"""Adapt native baseline CIFs to the unchanged exp381 structural scorer."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from biotite.sequence import ProteinSequence
from biotite.structure.io.pdbx import CIFFile, get_structure

from score import score_one

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    """Return file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_structure(folder: Path, target: dict) -> None:
    """Map one prediction to the exact native atoms used in original Helico scoring."""
    atoms = get_structure(
        CIFFile.read(folder / "prediction.cif"), model=1, use_author_fields=False
    )
    atoms = atoms[(atoms.chain_id == "A") & ~np.isin(atoms.element, ["H", "D"])]
    ca = atoms[atoms.atom_name == "CA"]
    if not np.array_equal(ca.res_id, np.arange(1, target["n_residues"] + 1)):
        raise ValueError(f"{folder.name}: unexpected residue identities")
    sequence = "".join(
        ProteinSequence.convert_letter_3to1(name) for name in ca.res_name
    )
    if sequence != target["sequence"]:
        raise ValueError(f"{folder.name}: predicted sequence mismatch")
    keys = [
        (int(rid) - 1, str(name))
        for rid, name in zip(atoms.res_id, atoms.atom_name, strict=True)
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate prediction atom identities")
    lookup = dict(zip(keys, atoms.coord, strict=True))
    source = HERE / "scratch/results" / f"{target['stem']}__default__r00/structure.npz"
    original_metadata = json.loads((source.parent / "result.json").read_text())
    if sha256(source) != original_metadata["output_sha256"]["structure.npz"]:
        raise ValueError("Original reference-containing coordinate archive changed")
    with np.load(source, allow_pickle=False) as original:
        arrays = {
            key: original[key]
            for key in ("ref_coords", "ref_atom_names", "ref_res_ids", "ref_elements")
        }
    aligned = np.full_like(arrays["ref_coords"], np.nan)
    covered = np.zeros(len(aligned), dtype=bool)
    for i, key in enumerate(
        zip(arrays["ref_res_ids"], arrays["ref_atom_names"], strict=True)
    ):
        if key in lookup:
            aligned[i] = lookup[key]
            covered[i] = True
    np.savez_compressed(
        folder / "structure.npz",
        **arrays,
        aligned_pred=aligned,
        covered=covered,
        coords=atoms.coord,
        atom_names=atoms.atom_name,
        res_ids=atoms.res_id - 1,
        elements=atoms.element,
        res_names=atoms.res_name,
    )


def main(partial: bool) -> None:
    """Validate identities, score predictions and preserve per-input runtime records."""
    tasks = json.loads((HERE / "inputs/baselines/tasks.json").read_text())
    plan_sha = sha256(HERE / "data/baseline_plan.json")
    build = json.loads((HERE / "data/tmscore_build.json").read_text())
    if sha256(HERE / "scratch/TMscore") != build["binary_sha256"]:
        raise ValueError("TMscore executable digest mismatch")
    rows, timings = [], []
    runtime = {}
    for task in tasks:
        folder = HERE / "scratch/baselines/results" / task["task_id"]
        if partial and not (folder / "result.json").exists():
            continue
        row = json.loads((folder / "result.json").read_text())
        model_runtime = dict(
            versions=row["versions"], weights_sha256=row["weights_sha256"]
        )
        if row["model"] in runtime and runtime[row["model"]] != model_runtime:
            raise ValueError("Model runtime or weights changed within a baseline arm")
        runtime[row["model"]] = model_runtime
        if row["plan_sha256"] != plan_sha or any(
            row[key] != val for key, val in task.items()
        ):
            raise ValueError("Input identity mismatch")
        if sha256(folder / "prediction.cif") != row["prediction_sha256"]:
            raise ValueError("Prediction hash mismatch")
        prepare_structure(folder, task)
        metrics = score_one(folder)
        if (
            metrics["ca_coverage"] != 1
            or metrics["n_reference_ca"] != task["n_residues"]
        ):
            raise ValueError("Incomplete reference CA coverage")
        scalar = {k: v for k, v in row.items() if not isinstance(v, dict)}
        # Raw runner wall time covers the call; present setup-inclusive totals
        # in the repository's common timing schema without changing raw records.
        scalar["call_total_seconds"] = scalar["total_seconds"]
        scalar["total_seconds"] += scalar["model_load_seconds"]
        scalar["timing_scope"] = (
            "total=call+shared setup; deduplicate setup by worker; "
            "AF3 inference includes cold-shape JIT; ESM fold includes preparation"
        )
        rows.append({**scalar, **metrics})
        timings.append(scalar)
    if not rows:
        raise ValueError("No predictions found")
    prefix = "baseline_smoke" if partial else "baseline"
    pd.DataFrame(rows).to_csv(HERE / f"data/{prefix}_metrics.csv", index=False)
    pd.DataFrame(timings).to_csv(HERE / f"data/{prefix}_timings.csv", index=False)
    (HERE / f"data/{prefix}_runtime.json").write_text(
        json.dumps(runtime, indent=2) + "\n"
    )
    print(
        pd.DataFrame(rows)
        .groupby("model")[["tm_score", "lddt", "gdt_ts", "atom_coverage"]]
        .agg(["count", "mean"])
        .to_string()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true")
    main(parser.parse_args().partial)
