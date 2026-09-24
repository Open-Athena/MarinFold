"""Score every original ESMFold2 decoy with the same metrics as AF2/AF3/Boltz2."""

import concurrent.futures
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import pandas as pd

from score_alphafold import HELICO, HELICO_SHA, ROOT, structure_scores


def score_one(record: dict) -> dict:
    """Verify one generated structure and compare it with experimental protein atoms."""
    stem, seed = record["stem"], record["map_seed"]
    path = ROOT / record["input_structure"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record["structure_sha256"]:
        raise ValueError(f"Structure digest mismatch: {path}")
    metadata = json.loads(path.with_name("result.json").read_text())
    if metadata["stem"] != stem or metadata["seed"] != seed:
        raise ValueError(f"Prediction identity mismatch: {path}")
    truth = ROOT / "scratch/helico/structured/gt" / f"{stem}.cif.gz"
    metrics = structure_scores(path, truth, metadata["sequence"])
    if metrics["n_matched_ca"] != len(metadata["sequence"]):
        raise ValueError(f"Incomplete backbone coverage: {stem}/{seed}")
    return {"stem": stem, "map_seed": seed, "arm": "esmfold2", "structure_sha256": digest,
            "ground_truth_sha256": hashlib.sha256(truth.read_bytes()).hexdigest(),
            "input_structure": record["input_structure"], **metrics}


def main() -> None:
    """Cache all 500 accuracies without selecting samples or changing predictor inference."""
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HELICO, text=True).strip() != HELICO_SHA:
        raise ValueError("Helico metrics differ from the pinned implementation")
    source = ROOT / "data/structured_decoy_maps.csv"
    maps = pd.read_csv(source).query("arm == 'esmfold2'")
    if len(maps) != 500 or maps.duplicated(["stem", "map_seed"]).any():
        raise ValueError("Expected exactly 500 seeded ESMFold2 maps")
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(score_one, maps.to_dict("records")))
    pd.DataFrame(results).sort_values(["stem", "map_seed"]).to_csv(ROOT / "data/esmfold2_decoy_structure_metrics.csv", index=False)
    provenance = {
        "n_targets": 5, "n_structures": len(results), "metric_source_sha": HELICO_SHA,
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shared_scorer_sha256": hashlib.sha256((Path(__file__).parent / "score_alphafold.py").read_bytes()).hexdigest(),
        "map_metadata_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "metric": "GDT-TS/TM/RMSD on matched protein CA; lDDT on matched protein atoms",
        "accuracy_object": "original ESMFold2 structure; distinct from downstream Helico accuracy",
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "gemmi", "tmtools", "scipy")},
    }
    (ROOT / "data/esmfold2_decoy_scores_run.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Scored {len(results)} original ESMFold2 structures")


if __name__ == "__main__":
    main()
