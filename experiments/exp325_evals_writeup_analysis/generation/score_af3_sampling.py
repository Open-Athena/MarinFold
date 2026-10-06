"""Score every AF3 candidate, with content-addressed CPU caches and strict coverage.

uv run --project /home/bizon/git/helico --no-sync --with scikit-learn python \
  generation/score_af3_sampling.py --budget 100
"""

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import pandas as pd

from score_alphafold import HELICO, HELICO_SHA, ROOT, structure_scores


def digest(path: Path) -> str:
    """SHA-256 of one immutable scientific input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_one(task: dict) -> dict:
    """Verify inputs and cache the existing benchmark's atom-matched metrics."""
    path, truth = Path(task["structure_file"]), Path(task["ground_truth_file"])
    provenance = {"structure_sha256": digest(path), "ground_truth_sha256": digest(truth),
                  "structure_scorer_sha256": digest(Path(__file__).with_name("score_alphafold.py")),
                  "helico_sha": HELICO_SHA, "sequence": task["sequence"]}
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    cache = ROOT / "scratch/af3_sampling/score_cache" / f"{key}.json"
    if cache.exists():
        metrics = json.loads(cache.read_text())
    else:
        metrics = structure_scores(path, truth, task["sequence"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(metrics) + "\n")
    return {k: v for k, v in {**task, **provenance, **metrics}.items() if k != "sequence"}


def main() -> None:
    """Require every planned seed and retain the old 25-candidate pool separately."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, choices=[100, 1000], required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HELICO, text=True).strip() != HELICO_SHA:
        raise ValueError("Helico scoring revision changed")
    protocol_path = ROOT / "data/af3_sampling_protocol.json"
    protocol, protocol_sha = json.loads(protocol_path.read_text()), digest(protocol_path)
    targets = pd.read_csv(HELICO / "experiments/exp14_foldbench_held_out_monomers/data/targets.csv").set_index("stem")
    tasks, timings = [], []
    for record in protocol["targets"]:
        stem = record["stem"]
        truth = Path.home() / ".cache/helico/data/benchmarks/FoldBench/examples/ground_truths" / f"{targets.loc[stem, 'pdb_id']}.cif.gz"
        base = dict(stem=stem, sequence=record["sequence"], msa_depth=record["msa_depth"],
                    n_residues=record["n_residues"], eval_set=record["eval_set"], ground_truth_file=str(truth))
        directory = ROOT / "scratch/af3_sampling/results" / stem
        for seed in range(protocol["seed_start"], protocol["seed_start"] + args.budget):
            output = directory / f"seed-{seed}"
            report = json.loads((output / "complete.json").read_text())
            if (report["protocol_sha256"] != protocol_sha or report["seed"] != seed
                    or report["n_samples"] != 1 or report["n_cycles"] != 10
                    or any(report[k] != v for k, v in record.items())):
                raise ValueError(f"Changed generation protocol: {stem}, {seed}")
            tag = f"seed-{seed}_sample-0"
            path = output / tag / f"{stem.lower()}_{tag}_model.cif"
            summary = json.loads((output / tag / f"{stem.lower()}_{tag}_summary_confidences.json").read_text())
            if abs(summary["ptm"] - report["ptm"]) > 0.00501:
                raise ValueError("pTM metadata disagrees with official confidence summary")
            tasks.append(dict(**base, series="fresh_seeds", seed=seed, sample=0, structure_file=str(path),
                              ptm=report["ptm"], ranking_score=report["selection_confidence"],
                              fraction_disordered=summary["fraction_disordered"], has_clash=summary["has_clash"]))
            timings.append({k: v for k, v in report.items() if k != "sequence"})
        old = ROOT / "scratch/alphafold/results/af3" / stem
        ranking = pd.read_csv(old / f"{stem.lower()}_ranking_scores.csv")
        if set(zip(ranking.seed, ranking["sample"])) != {(s, i) for s in range(42, 47) for i in range(5)}:
            raise ValueError("Original AF3 budget incomplete")
        for row in ranking.to_dict("records"):
            tag = f"seed-{row['seed']}_sample-{row['sample']}"
            path = old / tag / f"{stem.lower()}_{tag}_model.cif"
            summary = json.loads((old / tag / f"{stem.lower()}_{tag}_summary_confidences.json").read_text())
            tasks.append(dict(**base, series="original_5x5", seed=row["seed"], sample=row["sample"],
                              structure_file=str(path), ptm=summary["ptm"], ranking_score=row["ranking_score"],
                              fraction_disordered=summary["fraction_disordered"], has_clash=summary["has_clash"]))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        scores = pd.DataFrame(pool.map(score_one, tasks))
    # Paths are relative to the experiment/publication artifact root, never tied
    # to this workstation. Hashes identify the exact structure and reference.
    scores["structure_file"] = scores.structure_file.map(lambda p: str(Path(p).relative_to(ROOT)))
    scores["ground_truth_file"] = scores.ground_truth_file.map(lambda p: Path(p).name)
    for series, name in [("fresh_seeds", "af3_sampling_samples.csv"), ("original_5x5", "af3_sampling_original.csv")]:
        scores[scores.series == series].sort_values(["stem", "seed", "sample"]).to_csv(ROOT / "data" / name, index=False)
    pd.DataFrame(timings).sort_values(["stem", "seed"]).to_csv(ROOT / "data/af3_sampling_timings.csv", index=False)
    # A real-data regression check ensures the new scorer reproduces the five
    # original selected results, without changing tolerances or atom matching.
    reference = pd.read_csv(ROOT / "data/af3_structure_metrics.csv").set_index("stem")
    for stem, group in scores[scores.series == "original_5x5"].groupby("stem"):
        selected = group.loc[group.ranking_score.idxmax()]
        if abs(selected.tm_score - reference.loc[stem, "tm_score"]) > 1e-8:
            raise ValueError(f"Original AF3 TM score failed parity: {stem}")
    manifest = dict(budget=args.budget, n_proteins=5, n_new_samples=args.budget * 5,
                    protocol_sha256=protocol_sha, helico_metric_revision=HELICO_SHA,
                    scorer_sha256=digest(Path(__file__)),
                    packages={p: importlib.metadata.version(p) for p in ["numpy", "pandas", "gemmi", "tmtools"]},
                    original_selected_tm_parity="passed all five, tolerance 1e-8",
                    tm_definition="TM-align on atom-matched CA coordinates; reference length normalization",
                    original_ptm_precision="Historical official summaries are rounded to 2 decimals; fresh seeds retain full precision.")
    (ROOT / "data/af3_sampling_scoring.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Scored {args.budget * 5} fresh candidates and 125 original candidates")


if __name__ == "__main__":
    main()
