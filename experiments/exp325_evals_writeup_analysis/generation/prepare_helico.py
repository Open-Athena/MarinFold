"""Prepare verified Helico inputs using the frozen exp14 coordinate mapping."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HELICO_SHA = "b10385d736673c81b10e70d1099962af6f2573c0"


def main() -> None:
    """Rebuild index maps, require the archived verification, and freeze input bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["confidence", "folding"], required=True)
    parser.add_argument("--helico-repo", type=Path, default=Path("/home/bizon/git/helico"))
    args = parser.parse_args()
    repo = args.helico_repo
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != HELICO_SHA:
        raise ValueError("Helico checkout differs from pinned source revision")
    upstream = repo / "experiments/exp14_foldbench_held_out_monomers"
    spec = importlib.util.spec_from_file_location("index_mapping", upstream / "build_index_map.py")
    mapping_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mapping_module)
    truth_path = ROOT / "scratch/contacts/inputs/gt_universe_scored.jsonl"
    if hashlib.sha256(truth_path.read_bytes()).hexdigest() != "f30c23e3d2fbab245755fc01548388b41730ddfa45da87325539698cadb153e5":
        raise ValueError("Ground-truth universe has changed")
    truth = {r["stem"]: r for r in map(json.loads, truth_path.read_text().splitlines())}
    targets = pd.read_csv(upstream / "data/targets.csv")
    report = pd.read_csv(upstream / "data/index_map_report.csv").set_index("target_id")
    prompts = pd.read_csv(ROOT.parents[0] / "exp245_evals_foldbench_held_out_monomers/data/eval_sets.csv").set_index("stem")
    wanted_path = ROOT / "data" / ("confidence_targets.csv" if args.phase == "confidence" else "missing_eval_test_targets.csv")
    wanted = set(pd.read_csv(wanted_path).stem)
    targets = targets[targets.stem.isin(wanted)].sort_values("stem")
    if len(targets) != len(wanted):
        raise ValueError("Missing target metadata")
    destination = ROOT / "scratch/helico" / args.phase
    (destination / "gt").mkdir(parents=True, exist_ok=True)
    rankings, pins, mappings = {}, {}, {}
    excluded = []
    scores = None
    if args.phase == "folding":
        scores = pd.concat([pd.read_parquet(p) for p in sorted((ROOT / "scratch/contacts/results/scores").glob("*.parquet"))])
    for target in targets.itertuples():
        stem = target.stem
        record = truth[stem]
        mapping, rule = mapping_module.build_map(target.input_seq, prompts.loc[stem, "sequence"], record["resolved"])
        verified = report.loc[stem]
        if not verified.ok or not mapping or rule != verified.rule or len(mapping) != verified.n_mapped:
            if args.phase == "folding":
                excluded.append(stem)
                continue
        mappings[stem] = mapping
        if scores is not None:
            sparse = scores[scores.stem == stem]
            if sparse.empty:
                raise ValueError(f"Missing votes for {stem}")
            matrix = np.zeros((record["L"], record["L"]), dtype=np.int16)
            matrix[sparse.i.to_numpy(), sparse.j.to_numpy()] = sparse.votes
            resolved = np.asarray(record["resolved"], dtype=int)
            a, b = np.triu_indices(len(resolved), 1)
            i, j = resolved[a], resolved[b]
            keep = j - i >= 6
            i, j = i[keep], j[keep]
            order = np.argsort(-matrix[i, j], kind="mergesort")[:record["L"]]
            rankings[stem] = [[mapping[int(i[k])], mapping[int(j[k])]] for k in order]
            if len(rankings[stem]) != record["L"]:
                raise ValueError(f"{stem}: cannot supply exactly top-L contacts")
        source = Path.home() / ".cache/helico/data/benchmarks/FoldBench/examples/ground_truths" / f"{target.pdb_id}.cif.gz"
        target_path = destination / "gt" / f"{stem}.cif.gz"
        shutil.copyfile(source, target_path)
        pins[stem] = hashlib.sha256(target_path.read_bytes()).hexdigest()
    targets = targets[~targets.stem.isin(excluded)]
    targets.to_csv(destination / "targets.csv", index=False)
    (destination / "ranked_pairs.json").write_text(json.dumps(rankings))
    (destination / "token_map.json").write_text(json.dumps(mappings))
    (ROOT / "data" / f"helico_{args.phase}_inputs.json").write_text(json.dumps({
        "source_sha": HELICO_SHA, "phase": args.phase, "n_targets": len(targets),
        "mapping_control": "exp14 frozen index_map_report.csv; identical rule and mapped residue count",
        "files_sha256": pins, "excluded_unverified_mapping": excluded,
    }, indent=2) + "\n")
    print(f"Prepared {len(targets)} targets for {args.phase}: {destination}")


if __name__ == "__main__":
    main()
