"""Compute the exact-margin shuffle controls with resumable CPU workers."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import pandas as pd

from analysis_stats import curveball, diversity
from analyze import candidate_sets, load

HERE = Path(__file__).resolve().parent


def run_one(task: tuple[str, str, dict, int]) -> list[dict]:
    """Evaluate one protein/model and cache only hash-identified complete results."""
    model, stem, truth, repeats = task
    source = HERE / "scratch/raw" / model / f"{stem}-t1.0-fixed.json.gz"
    digest = hashlib.sha256(
        source.read_bytes()
        + (HERE / "analysis_stats.py").read_bytes()
        + Path(__file__).read_bytes()
    ).hexdigest()
    cache = HERE / "scratch/shuffle" / f"{model}-{stem}-{repeats}.json"
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved["sha256"] == digest:
            return saved["rows"]
    record = load(source)
    sets, _, _, _ = candidate_sets(record, truth)
    actual = diversity(sets, truth["L"] ** 2)
    frequencies = Counter(x for s in sets for x in s)
    rows = []
    for repeat in range(repeats):
        randomized = curveball(sets, 373 + repeat, 10 * len(sets))
        assert [len(s) for s in sets] == [len(s) for s in randomized]
        assert frequencies == Counter(x for s in randomized for x in s)
        null = diversity(randomized, truth["L"] ** 2)
        half = diversity(curveball(sets, 1373 + repeat, 5 * len(sets)), truth["L"] ** 2)
        rows.append(
            {
                "stem": stem,
                "model": model,
                "eval_set": record["target"]["eval_set"],
                "repeat": repeat,
                "observed_jaccard_sd": actual["jaccard_sd"],
                "null_jaccard_sd": null["jaccard_sd"],
                "half_chain_jaccard_sd": half["jaccard_sd"],
                "observed_overlap_sd": actual["overlap_sd"],
                "null_overlap_sd": null["overlap_sd"],
            }
        )
    cache.write_text(json.dumps({"sha256": digest, "rows": rows}))
    return rows


def main() -> None:
    """Write a final table only when every expected protein/model is present."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    truth = json.loads((HERE / "inputs/ground_truth.json").read_text())
    tasks = [
        (model, r["stem"], r, args.repeats)
        for r in truth
        for model in ["default", "mpnn"]
        if (HERE / "scratch/raw" / model / f"{r['stem']}-t1.0-fixed.json.gz").exists()
    ]
    if not args.allow_partial:
        assert len(tasks) == 232
    (HERE / "scratch/shuffle").mkdir(exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(run_one, tasks), 1):
            rows.extend(result)
            if index % 10 == 0:
                print("SHUFFLED", index, "/", len(tasks), flush=True)
    destination = HERE / ("scratch/partial" if args.allow_partial else "data")
    destination.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(destination / "shuffle_controls.csv", index=False)
    if not args.allow_partial:
        path = HERE / "data/completeness.json"
        complete = json.loads(path.read_text())
        complete["shuffle_replicates"] = args.repeats
        complete["shuffle_protein_models"] = len(tasks)
        path.write_text(json.dumps(complete, indent=2) + "\n")


if __name__ == "__main__":
    main()
