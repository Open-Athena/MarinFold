"""Diagnose coverage, ranking and cardinality using existing exp321 rollouts.

No predictor runs here. The 81-protein partition was held out for exp321's
method selection, but is reused exploratorily here. It is not eval-test.
Within-map reranking learns contact frequencies from an independent 100-map
pool; truth is used only for scoring, never to choose the ranking or selector.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
SOURCE = EXPERIMENTS / "exp321_evals_null_sequence_contrastive_guidance_for_contacts"
sys.path.insert(0, str(SOURCE))
from analyze_results import (
    ordered_true_pairs,
    predicted_maps,
    r_precision,
    rollout_r_precision,
    vote_matrix,
)


def digest(path: Path) -> str:
    """Return a source file's SHA256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values: np.ndarray) -> tuple[float, float, float]:
    """Compute a paired-protein percentile bootstrap interval."""
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(339)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def read_frame(path: Path, stem: str, length: int, count: int) -> pd.DataFrame:
    """Validate identity and complete ordered rollout coverage."""
    frame = pd.read_parquet(path).sort_values("rollout").reset_index(drop=True)
    if len(frame) != count or frame.rollout.tolist() != list(range(count)):
        raise ValueError(f"bad rollout coverage: {path}")
    if set(frame.stem) != {stem} or set(frame.L) != {length}:
        raise ValueError(f"bad target identity: {path}")
    for row in frame.itertuples():
        if any(not (0 <= int(i) < int(j) < length) for i, j in row.contacts):
            raise ValueError(f"noncanonical contact in {path}: {row.rollout}")
    return frame


def analyze_pool(
    frame: pd.DataFrame, discovery: pd.DataFrame, truth: dict, region: str
) -> dict[str, float]:
    """Separate oracle map coverage from emission and frequency rankings."""
    maps = [list(dict.fromkeys(m)) for m in predicted_maps(frame, truth, region)]
    discovery_maps = predicted_maps(discovery, truth, region)
    true = ordered_true_pairs(truth, region)
    if not true:
        raise ValueError(f"empty truth: {truth['stem']}/{region}")
    valid = frame.finished.to_numpy(bool) & (frame.malformed_contacts.to_numpy(int) == 0)
    discovery_valid = discovery.finished.to_numpy(bool) & (
        discovery.malformed_contacts.to_numpy(int) == 0
    )
    if not discovery_valid.any():
        raise ValueError("independent ranking pool has no valid samples")
    frequencies = Counter(
        pair for m, ok in zip(discovery_maps, discovery_valid, strict=True)
        if ok for pair in set(m)
    )
    reranked = [sorted(m, key=lambda pair: (-frequencies[pair], pair)) for m in maps]
    original = np.array([rollout_r_precision(m, true) for m in maps])
    rerank_scores = np.array([rollout_r_precision(m, true) for m in reranked])
    true_counts = np.array([len(set(m) & true) for m in maps])
    sizes = np.array([len(m) for m in maps])
    # Expected precision under a uniformly random permutation of each map.
    # This is analytic, so it adds no Monte Carlo noise or expensive inference.
    random_order = true_counts / np.maximum(sizes, 1) * np.minimum(sizes, len(true)) / len(true)
    original = np.where(valid, original, 0.0)
    rerank_scores = np.where(valid, rerank_scores, 0.0)
    full_recall = np.where(valid, true_counts / len(true), 0.0)
    valid_maps = [m if ok else [] for m, ok in zip(maps, valid, strict=True)]
    union = set().union(*(set(m) for m in valid_maps))
    votes = Counter(pair for m in valid_maps for pair in set(m))
    # The selector must not use R or the resolved-residue mask: both come from
    # the reference structure. Use a sequence-length top-L budget on raw maps.
    minimum = 24 if region == "long" else 6
    raw_frequencies = Counter(
        (int(left), int(right))
        for row, ok in zip(discovery.itertuples(), discovery_valid, strict=True)
        if ok for left, right in set(map(tuple, row.contacts))
        if int(right) - int(left) >= minimum
    )
    selector = np.array([
        sum(sorted((raw_frequencies[(int(left), int(right))]
                    for left, right in set(map(tuple, row.contacts))
                    if int(right) - int(left) >= minimum), reverse=True)[:truth["L"]])
        / (int(discovery_valid.sum()) * truth["L"])
        for row in frame.itertuples()
    ])
    selector[~valid] = -np.inf
    selected = int(np.argmax(selector))
    nll = frame.native_nll.to_numpy(float, copy=True)
    nll[~valid] = np.inf
    result = {
        "consensus_published": r_precision(vote_matrix(maps, truth["L"]), truth, region),
        "consensus_valid_only": r_precision(vote_matrix(valid_maps, truth["L"]), truth, region),
        "mean_emission_fixed_r": float(original.mean()),
        "best_emission_fixed_r": float(original.max()),
        "mean_frequency_fixed_r": float(rerank_scores.mean()),
        "best_frequency_fixed_r": float(rerank_scores.max()),
        "mean_random_order_fixed_r": float(np.where(valid, random_order, 0.0).mean()),
        "best_within_map_truth_ranking": float(full_recall.max()),
        "mean_map_recall": float(full_recall.mean()),
        "union_recall": len(union & true) / len(true),
        "mean_contacts_over_r": float((sizes / len(true)).mean()),
        "fraction_maps_shorter_than_r": float((sizes < len(true)).mean()),
        "mean_cardinality_ceiling": float(np.minimum(sizes / len(true), 1).mean()),
        "best_cardinality_ceiling": float(np.minimum(sizes / len(true), 1).max()),
        "fraction_maps_longer_than_r": float((sizes > len(true)).mean()),
        "unique_maps": len({frozenset(m) for m in maps}),
        "true_contacts_seen_once_fraction": sum(votes[p] == 1 for p in true) / len(true),
        "true_contacts_seen_at_most_five_fraction": sum(0 < votes[p] <= 5 for p in true) / len(true),
        "true_contacts_seen_majority_fraction": sum(votes[p] >= 50 for p in true) / len(true),
        "blind_frequency_selected": float(rerank_scores[selected]),
        "blind_nll_selected": float(original[int(np.argmin(nll))]),
        "invalid": int((~valid).sum()),
        "R": len(true),
        "union_size": len(union),
    }
    for budget in (1, 5, 10, 25, 50, 100):
        result[f"best_emission_{budget}"] = float(original[:budget].max())
        result[f"best_map_recall_{budget}"] = float(full_recall[:budget].max())
    return result


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize target-weighted metrics and paired mechanistic contrasts."""
    rows, contrasts = [], []
    excluded = {"mode", "stem", "split", "range", "L"}
    for (mode, split, region), group in frame.groupby(["mode", "split", "range"]):
        identity = {"mode": mode, "split": split, "range": region, "n": len(group)}
        for column in sorted(set(frame.columns) - excluded):
            mean, low, high = interval(group[column].to_numpy(float))
            rows.append({**identity, "metric": column, "mean": mean, "low": low, "high": high})
        pairs = [
            ("best_frequency_fixed_r", "best_emission_fixed_r"),
            ("mean_frequency_fixed_r", "mean_emission_fixed_r"),
            ("best_within_map_truth_ranking", "best_emission_fixed_r"),
            ("union_recall", "best_within_map_truth_ranking"),
            ("blind_frequency_selected", "consensus_published"),
            ("blind_frequency_selected", "mean_frequency_fixed_r"),
            ("mean_emission_fixed_r", "mean_random_order_fixed_r"),
            ("best_emission_100", "best_emission_50"),
        ]
        for left, right in pairs:
            mean, low, high = interval((group[left] - group[right]).to_numpy(float))
            contrasts.append({**identity, "contrast": f"{left} - {right}",
                              "mean": mean, "low": low, "high": high})
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def main() -> None:
    """Run the reproducible exploratory analysis against cached public parquets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True,
                        help="exp321 cache with full_iid_single and full_g05_pa_all")
    args = parser.parse_args()
    started = time.perf_counter()
    targets_path = SOURCE / "data/targets.csv"
    targets = pd.read_csv(targets_path).query("cohort == 'eval-val'")
    if len(targets) != 97 or targets.stem.nunique() != 97:
        raise ValueError("expected 97 unique eval-val targets")
    truth_path = EXPERIMENTS / "exp245_evals_foldbench_held_out_monomers/data/gt_universe_scored.jsonl"
    truth = {}
    for line in truth_path.read_text().splitlines():
        record = json.loads(line)
        if record["stem"] in set(targets.stem) and record["dataset"] == "foldbench_monomer":
            truth[record["stem"]] = record
    if set(truth) != set(targets.stem):
        raise ValueError("truth/target mismatch")
    hashes, rows = [], []
    for i, target in enumerate(targets.itertuples()):
        iid_path = args.cache / "full_iid_single/eval-val" / f"{target.stem}.parquet"
        iid = read_frame(iid_path, target.stem, target.L, 200)
        discovery = iid.iloc[100:].copy()
        for mode in ("full_iid_single", "full_g05_pa_all"):
            path = args.cache / mode / "eval-val" / f"{target.stem}.parquet"
            frame = iid.iloc[:100].copy() if mode == "full_iid_single" else read_frame(path, target.stem, target.L, 100)
            hashes.append({"relative_path": str(path.relative_to(args.cache)),
                           "sha256": digest(path), "bytes": path.stat().st_size})
            for region in ("all", "long"):
                rows.append({"mode": mode, "stem": target.stem, "split": target.split,
                             "L": target.L, "range": region,
                             **analyze_pool(frame, discovery, truth[target.stem], region)})
        if (i + 1) % 10 == 0:
            print(f"analyzed {i + 1}/97 targets", flush=True)
    frame = pd.DataFrame(rows)
    reference = pd.read_csv(SOURCE / "data/heldout_natural.csv").query("N == 100")
    reference = reference[reference['mode'].isin(frame['mode'].unique())]
    joined = frame.merge(reference, on=["mode", "stem", "split", "range"], validate="one_to_one")
    checks = {}
    for actual, expected in [("consensus_published", "consensus_r_precision"),
                             ("best_emission_fixed_r", "validity_gated_oracle_r_precision")]:
        error = float((joined[actual] - joined[expected]).abs().max())
        if error > 1e-12 or len(joined) != 324:
            raise ValueError(f"published control failed: {actual}, {error}, {len(joined)}")
        checks[actual] = {"n": len(joined), "max_absolute_error": error}
    output = HERE / "data"
    output.mkdir(exist_ok=True)
    frame.to_csv(output / "per_protein.csv", index=False)
    summary, deltas = summarize(frame)
    summary.to_csv(output / "summary.csv", index=False)
    deltas.to_csv(output / "paired_deltas.csv", index=False)
    pd.DataFrame(hashes).to_csv(output / "raw_manifest.csv", index=False)
    provenance = {
        "checkpoint": "contacts-v1-exp277-m2-p06-full-epoch-1.5B-step-266344",
        "public_raw_prefix": "hf://buckets/open-athena/MarinFold/data/exp321/null-sequence-guidance-v1",
        "analysis": "exploratory reuse of 97 eval-val; no new eval-test scoring",
        "targets_sha256": digest(targets_path), "truth_sha256": digest(truth_path),
        "source_scorer_sha256": digest(SOURCE / "analyze_results.py"),
        "checks": checks, "raw_files": len(hashes),
        "raw_bytes": sum(row['bytes'] for row in hashes),
        "rollouts_read": 97 * 300, "new_predictor_runs": 0,
        "cpu_analysis_wall_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
