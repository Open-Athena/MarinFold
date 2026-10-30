"""Reduce saved individual samples into reproducible protein-level source tables."""

import argparse
from collections import Counter
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_stats import curveball, diversity, thin_matched
from contacts import parse_contacts

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location(
    "canonical_metrics",
    REPO / "experiments/exp89_evals_contacts_v1_model_on_eval_set/compute_metrics.py",
)
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


def load(path: Path) -> dict:
    """Read one compressed raw result."""
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def candidate_sets(
    record: dict, truth: dict
) -> tuple[list[set[int]], set[int], np.ndarray, np.ndarray]:
    """Validate raw canonicalization and select exactly exp89's candidate pairs."""
    length = truth["L"]
    pi, pj, sep = METRICS.resolved_pairs(np.array(truth["resolved"]))
    keep = sep >= 6
    ids = pi[keep] * length + pj[keep]
    candidates = set(ids.tolist())
    tmat = METRICS.true_matrix(length, truth["contacts"])
    labels = tmat[pi[keep], pj[keep]]
    true = set(ids[labels].tolist())
    sets = []
    for sample in record["samples"]:
        parsed = parse_contacts(sample["text"], sample["offset"], length)
        assert [list(p) for p in parsed["contacts"]] == sample["contacts"]
        sets.append({i * length + j for i, j in sample["contacts"]} & candidates)
    return sets, true, ids, labels


def r_precision(votes: np.ndarray, ids: np.ndarray, labels: np.ndarray) -> float:
    """Use the canonical scorer's stable frequency ranking and true-contact cut."""
    n = int(labels.sum())
    if n == 0:
        return float("nan")
    return float(labels[np.argsort(-votes[ids], kind="mergesort")[:n]].mean())


def analyze_one(
    record: dict,
    truth: dict,
    sets: list[set[int]],
    true: set[int],
    ids: np.ndarray,
    labels: np.ndarray,
) -> tuple[list, list, list, list]:
    """Compute distribution statistics, discovery curves and full final metrics."""
    length = truth["L"]
    base = {k: record[k] for k in ["model", "temperature", "mode"]}
    base.update(
        stem=truth["stem"],
        eval_set=record["target"]["eval_set"],
        L=length,
        n_rollouts=len(sets),
    )
    diversity_rows, curves, diagnostics, metric_rows = [], [], [], []
    unrestricted = [
        {i * length + j for i, j in s["contacts"]} for s in record["samples"]
    ]
    diversity_rows.append(
        {
            **base,
            "range": "all-valid",
            "block": "pooled",
            **diversity(unrestricted, length * length),
        }
    )
    for sample, s in zip(record["samples"], sets, strict=True):
        diagnostics.append(
            {
                **base,
                "rollout": sample["rollout"],
                "block": sample["block"],
                "contacts": len(s),
                "valid_contacts_unrestricted": len(sample["contacts"]),
                "precision": len(s & true) / len(s) if s else 0.0,
                "recall": len(s & true) / len(true) if true else float("nan"),
                "generated_tokens": len(sample["token_ids"]),
                "initial_tokens": sample["initial_tokens"],
                "unfinished": sample["finish_reason"] != "stop",
                "extended": sample["continuation_seed"] is not None,
                **{
                    k: sample[k]
                    for k in [
                        "malformed_statements",
                        "duplicate_contacts",
                        "invalid_positions",
                        "near_diagonal",
                        "retractions",
                        "orphan_retractions",
                        "unexpected_text",
                    ]
                },
            }
        )
    for region in ["all", "long"]:

        def long_filter(x: int) -> bool:
            return x % length - x // length >= 24

        ranged = (
            sets
            if region == "all"
            else [{x for x in s if long_filter(x)} for s in sets]
        )
        selected = (
            np.ones(len(ids), bool)
            if region == "all"
            else ids % length - ids // length >= 24
        )
        ranged_ids, ranged_labels = ids[selected], labels[selected]
        true_range = true if region == "all" else {x for x in true if long_filter(x)}
        stats = diversity(ranged, length * length)
        stats["entropy_per_candidate"] = (
            stats["entropy_sum"] / len(ranged_ids) if len(ranged_ids) else float("nan")
        )
        diversity_rows.append({**base, "range": region, "block": "pooled", **stats})
        if record["mode"] == "fixed":
            for block in range(3):
                stats_b = diversity(
                    ranged[block * 100 : (block + 1) * 100], length * length
                )
                block_votes = np.zeros(length * length, np.int32)
                for sample in ranged[block * 100 : (block + 1) * 100]:
                    if sample:
                        block_votes[list(sample)] += 1
                stats_b["r_precision"] = r_precision(
                    block_votes, ranged_ids, ranged_labels
                )
                stats_b["entropy_per_candidate"] = (
                    stats_b["entropy_sum"] / len(ranged_ids)
                    if len(ranged_ids)
                    else float("nan")
                )
                diversity_rows.append(
                    {**base, "range": region, "block": str(block), **stats_b}
                )
        votes = np.zeros(length * length, np.int32)
        discovered: set[int] = set()
        for k, s in enumerate(ranged, 1):
            if s:
                votes[list(s)] += 1
            discovered |= s
            if k in [1, 10, 25, 50, 100, 200, 300]:
                curves.append(
                    {
                        **base,
                        "range": region,
                        "k": k,
                        "r_precision": r_precision(votes, ranged_ids, ranged_labels),
                        "true_discovered": len(discovered & true_range),
                        "false_discovered": len(discovered - true_range),
                        "true_coverage": len(discovered & true_range) / len(true_range)
                        if true_range
                        else float("nan"),
                        "discovery_precision": len(discovered & true_range)
                        / len(discovered)
                        if discovered
                        else 0.0,
                        "n_true": len(true_range),
                        "n_candidate": len(ranged_ids),
                    }
                )
    votes = np.zeros(length * length, np.float64)
    for s in sets:
        if s:
            votes[list(s)] += 1
    score = votes.reshape(length, length)
    score += score.T
    pi, pj, psep = METRICS.resolved_pairs(np.array(truth["resolved"]))
    final = METRICS.metric_rows(
        score,
        METRICS.true_matrix(length, truth["contacts"]),
        pi,
        pj,
        psep,
        length,
        with_precision=True,
    )
    metric_rows.extend({**base, **m} for m in final)
    canonical_r = next(
        m["precision"] for m in final if m["range"] == "all" and m["cut"] == "R"
    )
    curve_r = next(
        r["r_precision"] for r in curves if r["range"] == "all" and r["k"] == len(sets)
    )
    assert np.isclose(canonical_r, curve_r, equal_nan=True)
    return diversity_rows, curves, diagnostics, metric_rows


def main() -> None:
    """Require complete paired inputs before writing final analysis tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=HERE / "scratch/raw")
    parser.add_argument("--shuffles", type=int, default=10)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    plan = json.loads((HERE / "data/plan.json").read_text())
    truth = {
        r["stem"]: r
        for r in json.loads((HERE / "inputs/ground_truth.json").read_text())
    }
    targets = pd.read_csv(HERE / "data/targets.csv")
    rows, curves, diagnostics, metrics, controls, shuffles, timings = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    missing = []
    for stem in targets.stem:
        gt = truth[stem]
        for temperature in [1.0, 0.7, 1.3]:
            for mode in ["fixed", "randomized"]:
                key = f"{stem}-t{temperature:.1f}-{mode}"
                paths = {
                    model: args.raw / model / (key + ".json.gz")
                    for model in ["default", "mpnn"]
                }
                if not all(path.exists() for path in paths.values()):
                    missing.append(key)
                    continue
                records = {model: load(path) for model, path in paths.items()}
                left, right = records["default"], records["mpnn"]
                assert left["prompts"] == right["prompts"]
                expected = 300 if mode == "fixed" else 100
                assert len(left["samples"]) == len(right["samples"]) == expected
                assert [s["seed"] for s in left["samples"]] == [
                    s["seed"] for s in right["samples"]
                ]
                assert [s["prompt_token_ids"] for s in left["samples"]] == [
                    s["prompt_token_ids"] for s in right["samples"]
                ]
                paired_sets = {}
                for model, record in records.items():
                    assert (
                        record["model"] == model
                        and record["mode"] == mode
                        and record["temperature"] == temperature
                    )
                    assert (
                        record["target"]["stem"] == stem
                        and record["target"]["L"] == gt["L"]
                    )
                    sets, true, ids, labels = candidate_sets(record, gt)
                    paired_sets[model] = sets
                    a, b, c, d = analyze_one(record, gt, sets, true, ids, labels)
                    rows.extend(a)
                    curves.extend(b)
                    diagnostics.extend(c)
                    metrics.extend(d)
                    marker_path = args.raw / model / (key + ".complete.json")
                    marker = json.loads(marker_path.read_text())
                    assert marker["count"] == expected
                    assert marker["unfinished"] == sum(
                        s["finish_reason"] != "stop" for s in record["samples"]
                    )
                    timings.append(
                        {
                            **marker["timing"],
                            "model": model,
                            "eval_set": record["target"]["eval_set"],
                        }
                    )
                    if temperature == 1.0 and mode == "fixed" and args.shuffles:
                        actual = diversity(sets, gt["L"] ** 2)
                        for repeat in range(args.shuffles):
                            shuffled = curveball(sets, 373 + repeat, 10 * len(sets))
                            null = diversity(shuffled, gt["L"] ** 2)
                            assert [len(s) for s in sets] == [len(s) for s in shuffled]
                            assert Counter(x for s in sets for x in s) == Counter(
                                x for s in shuffled for x in s
                            )
                            # A second independently seeded half-length chain diagnoses sensitivity to mixing duration.
                            half = diversity(
                                curveball(sets, 1373 + repeat, 5 * len(sets)),
                                gt["L"] ** 2,
                            )
                            shuffles.append(
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
                for region in ["all", "long"]:
                    pair = [
                        paired_sets[m]
                        if region == "all"
                        else [
                            {x for x in s if x % gt["L"] - x // gt["L"] >= 24}
                            for s in paired_sets[m]
                        ]
                        for m in ["default", "mpnn"]
                    ]
                    a, b = thin_matched(*pair, 373)
                    for model, sets in [("default", a), ("mpnn", b)]:
                        controls.append(
                            {
                                "stem": stem,
                                "model": model,
                                "temperature": temperature,
                                "mode": mode,
                                "range": region,
                                "eval_set": left["target"]["eval_set"],
                                **diversity(sets, gt["L"] ** 2),
                            }
                        )
        print("ANALYZED", stem, "paired_conditions", len(timings) // 2, flush=True)
    if missing and not args.allow_partial:
        raise ValueError(f"Missing {len(missing)} paired conditions: {missing[:10]}")
    destination = HERE / ("scratch/partial" if args.allow_partial else "data")
    destination.mkdir(exist_ok=True, parents=True)
    for name, table in [
        ("diversity", rows),
        ("ensemble_curves", curves),
        ("rollout_diagnostics", diagnostics),
        ("metrics", metrics),
        ("count_controls", controls),
        ("shuffle_controls", shuffles),
        ("timings", timings),
    ]:
        frame = pd.DataFrame(table)
        frame.to_csv(
            destination
            / (name + (".csv.gz" if name == "rollout_diagnostics" else ".csv")),
            index=False,
        )
    (destination / "completeness.json").write_text(
        json.dumps(
            {
                "paired_conditions": len(timings) // 2,
                "sample_count": len(diagnostics),
                "expected_samples": 278400,
                "missing": missing,
                "shuffle_replicates": args.shuffles,
                "plan": plan,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
