"""Separate map coverage, trajectory diversity and proofreader calibration.

Diagnostics require the entire named cohort. Contact calibration pools eligible
first occurrences within each rollout; structure-level error metrics give every
structure equal weight. Precision/recall curves use fixed multiples of sequence
length, with absent prediction slots counted as misses, as in the main metrics.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from metrics import aggregate, rank_key


def read_maps(root: Path, entry: str, policy: str, n: int, plan: dict) -> list[dict]:
    """Read only committed complete maps under the frozen pool semantics."""
    if policy not in {'ordinary', 'ordinary-direct'}:
        key = root / entry / policy
        if not key.with_suffix('.done.json').exists():
            raise FileNotFoundError('Incomplete policy: ' + str(key))
        rows = sorted(pq.read_table(key.with_suffix('.parquet')).to_pylist(), key=lambda row: row['sample'])
        rows = rows[:n]
    else:
        independent = plan.get('pool_mode') == 'independent'
        if independent != (policy == 'ordinary-direct'):
            raise ValueError('Pool label conflicts with its frozen plan')
        sizes = [n] if independent else [size for size in plan.get('pool_sizes', [8, 32, 128]) if size <= n]
        rows = []
        for size in sizes:
            key = root / entry / f'pool-{size}'
            if not key.with_suffix('.done.json').exists():
                raise FileNotFoundError('Incomplete pool: ' + str(key))
            rows.extend(pq.read_table(key.with_suffix('.parquet')).to_pylist())
    if len(rows) != n or {row['sample'] for row in rows} != set(range(n)):
        raise ValueError('Missing or duplicate rollout identities')
    return rows


def eligible_observations(row: dict, truth: set[tuple[int, int]]) -> tuple[dict, int]:
    """Verify labels and retain the first probability of each eligible pair."""
    unique = {}
    duplicates = 0
    for pair, q, label in zip(row['pairs'], row['probabilities'], row['labels'], strict=True):
        a, b = pair
        pair = (a, b)
        if bool(label) != (pair in truth):
            raise ValueError('Stored contact label disagrees with target')
        if not 0 <= a < b < row['L'] or b - a < 6:
            continue
        if not np.isfinite(q) or not 0 <= q <= 1:
            raise ValueError('Invalid contact probability')
        duplicates += int(pair in unique)
        unique.setdefault(pair, q)
    return unique, duplicates


def diversity(pair_sets: list[set]) -> dict:
    """Measure independent trajectory overlap using at most 128 fixed pairs."""
    pairs = list(itertools.combinations(range(len(pair_sets)), 2))
    if len(pairs) > 128:
        selected = np.random.default_rng(368).choice(len(pairs), size=128, replace=False)
        pairs = [pairs[index] for index in selected]
    overlap = []
    for i, j in pairs:
        union = pair_sets[i] | pair_sets[j]
        overlap.append(len(pair_sets[i] & pair_sets[j]) / len(union) if union else 1.)
    union = set().union(*pair_sets)
    observations = sum(map(len, pair_sets))
    return dict(pairwise_jaccard=float(np.mean(overlap)) if overlap else np.nan,
                compared_trajectory_pairs=len(pairs), unique_candidates=len(union),
                unique_per_observation=len(union) / observations if observations else 0.)


def main() -> None:
    """Save complete-cohort calibration, diversity and fixed-budget curves."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--spec', nargs='+', required=True, help='policy:rollouts:aggregation')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    targets = {row['entry_id']: row for row in pq.read_table(args.targets,
        filters=[('entry_id', 'in', plan['entries'])]).to_pylist()}
    if set(targets) != set(plan['entries']) or any(row['split'] != plan['split'] for row in targets.values()):
        raise ValueError('Missing targets or wrong split')
    quality, bins, curves = [], [], []
    for spec in args.spec:
        policy, count, method = spec.split(':')
        n = int(count)
        for entry in plan['entries']:
            target = targets[entry]
            truth = set(map(tuple, target['contacts']))
            rows = read_maps(args.root, entry, policy, n, plan)
            identity = dict(entry_id=entry, group_id=target['group_id'], L=target['L'],
                            policy=policy, rollouts=n, method=method)
            observations = [eligible_observations(row, truth) for row in rows]
            unique = [item[0] for item in observations]
            pairs = [set(item) for item in unique]
            probability = np.array([q for item in unique for q in item.values()])
            labels = np.array([float(pair in truth) for item in unique for pair in item])
            if not len(truth):
                raise ValueError('Target has no eligible native contacts')
            precision = np.array([len(item & truth) / len(item) if item else 0. for item in pairs])
            recall = np.array([len(item & truth) / len(truth) for item in pairs])
            predicted_precision = np.array([np.mean(list(item.values())) if item else 0. for item in unique])
            predicted_recall = np.array([row['predicted_recall'] for row in rows])
            bin_ids = np.minimum((probability * 10).astype(int), 9)
            for bin_id in range(10):
                mask = bin_ids == bin_id
                bins.append(dict(**identity, bin=bin_id, count=int(mask.sum()),
                    sum_probability=float(probability[mask].sum()), positives=float(labels[mask].sum())))
            quality.append(dict(**identity, **diversity(pairs),
                oracle_coverage=len(set().union(*pairs) & truth) / len(truth),
                mean_contacts=float(np.mean(list(map(len, pairs)))),
                trajectory_precision=float(precision.mean()), trajectory_recall=float(recall.mean()),
                predicted_precision=float(predicted_precision.mean()), predicted_recall=float(predicted_recall.mean()),
                precision_mae=float(np.abs(predicted_precision - precision).mean()),
                recall_mae=float(np.abs(predicted_recall - recall).mean()),
                contact_brier=float(np.mean((probability - labels) ** 2)) if len(labels) else np.nan,
                finished_fraction=float(np.mean([row['finished'] for row in rows])),
                eligible_duplicates=sum(item[1] for item in observations),
                malformed=sum(row['malformed'] for row in rows),
                invalid_contacts=sum(row['invalid_contacts'] for row in rows)))
            scores = aggregate(rows, method)
            for separation in [6, 24]:
                native = {pair for pair in truth if pair[1] - pair[0] >= separation}
                ranked = sorted((pair for pair in scores if pair[1] - pair[0] >= separation),
                                key=lambda pair: (-scores[pair], rank_key(pair)))
                cumulative = np.cumsum([pair in native for pair in ranked])
                for multiplier in [.1, .25, .5, 1., 2., 4., 8., 16.]:
                    budget = max(1, int(target['L'] * multiplier))
                    tp = int(cumulative[min(budget, len(cumulative)) - 1]) if len(cumulative) else 0
                    curves.append(dict(**identity, separation=separation, budget_L=multiplier,
                        prediction_budget=budget, native_contacts=len(native),
                        precision=tp / budget if native else np.nan,
                        recall=tp / len(native) if native else np.nan))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, records in [('quality', quality), ('calibration', bins), ('curves', curves)]:
        pd.DataFrame(records).to_csv(args.out / f'{name}.csv.gz', index=False)
    print(f'Complete map configurations analyzed: {len(quality)}')


if __name__ == '__main__':
    main()
