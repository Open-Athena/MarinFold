"""Analyze complete guided trajectories and ordinary pools at measured cost."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from metrics import F1_WEIGHT_POWERS, aggregate, bootstrap, map_metrics

METHODS = ['frequency', 'sumq', 'meanq', 'sqrt_frequency_meanq', 'diversity_sumq',
           'frequency_then_meanq', 'f1_weighted_frequency']
SUPPORTED_METHODS = METHODS + [method for method in F1_WEIGHT_POWERS if method not in METHODS]


def evaluation_methods(plan: dict, policy: str, rollouts: int) -> list[str]:
    """Restrict reserved-test metrics to the two configurations frozen in its plan."""
    if plan['split'] != 'test':
        methods = plan.get('analysis_methods', METHODS)
        if not methods or len(set(methods)) != len(methods) or not set(methods) <= set(SUPPORTED_METHODS):
            raise ValueError('Invalid explicitly planned aggregation methods')
        return methods
    configs = plan.get('evaluation_configs', [])
    if len(configs) != 2 or len(set(configs)) != 2:
        raise ValueError('Reserved-test analysis requires exactly two frozen configurations')
    prefix = f'{policy}:{rollouts}:'
    return [method for method in SUPPORTED_METHODS if prefix + method in configs]


def require_complete_test_inputs(plan: dict, guided: Path | None, pool: Path | None) -> None:
    """Fail before reading labels if any frozen reserved-test input is uncommitted."""
    if plan['split'] != 'test':
        return
    evaluation_methods(plan, '', 0)
    for spec in plan['evaluation_configs']:
        policy, count, method = spec.split(':')
        n = int(count)
        if method not in SUPPORTED_METHODS or n <= 0:
            raise ValueError('Invalid reserved-test configuration: ' + spec)
        ordinary = policy in {'ordinary', 'ordinary-direct'}
        root = pool if ordinary else guided
        if root is None:
            raise ValueError('Missing input root for ' + spec)
        if ordinary:
            if (policy == 'ordinary-direct') != (plan.get('pool_mode') == 'independent'):
                raise ValueError('Reserved-test pool mode conflicts with its configuration')
            if n not in plan['pool_sizes']:
                raise ValueError('Reserved-test pool size absent from plan')
            sizes = [n] if policy == 'ordinary-direct' else [s for s in plan['pool_sizes'] if s <= n]
            names = [f'pool-{size}' for size in sizes]
        else:
            if policy not in {p['name'] for p in plan['policies']} or n not in [1, 2, 4, 8, 16] or n > plan['rollouts']:
                raise ValueError('Reserved-test guided configuration absent from plan')
            names = [policy]
        for entry in plan['entries']:
            for name in names:
                if not (root / entry / f'{name}.done.json').exists():
                    raise ValueError(f'Reserved-test cohort is incomplete: {entry}/{name}')


def evaluate(rows: list[dict], target: dict, method: str, seconds: float, policy: str, hardware: str) -> dict:
    """Score one complete map and the candidate pool from which it was ranked."""
    scores = aggregate(rows, method)
    truth = set(map(tuple, target['contacts']))
    result = map_metrics(scores, truth, target['L'])
    long = map_metrics(scores, truth, target['L'], 24)
    return dict(entry_id=target['entry_id'], group_id=target['group_id'], L=target['L'],
        policy=policy, rollouts=len(rows), method=method, elapsed_seconds=seconds, hardware=hardware,
        mean_contacts=np.mean([len(row['pairs']) for row in rows]),
        finished_fraction=np.mean([row['finished'] for row in rows]),
        **result, **{'long_' + key: value for key, value in long.items()})


def main() -> None:
    """Build metrics only from committed per-policy/per-wave output groups."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--guided', type=Path)
    parser.add_argument('--pool', type=Path)
    parser.add_argument('--pool-generation', type=Path)
    parser.add_argument('--frequency-only', action='store_true',
                        help='Evaluate raw ordinary pools without waiting for GPU proofreading')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.frequency_only and (args.guided or not args.pool):
        raise ValueError('Frequency-only analysis requires an ordinary pool and no guided inputs')
    plan = json.loads(args.plan.read_text())
    if plan['split'] == 'test' and args.frequency_only:
        raise ValueError('Reserved-test analysis must execute its complete frozen comparison')
    require_complete_test_inputs(plan, args.guided, args.pool)
    targets = {row['entry_id']: row for row in pq.read_table(args.targets,
        filters=[('entry_id', 'in', plan['entries'])]).to_pylist()}
    if any(targets[entry]['split'] != plan['split'] for entry in plan['entries']):
        raise ValueError('Analysis split differs from the frozen plan')
    results, ledger, calibration, trajectory_quality, completeness = [], [], [], [], []
    for entry in plan['entries']:
        target = targets[entry]
        if args.guided:
            for policy in plan['policies']:
                key = args.guided / entry / policy['name']
                complete = key.with_suffix('.done.json').exists()
                completeness.append(dict(entry_id=entry, policy=policy['name'], complete=complete))
                if not complete:
                    continue
                rows = sorted(pq.read_table(key.with_suffix('.parquet')).to_pylist(), key=lambda r: r['sample'])
                if len(rows) != plan['rollouts']:
                    raise ValueError('Incomplete rollout count despite commit marker')
                timings = [pd.read_csv(str(key) + f'-r{r["sample"]}.timings.csv').iloc[0].to_dict() for r in rows]
                ledger.extend(timings)
                for n in [n for n in [1, 2, 4, 8, 16] if n <= len(rows)]:
                    cost = sum(row['elapsed_seconds'] for row in timings[:n])
                    hardware = ';'.join(sorted({row['gpu_name'] for row in timings[:n]}))
                    for method in evaluation_methods(plan, policy['name'], n):
                        results.append(evaluate(rows[:n], target, method, cost, policy['name'], hardware))
                for row in rows:
                    seen = set()
                    probabilities, labels = [], []
                    for pair, probability, label in zip(row['pairs'], row['probabilities'], row['labels'], strict=True):
                        pair = tuple(pair)
                        if pair in seen:
                            continue
                        seen.add(pair)
                        probabilities.append(probability)
                        labels.append(label)
                    probability = np.array(probabilities)
                    actual = np.array(labels)
                    trajectory_quality.append(dict(entry_id=entry, group_id=target['group_id'],
                        policy=policy['name'], sample=row['sample'], contacts=len(probability),
                        predicted_precision=probability.mean() if len(probability) else 0.,
                        precision=actual.mean() if len(actual) else 0.,
                        recall=actual.sum() / len(target['contacts']), predicted_recall=row['predicted_recall']))
                    for bin_id in range(10):
                        mask = np.minimum((probability * 10).astype(int), 9) == bin_id
                        calibration.append(dict(entry_id=entry, policy=policy['name'], sample=row['sample'],
                            bin=bin_id, count=int(mask.sum()), sum_probability=float(probability[mask].sum()),
                            positives=float(actual[mask].sum())))
        if args.pool:
            rows, generation_seconds, score_seconds = [], 0., 0.
            generation_hardware, score_hardware = set(), set()
            if args.pool_generation is None:
                raise ValueError('Pool comparisons require measured generation costs')
            for size in plan.get('pool_sizes', [8, 32, 128]):
                key = args.pool / entry / f'pool-{size}'
                complete = key.with_suffix('.done.json').exists()
                completeness.append(dict(entry_id=entry, policy=f'ordinary-{size}', complete=complete))
                if not complete:
                    break
                wave = pq.read_table(key.with_suffix('.parquet')).to_pylist()
                if plan.get('pool_mode') == 'independent':
                    rows, generation_seconds, score_seconds = [], 0., 0.
                    generation_hardware, score_hardware = set(), set()
                rows.extend(wave)
                if len(rows) != size or {r['sample'] for r in rows} != set(range(size)):
                    raise ValueError('Ordinary pool contains missing or duplicate samples')
                gt = pd.read_csv(args.pool_generation / entry / f'pool-{size}.timings.csv').iloc[0].to_dict()
                ledger.append(gt)
                generation_seconds += gt['elapsed_seconds']
                generation_hardware.add(gt['gpu_name'])
                if not args.frequency_only:
                    st = pd.read_csv(key.with_suffix('.timings.csv')).iloc[0].to_dict()
                    ledger.append(st)
                    score_seconds += st['elapsed_seconds']
                    score_hardware.add(st['gpu_name'])
                name = 'ordinary-direct' if plan.get('pool_mode') == 'independent' else 'ordinary'
                for method in ['frequency'] if args.frequency_only else evaluation_methods(plan, name, len(rows)):
                    seconds = generation_seconds + (score_seconds if method != 'frequency' else 0.)
                    hardware = ';'.join(sorted(generation_hardware | (score_hardware if method != 'frequency' else set())))
                    results.append(evaluate(rows, target, method, seconds, name, hardware))
    args.out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    pd.DataFrame(completeness).to_csv(args.out / 'completeness.csv', index=False)
    if frame.empty:
        print('No complete policy groups yet')
        return
    if plan['split'] == 'test':
        observed = set(frame.policy + ':' + frame.rollouts.astype(str) + ':' + frame.method)
        if observed != set(plan['evaluation_configs']):
            raise ValueError('Reserved-test output differs from the frozen comparison')
    frame.to_csv(args.out / 'per_protein.csv.gz', index=False)
    pd.DataFrame(ledger).to_csv(args.out / 'timings.csv', index=False)
    pd.DataFrame(calibration).to_csv(args.out / 'calibration.csv.gz', index=False)
    pd.DataFrame(trajectory_quality).to_csv(args.out / 'trajectory_quality.csv.gz', index=False)
    summary = []
    for (policy, n, method), rows in frame.groupby(['policy', 'rollouts', 'method']):
        record = dict(policy=policy, rollouts=n, method=method, structures=len(rows),
            hardware=';'.join(sorted(set(rows.hardware))),
            groups=rows.group_id.nunique(), gpu_seconds=rows.elapsed_seconds.sum(),
            mean_seconds=rows.elapsed_seconds.mean(), long_structures=int(rows.long_r_precision.notna().sum()))
        for metric in ['r_precision', 'average_precision', 'oracle_coverage', 'long_r_precision',
                       'long_average_precision', 'long_oracle_coverage', 'mean_contacts',
                       'precision_L', 'precision_L2', 'precision_L5']:
            record[metric] = rows[metric].mean()
            lo, hi = bootstrap(rows[metric].tolist(), rows.group_id.tolist())
            record[metric + '_low'], record[metric + '_high'] = lo, hi
        summary.append(record)
    summary = pd.DataFrame(summary)
    summary.to_csv(args.out / 'summary.csv', index=False)
    print(summary[summary.method == ('frequency' if args.frequency_only else 'sumq')][['policy', 'rollouts', 'structures', 'r_precision',
        'long_r_precision', 'oracle_coverage', 'mean_seconds']].to_string(index=False))


if __name__ == '__main__':
    main()
