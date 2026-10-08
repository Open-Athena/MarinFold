"""Measure oracle best-of-k interface contacts from saved individual rollouts.

Individual maps are unranked sets. F1 balances precision and recall; individual
R-precision averages over random tie-breaking within emitted and un-emitted
pairs. Oracle selection uses experimental truth and is a sampling diagnostic,
not an inference-time selection algorithm. Expected best-of-k is exact for a
uniform k-subset of the saved pool, removing arbitrary sample-order effects.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.special import gammaln

from score_foldbench_contacts import target_r_precision

HERE = Path(__file__).resolve().parent
BUDGETS = (1, 10, 100, 1000)
SEED = 350


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a nonempty result table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def expected_r_precision(n_correct: int, n_predicted: int, n_true: int, universe: int) -> float:
    """Return top-R precision for a binary map with uniformly broken score ties."""
    if n_predicted >= n_true:
        return n_correct / n_predicted
    remaining = n_true - n_predicted
    expected_padding_correct = remaining * (n_true - n_correct) / (universe - n_predicted)
    return (n_correct + expected_padding_correct) / n_true


def maximum_weights(n: int, k: int) -> np.ndarray:
    """Return the probabilities of each sorted observation being a subset maximum."""
    if not 1 <= k <= n:
        raise ValueError(f'Invalid subset size: {k} of {n}')
    ranks = np.arange(k, n + 1, dtype=np.float64)
    log_cdf = (
        gammaln(ranks + 1) - gammaln(ranks - k + 1)
        - gammaln(n + 1) + gammaln(n - k + 1)
    )
    cdf = np.exp(log_cdf)
    cdf[-1] = 1.0
    return np.concatenate([np.zeros(k - 1), np.diff(np.concatenate([[0.0], cdf]))])


def expected_best(values: np.ndarray, k: int) -> float:
    """Return the exact expected maximum in a uniform k-subset without replacement."""
    return float(np.dot(np.sort(values), maximum_weights(len(values), k)))


def random_union_recall(n_predicted: np.ndarray, universe: int) -> np.ndarray:
    """Return exact matched-random union recall for every uniform subset size.

    Each map includes a fixed true pair with probability its size / universe.
    Average products of omission probabilities using normalized elementary
    symmetric polynomials, avoiding combinatorial overflow at k=1000.
    """
    omission = np.zeros(len(n_predicted) + 1)
    omission[0] = 1.0
    for m, count in enumerate(n_predicted, start=1):
        fraction = np.arange(1, m + 1) / m
        omission[1:m + 1] = ((1 - fraction) * omission[1:m + 1]
                            + fraction * (1 - count / universe) * omission[:m])
    return 1 - omission


def group_interval(rows: list[dict], metric: str) -> tuple[float, float]:
    """Bootstrap frozen homology groups, retaining their member targets."""
    groups = sorted({row['group_id'] for row in rows})
    values = [np.array([row[metric] for row in rows if row['group_id'] == g]) for g in groups]
    sums = np.array([x.sum() for x in values])
    sizes = np.array([len(x) for x in values])
    draws = np.random.default_rng(SEED).integers(0, len(groups), size=(10000, len(groups)))
    means = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return tuple(float(v) for v in np.quantile(means, [0.025, 0.975]))


def score_target(target: dict, samples: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Score individual maps and all sampling budgets on one frozen target."""
    if [r['rollout'] for r in samples] != list(range(1000)):
        raise ValueError(f"{target['stem']}: expected exactly rollout indices 0..999")
    left, right = map(set, target['resolved_positions_by_chain'])
    universe = len(left) * len(right)
    truth = {tuple(pair) for pair in target['gt_contacts']}
    n_true = len(truth)
    per_sample = []
    maps = []
    for row in samples:
        complete = row['finish_reason'] == 'stop'
        contacts = {(int(i), int(j)) for i, j in row['contacts'] if i in left and j in right}
        # Keep failed attempts in the sample budget. They cannot win selection.
        contacts = contacts if complete else set()
        maps.append(contacts)
        tp = len(contacts & truth)
        p = len(contacts)
        per_sample.append({
            'target_id': target['target_id'], 'split': target['split'],
            'group_id': target['group_id'], 'rollout': row['rollout'],
            'complete': complete, 'n_true': n_true, 'n_predicted': p, 'n_correct': tp,
            'precision': tp / p if p else 0.0, 'recall': tp / n_true,
            'f1': 2 * tp / (n_true + p),
            'r_precision_tie_expected': expected_r_precision(tp, p, n_true, universe) if complete else 0.0,
        })
    f1 = np.array([r['f1'] for r in per_sample])
    rp = np.array([r['r_precision_tie_expected'] for r in per_sample])
    recall = np.array([r['recall'] for r in per_sample])
    n_pred = np.array([r['n_predicted'] for r in per_sample])
    complete = np.array([r['complete'] for r in per_sample])
    rng = np.random.default_rng(SEED)
    # Conditional null preserves every sample's contact count and completion.
    # Its interfaces are sampled uniformly without replacement from the same
    # resolved cross-chain universe; 1000 trials estimate the order statistics.
    random_tp = rng.hypergeometric(n_true, universe - n_true, n_pred, size=(1000, 1000))
    random_f1 = 2 * random_tp / (n_true + n_pred)
    random_f1[:, ~complete] = 0.0
    random_f1.sort(axis=1)
    random_union = random_union_recall(n_pred, universe)
    coverage = Counter(pair for pairs in maps for pair in pairs)
    true_frequencies = np.array([coverage[pair] for pair in sorted(truth)])
    output = []
    for k in BUDGETS:
        weights = maximum_weights(len(samples), k)
        log_miss = np.full(n_true, -np.inf)
        missing_counts = len(samples) - true_frequencies
        valid = missing_counts >= k
        log_miss[valid] = (
            gammaln(missing_counts[valid] + 1) - gammaln(missing_counts[valid] - k + 1)
            - gammaln(len(samples) + 1) + gammaln(len(samples) - k + 1)
        )
        prefix_votes = Counter(pair for pairs in maps[:k] for pair in pairs)
        best_idx = int(np.argmax(f1[:k]))
        best = per_sample[best_idx]
        output.append({
            'target_id': target['target_id'], 'split': target['split'],
            'group_id': target['group_id'], 'complex_type': target['complex_type'],
            'k': k, 'n_true': n_true, 'n_complete_pool': int(complete.sum()),
            'expected_best_f1': expected_best(f1, k),
            'expected_best_r_precision': expected_best(rp, k),
            'expected_best_recall': expected_best(recall, k),
            'expected_union_recall': float(np.mean(1 - np.exp(log_miss))),
            'random_expected_best_f1': float(np.mean(random_f1 @ weights)),
            'random_expected_union_recall': float(random_union[k]),
            'prefix_best_f1': best['f1'], 'prefix_best_rollout': best_idx,
            'prefix_best_precision': best['precision'], 'prefix_best_recall': best['recall'],
            'prefix_consensus_r_precision': target_r_precision(target, prefix_votes)['r_precision'],
            'n_pool_samples_f1_ge_0_25': int(np.sum(f1 >= 0.25)),
            'n_pool_samples_f1_ge_0_5': int(np.sum(f1 >= 0.5)),
            'n_pool_samples_precision_and_recall_ge_0_5': sum(
                r['precision'] >= 0.5 and r['recall'] >= 0.5 for r in per_sample
            ),
        })
    best = per_sample[int(np.argmax(f1))]
    selected = {'target_id': target['target_id'], 'split': target['split'], **best,
                'contacts': sorted(maps[best['rollout']])}
    return per_sample, output, selected


def main() -> None:
    """Validate a 1000-rollout pool and publish sampling diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, default=HERE / 'data/sampling_v1')
    args = parser.parse_args()
    targets_path = HERE / 'data/foldbench_complex_contact_eval_targets.parquet'
    targets = pq.read_table(targets_path).to_pylist()
    paths = sorted((args.root / 'rollouts').glob('*.parquet'))
    samples_by_target: dict[str, list[dict]] = {}
    for path in paths:
        for row in pq.read_table(path).to_pylist():
            samples_by_target.setdefault(row['stem'], []).append(row)
    if set(samples_by_target) != {t['stem'] for t in targets}:
        raise ValueError('Saved rollout target set does not match the frozen 23 targets')
    aggregate_votes: dict[str, dict[tuple[int, int], int]] = {}
    for path in sorted((args.root / 'scores').glob('*.parquet')):
        for row in pq.read_table(path).to_pylist():
            votes = aggregate_votes.setdefault(row['stem'], {})
            pair = (row['i'], row['j'])
            if pair in votes:
                raise ValueError(f"Duplicate saved aggregate vote: {row['stem']} {pair}")
            votes[pair] = row['votes']
    for stem, samples in samples_by_target.items():
        reconstructed = Counter(tuple(pair) for row in samples
                                if row['finish_reason'] == 'stop' for pair in row['contacts'])
        if reconstructed != aggregate_votes.get(stem, {}):
            raise ValueError(f'{stem}: individual maps do not reconstruct aggregate votes')
    all_samples, curves, selected = [], [], []
    for target in targets:
        samples = sorted(samples_by_target[target['stem']], key=lambda r: r['rollout'])
        sample_rows, target_curves, best = score_target(target, samples)
        all_samples.extend(sample_rows)
        curves.extend(target_curves)
        selected.append(best)
        print(f"{target['target_id']} {target['split']}: best@1000 F1={best['f1']:.3f} "
              f"P={best['precision']:.3f} R={best['recall']:.3f} "
              f"TP={best['n_correct']}/{best['n_true']} predicted={best['n_predicted']}", flush=True)
    metrics = ['expected_best_f1', 'expected_best_r_precision', 'expected_best_recall',
               'expected_union_recall', 'random_expected_best_f1',
               'random_expected_union_recall', 'prefix_consensus_r_precision']
    summaries = []
    for split in ('dev', 'test', 'all'):
        for k in BUDGETS:
            rows = [r for r in curves if r['k'] == k and (split == 'all' or r['split'] == split)]
            result = {'split': split, 'k': k, 'n_targets': len(rows),
                      'n_groups': len({r['group_id'] for r in rows})}
            for metric in metrics:
                result['mean_' + metric] = float(np.mean([r[metric] for r in rows]))
                if metric in ('expected_best_f1', 'expected_best_r_precision'):
                    low, high = group_interval(rows, metric)
                    result[metric + '_95_low'] = low
                    result[metric + '_95_high'] = high
            result['n_prefix_best_f1_ge_0_25'] = sum(r['prefix_best_f1'] >= .25 for r in rows)
            result['n_prefix_best_f1_ge_0_5'] = sum(r['prefix_best_f1'] >= .5 for r in rows)
            summaries.append(result)
    write_csv(args.out_dir / 'per_rollout.csv', all_samples)
    write_csv(args.out_dir / 'per_target.csv', curves)
    write_csv(args.out_dir / 'summary.csv', summaries)
    (args.out_dir / 'oracle_selected.json').write_text(json.dumps(selected, indent=2) + '\n')
    timings = [row for path in sorted((args.root/'timings').glob('*.parquet'))
               for row in pq.read_table(path).to_pylist()]
    if len(timings) != 23 or {r['stem'] for r in timings} != set(samples_by_target):
        raise ValueError('Expected 23 unique per-target timing records')
    write_csv(args.out_dir/'timings.csv', sorted(timings, key=lambda r: r['stem']))
    manifest = {
        'checkpoint': 'contacts-v1-exp343-m2-p06-complex-1.5B-step-280154',
        'targets_sha256': hashlib.sha256(targets_path.read_bytes()).hexdigest(),
        'n_targets': 23, 'n_rollouts': len(all_samples),
        'n_unfinished': sum(not r['complete'] for r in all_samples),
        'budgets': list(BUDGETS),
        'selection': 'oracle experimental inter-chain F1; R-precision maximized separately',
        'expectation': 'exact uniform k-subset of 1000 attempted rollouts without replacement',
        'unfinished': 'retained as zero-score failed attempts; never contribute contacts',
        'random_null': '1000 Monte Carlo pools, matched per-sample predicted contact counts',
        'seed': SEED,
        'sampling': {'temperature': 1.0, 'top_p': 0.95, 'top_k': -1,
                     'seed': 0, 'chunk': 1, 'context': 8192},
        'iris_jobs': [f'/bizon/exp350-foldbench-complex-s{i}of12-best1000-v1' for i in range(12)],
        'source_s3': 's3://marin-us-east-02a/MarinFold/exp350_evals_complex_holdout_survival/foldbench-pair-holdout-v1/rollout-best1000-v1/contacts-v1-exp343-m2-p06-complex-1.5B-step-280154',
        'code_sha256': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                        for name in ('score_complex_rollout_worker.py', 'dispatch_foldbench_contacts_cw.py',
                                     'score_rollout_sampling.py')},
        'validation': 'all saved individual contact maps exactly reconstruct the aggregate votes',
        'public_prefix': 'hf://buckets/open-athena/MarinFold/data/evals/exp350_foldbench_pair_holdout/contact_eval_v1/sampling_v1',
        'raw_files': {str(p.relative_to(args.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(args.root.rglob('*')) if p.is_file()},
        'derived': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(args.out_dir.iterdir()) if p.is_file() and p.name != 'manifest.json'},
    }
    (args.out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for row in summaries:
        if row['split'] == 'test':
            print(row)


if __name__ == '__main__':
    main()
