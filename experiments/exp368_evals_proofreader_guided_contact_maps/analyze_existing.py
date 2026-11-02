"""Measure ranking/coverage headroom on the existing validation-only rollouts."""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from metrics import aggregate, bootstrap, map_metrics


def main() -> None:
    """Save reproducible structure-level scores and group-bootstrap summaries."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'data')
    args = parser.parse_args()
    targets = {row['entry_id']: row for row in pq.read_table(args.targets).to_pylist()}
    by_entry = defaultdict(list)
    for path in sorted(args.scores.glob('contact_scores-rank-*.parquet')):
        for row in pq.read_table(path).to_pylist():
            if targets[row['entry_id']]['split'] != 'validation':
                raise ValueError('This diagnostic must not read test scores')
            by_entry[row['entry_id']].append(row)
    records = []
    for entry, rows in sorted(by_entry.items()):
        rows = sorted(rows, key=lambda row: int(row['identity'].rsplit(':r', 1)[1]))
        target = targets[entry]
        truth = set(map(tuple, target['contacts']))
        for n in [1, 2, 4, 8]:
            for method in ['frequency', 'sumq', 'meanq', 'sqrt_frequency_meanq', 'diversity_sumq']:
                scores = aggregate(rows[:n], method)
                metrics = map_metrics(scores, truth, target['L'])
                long = map_metrics(scores, truth, target['L'], separation=24)
                records.append(dict(entry_id=entry, group_id=target['group_id'], L=target['L'],
                    requested_rollouts=n, observed_rollouts=min(n, len(rows)), method=method,
                    **metrics, **{'long_' + key: value for key, value in long.items()}))
    frame = pd.DataFrame(records)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / 'existing_validation_per_protein.csv.gz', index=False)
    summary = []
    for (n, method), rows in frame.groupby(['requested_rollouts', 'method']):
        for metric in ['r_precision', 'average_precision', 'oracle_coverage', 'long_r_precision',
                       'long_average_precision', 'long_oracle_coverage']:
            lo, hi = bootstrap(rows[metric].tolist(), rows.group_id.tolist())
            summary.append(dict(rollouts=n, method=method, metric=metric, mean=rows[metric].mean(),
                ci_low=lo, ci_high=hi, structures=len(rows), groups=rows.group_id.nunique()))
    pd.DataFrame(summary).to_csv(args.out / 'existing_validation_summary.csv', index=False)
    print(frame[frame.requested_rollouts == 8].groupby('method')[['r_precision', 'long_r_precision', 'average_precision']].mean().to_string())


if __name__ == '__main__':
    main()
