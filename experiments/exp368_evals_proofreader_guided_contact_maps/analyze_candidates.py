"""Measure conditional proposal ranking from committed guided trajectories."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from metrics import bootstrap


def main() -> None:
    """Compare selected contacts with the sampled alternatives at each prefix."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--guided', type=Path, required=True)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    targets = {r['entry_id']: r for r in pq.read_table(args.targets,
        filters=[('entry_id', 'in', plan['entries'])]).to_pylist()}
    records = []
    for entry in plan['entries']:
        target = targets[entry]
        if target['split'] != plan['split']:
            raise ValueError('Wrong split in diagnostic panel')
        truth = set(map(tuple, target['contacts']))
        for policy in plan['policies']:
            for sample in range(plan['rollouts']):
                key = args.guided / entry / f'{policy["name"]}-r{sample}'
                if not Path(str(key) + '.done.json').exists():
                    continue
                if not Path(str(key) + '.candidates.done.json').exists():
                    continue
                frame = pq.read_table(str(key) + '.candidates.parquet').to_pandas()
                for prefix, rows in frame.groupby('prefix_contacts'):
                    labels = rows['label'].to_numpy()
                    actual = np.array([int(tuple(p) in truth) for p in rows['pair']])
                    if not np.array_equal(labels, actual) or rows.selected.sum() != 1:
                        raise ValueError('Candidate labels or selected count are inconsistent')
                    selected = rows[rows.selected].iloc[0]
                    auc = roc_auc_score(labels, rows.selection_score) if len(set(labels)) == 2 else np.nan
                    records.append(dict(entry_id=entry, group_id=target['group_id'], L=target['L'],
                        policy=policy['name'], sample=sample, prefix_contacts=prefix,
                        candidate_count=len(rows), candidate_unique=len({tuple(p) for p in rows['pair']}),
                        proposal_accuracy=labels.mean(), selected_accuracy=selected['label'],
                        conditional_lift=selected['label'] - labels.mean(), within_prefix_auc=auc,
                        selected_q=selected.immediate_probability, mean_q=rows.immediate_probability.mean(),
                        proposal_brier=np.mean((rows.immediate_probability.to_numpy() - labels) ** 2),
                        selected_brier=(selected.immediate_probability - selected['label']) ** 2))
    args.out.mkdir(parents=True, exist_ok=True)
    if not records:
        print('No committed candidate diagnostics yet')
        return
    frame = pd.DataFrame(records)
    frame.to_csv(args.out / 'candidate_steps.csv.gz', index=False)
    metrics = ['proposal_accuracy', 'selected_accuracy', 'conditional_lift', 'within_prefix_auc',
               'selected_q', 'mean_q', 'proposal_brier', 'selected_brier', 'candidate_unique']
    per_protein = frame.groupby(['entry_id', 'group_id', 'policy'])[metrics].mean().reset_index()
    per_protein.to_csv(args.out / 'candidate_per_protein.csv.gz', index=False)
    summary = []
    for policy, rows in per_protein.groupby('policy'):
        record = dict(policy=policy, structures=len(rows), groups=rows.group_id.nunique(),
                      sampled_steps=int((frame.policy == policy).sum()))
        for metric in metrics:
            record[metric] = rows[metric].mean()
            record[metric + '_low'], record[metric + '_high'] = bootstrap(rows[metric].tolist(), rows.group_id.tolist())
        summary.append(record)
    pd.DataFrame(summary).to_csv(args.out / 'candidate_summary.csv', index=False)
    print(pd.DataFrame(summary)[['policy', 'structures', 'sampled_steps', 'conditional_lift',
                                'within_prefix_auc', 'selected_q', 'selected_accuracy']].to_string(index=False))


if __name__ == '__main__':
    main()
