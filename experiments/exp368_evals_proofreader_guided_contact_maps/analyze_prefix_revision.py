"""Measure how forced continuation revises judgments of identical early contacts.

This is a retrospective diagnostic of the complete fixed-length validation
experiment, not a new policy or a subset chosen by accuracy. Early contact
identities and labels must match exactly between the L and 2L prefix records.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from build_summary import save_plot_with_meta
from metrics import bootstrap


def main() -> None:
    """Save structure-weighted calibration and paired early-contact score changes."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plot', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan['split'] != 'validation' or plan['contact_budgets_L'] != [1, 2]:
        raise ValueError('This diagnostic is restricted to the frozen L/2L validation experiment')
    records = []
    for entry in plan['entries']:
        for sampler in plan['samplers']:
            policy = sampler['name']
            prefixes = []
            for budget in [1, 2]:
                key = args.root / entry / f'{policy}-fixed{budget}L'
                if not key.with_suffix('.done.json').exists():
                    raise ValueError('Incomplete fixed-length cohort: ' + str(key))
                rows = sorted(pq.read_table(key.with_suffix('.parquet')).to_pylist(), key=lambda r: r['sample'])
                if [r['sample'] for r in rows] != list(range(plan['rollouts'])):
                    raise ValueError('Missing or duplicate trajectory identities')
                prefixes.append(rows)
            for early, full in zip(*prefixes, strict=True):
                length = early['L']
                if (full['L'] != length or len(early['pairs']) != length or len(full['pairs']) != 2 * length
                        or early['pairs'] != full['pairs'][:length]
                        or early['labels'] != full['labels'][:length]
                        or early['group_id'] != full['group_id']):
                    raise ValueError('L/2L records do not share the identical nested prefix')
                identity = dict(entry_id=entry, group_id=early['group_id'], L=length,
                                policy=policy, sample=early['sample'])
                for stage, row, selection in [('early_at_L', early, slice(None)),
                                              ('early_at_2L', full, slice(0, length)),
                                              ('late_at_2L', full, slice(length, None))]:
                    q = np.array(row['probabilities'])[selection]
                    y = np.array(row['labels'])[selection]
                    records.append(dict(**identity, stage=stage, precision=y.mean(), predicted_precision=q.mean(),
                        brier=np.mean((q - y) ** 2),
                        auroc=roc_auc_score(y, q) if len(set(y)) == 2 else np.nan,
                        true_contact_q=q[y == 1].mean() if (y == 1).any() else np.nan,
                        false_contact_q=q[y == 0].mean() if (y == 0).any() else np.nan))
    args.out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(args.out / 'per_trajectory.csv.gz', index=False)
    metrics = ['precision', 'predicted_precision', 'brier', 'auroc', 'true_contact_q', 'false_contact_q']
    protein = frame.groupby(['entry_id', 'group_id', 'policy', 'stage'], as_index=False)[metrics].mean()
    protein.to_csv(args.out / 'per_protein.csv.gz', index=False)
    summary, paired = [], []
    for (policy, stage), rows in protein.groupby(['policy', 'stage']):
        for metric in metrics:
            low, high = bootstrap(rows[metric].tolist(), rows.group_id.tolist())
            summary.append(dict(policy=policy, stage=stage, metric=metric, mean=rows[metric].mean(),
                                low=low, high=high, structures=rows[metric].notna().sum()))
    for policy, rows in protein.groupby('policy'):
        early = rows[rows.stage == 'early_at_L'].set_index('entry_id').sort_index()
        revised = rows[rows.stage == 'early_at_2L'].set_index('entry_id').sort_index()
        for metric in metrics:
            delta = revised[metric] - early[metric]
            low, high = bootstrap(delta.tolist(), early.group_id.tolist())
            paired.append(dict(policy=policy, metric=metric, difference=delta.mean(), low=low, high=high))
    summary = pd.DataFrame(summary)
    summary.to_csv(args.out / 'summary.csv', index=False)
    pd.DataFrame(paired).to_csv(args.out / 'paired_early_revision.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    stages = ['early_at_L', 'early_at_2L', 'late_at_2L']
    labels = ['First L contacts\nscored at L', 'Same first L contacts\nrescored at 2L', 'New second L contacts\nscored at 2L']
    for axis, sampler in zip(axes, plan['samplers'], strict=True):
        policy = sampler['name']
        for metric, offset, color, label in [('precision', -.15, '#2563eb', 'Actual precision'),
                                           ('predicted_precision', .15, '#d97706', 'Predicted precision')]:
            rows = summary[(summary.policy == policy) & (summary.metric == metric)].set_index('stage').loc[stages]
            axis.bar(np.arange(3) + offset, rows['mean'], width=.3, color=color, label=label,
                     yerr=np.stack([rows['mean'] - rows.low, rows.high - rows['mean']]), capsize=3)
        axis.set(title=policy, ylim=(0, 1), xticks=np.arange(3), xticklabels=labels)
        axis.tick_params(axis='x', labelsize=8)
        axis.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Precision / mean predicted correctness')
    axes[0].legend(fontsize=8)
    fig.suptitle('Forced continuation revises judgments of identical early contacts')
    fig.tight_layout()
    save_plot_with_meta(fig, args.plot, dpi=180, caption=(
        'Retrospective analysis of all 32 validation proteins and four trajectories per sampler. '
        'First-L pairs and labels are verified identical before and after continuation. '
        'Means average trajectories within each protein, then proteins; 95% intervals resample homology groups. '
        'The second half is generated with EOS disabled. No test-set data or new model inference is used.'))
    plt.close(fig)
    print(summary[summary.metric.isin(['precision', 'predicted_precision', 'brier'])].to_string(index=False))
    print(pd.DataFrame(paired).to_string(index=False))


if __name__ == '__main__':
    main()
