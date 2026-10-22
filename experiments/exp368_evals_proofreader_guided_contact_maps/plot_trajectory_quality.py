"""Compare actual rollout quality, critic predictions, and emitted contact counts."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta
from metrics import bootstrap


def main() -> None:
    """Plot complete, equally weighted protein cohorts without selecting winners."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, nargs='+', required=True)
    parser.add_argument('--plan', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--table', type=Path, required=True)
    parser.add_argument('--title', default='Rollout precision, recall and length')
    args = parser.parse_args()
    if len(args.data) != len(args.plan):
        raise ValueError('Each analysis directory requires its frozen plan')
    panels, order = [], []
    cohort = None
    for directory, plan_path in zip(args.data, args.plan, strict=True):
        plan = json.loads(plan_path.read_text())
        entries = set(plan['entries'])
        if cohort is not None and entries != cohort:
            raise ValueError('Plans differ in protein cohort')
        cohort = entries
        quality = pd.read_csv(directory / 'trajectory_quality.csv.gz')
        maps = pd.read_csv(directory / 'per_protein.csv.gz')
        lengths = maps[['entry_id', 'L']].drop_duplicates()
        if lengths.entry_id.duplicated().any():
            raise ValueError('Conflicting protein lengths')
        quality = quality.merge(lengths, on='entry_id', validate='many_to_one')
        quality['contacts_per_L'] = quality.contacts / quality.L
        for policy in plan['policies']:
            name = policy['name']
            if name in order:
                raise ValueError('Duplicate policy across panels: ' + name)
            rows = quality[quality.policy == name]
            if (set(rows.entry_id) != entries or rows.duplicated(['entry_id', 'sample']).any()
                    or not rows.groupby('entry_id')['sample'].apply(set).apply(
                        lambda samples: samples == set(range(plan['rollouts']))).all()):
                raise ValueError('Policy has incomplete trajectories: ' + name)
            order.append(name)
            panels.append(rows.groupby(['entry_id', 'group_id', 'policy'], as_index=False)[
                ['precision', 'recall', 'predicted_precision', 'predicted_recall', 'contacts_per_L']].mean())
    frame = pd.concat(panels, ignore_index=True)
    records = []
    for name in order:
        rows = frame[frame.policy == name]
        for metric in ['precision', 'recall', 'predicted_precision', 'predicted_recall', 'contacts_per_L']:
            low, high = bootstrap(rows[metric].tolist(), rows.group_id.tolist())
            records.append(dict(policy=name, metric=metric, mean=rows[metric].mean(),
                                low=low, high=high, structures=len(rows), groups=rows.group_id.nunique()))
    summary = pd.DataFrame(records)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.table, index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, max(5, len(order) * .32)), sharey=True)
    y = np.arange(len(order))
    for axis, metric, title in zip(axes, ['precision', 'recall', 'contacts_per_L'],
                                    ['Per-rollout precision', 'Per-rollout recall', 'Contacts / sequence length'], strict=True):
        rows = summary[summary.metric == metric].set_index('policy').loc[order]
        axis.errorbar(rows['mean'], y, xerr=np.stack([rows['mean'] - rows.low, rows.high - rows['mean']]),
                      fmt='o', color='#2166ac', markersize=4, capsize=2, label='Actual, 95% group bootstrap')
        if metric != 'contacts_per_L':
            predicted = summary[summary.metric == 'predicted_' + metric].set_index('policy').loc[order]
            axis.scatter(predicted['mean'], y, marker='x', color='#b35806', label='Proofreader prediction')
            axis.set_xlim(0, 1)
        axis.set_title(title)
        axis.grid(axis='x', alpha=.2)
    axes[0].set_yticks(y, order)
    axes[0].invert_yaxis()
    axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1.04), fontsize=8)
    fig.suptitle(args.title, y=1.01)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_plot_with_meta(fig, args.out, caption=(
        f'{len(cohort)} proteins, equally weighted after averaging trajectories within each protein. '
        'Blue intervals resample homology groups. Orange crosses show mean critic predictions.'
        + (' Fixed-length L and 2L maps are nested prefixes of the same continuations.'
           if any('-fixed' in name for name in order) else '')))
    plt.close(fig)


if __name__ == '__main__':
    main()
