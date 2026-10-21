"""Separate proposal ranking, realized selection gain, and critic confidence."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta
from metrics import bootstrap


def main() -> None:
    """Plot candidate diagnostics only after every planned map is complete."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--completeness', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    complete = pd.read_csv(args.completeness)
    expected = {(entry, policy['name']) for entry in plan['entries'] for policy in plan['policies']}
    observed = set(complete.loc[complete.complete, ['entry_id', 'policy']].itertuples(index=False, name=None))
    if not expected.issubset(observed):
        raise ValueError('Candidate interpretation requires the complete planned map cohort')
    frame = pd.read_csv(args.data / 'candidate_per_protein.csv.gz')
    if frame.duplicated(['entry_id', 'policy']).any() or not set(frame.entry_id).issubset(plan['entries']):
        raise ValueError('Candidate diagnostics have duplicate or unplanned observations')
    order = [policy['name'] for policy in plan['policies'] if policy['name'] in set(frame.policy)]
    if not order:
        raise ValueError('No candidate diagnostics to plot')
    frame['confidence_gap'] = frame.selected_q - frame.selected_accuracy
    records = []
    for name in order:
        rows = frame[frame.policy == name]
        for metric in ['within_prefix_auc', 'conditional_lift', 'confidence_gap']:
            valid = rows[metric].notna()
            low, high = bootstrap(rows[metric].tolist(), rows.group_id.tolist())
            records.append(dict(policy=name, metric=metric, mean=rows[metric].mean(),
                                low=low, high=high, structures=int(valid.sum()),
                                groups=rows.loc[valid, 'group_id'].nunique()))
    summary = pd.DataFrame(records)
    summary.to_csv(args.data / 'candidate_plot_summary.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, max(5, .34 * len(order))), sharey=True)
    y = np.arange(len(order))
    for axis, metric, title, reference in zip(axes,
            ['within_prefix_auc', 'conditional_lift', 'confidence_gap'],
            ['Within-prefix ranking AUC', 'Chosen accuracy − proposal accuracy',
             'Chosen confidence − chosen accuracy'], [.5, 0., 0.], strict=True):
        rows = summary[summary.metric == metric].set_index('policy').loc[order]
        axis.errorbar(rows['mean'], y, xerr=np.stack([rows['mean'] - rows.low, rows.high - rows['mean']]),
                      fmt='o', color='#2166ac', markersize=4, capsize=2)
        axis.axvline(reference, color='#777777', linestyle=':', linewidth=1)
        axis.set_title(title, fontsize=10)
        axis.grid(axis='x', alpha=.2)
    axes[0].set_yticks(y, order)
    axes[0].invert_yaxis()
    fig.suptitle('Does the proofreader choose better next contacts?')
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_plot_with_meta(fig, args.out, dpi=180, caption=(
        'Diagnostics at the first four contact decisions and every eighth decision thereafter. '
        'Sampled decisions are averaged within each protein, then proteins equally; intervals '
        'resample homology groups. AUC uses only candidate pools containing both labels. '
        'Proposal duplicates retain their sampling weight. A positive confidence gap means '
        'overconfidence. Policies without candidate comparisons are omitted. These local '
        'measurements do not by themselves establish an improvement in final contact maps.'))
    plt.close(fig)


if __name__ == '__main__':
    main()
