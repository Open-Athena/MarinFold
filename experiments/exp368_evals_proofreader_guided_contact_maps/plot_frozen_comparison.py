"""Plot a complete, frozen two-configuration comparison without extra analyses."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta

METRIC_LABELS = {
    'r_precision': 'R-precision',
    'long_r_precision': 'Long-range R-precision',
    'average_precision': 'Average precision',
    'long_average_precision': 'Long-range AP',
    'precision_L': 'Precision at L',
    'precision_L2': 'Precision at L/2',
    'precision_L5': 'Precision at L/5',
}


def main() -> None:
    """Validate the frozen cohort and plot only its recorded paired effects."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--labels', nargs=2, required=True, help='Baseline then candidate display names')
    parser.add_argument('--title', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    configs = plan['evaluation_configs']
    if len(configs) != 2 or len(set(configs)) != 2:
        raise ValueError('Exactly two distinct frozen configurations are required')
    complete = pd.read_csv(args.data / 'completeness.csv')
    if complete.empty or not complete.complete.all():
        raise ValueError('Comparison contains incomplete inputs')
    frame = pd.read_csv(args.data / 'per_protein.csv.gz')
    frame['configuration'] = frame.policy + ':' + frame.rollouts.astype(str) + ':' + frame.method
    if set(frame.configuration) != set(configs):
        raise ValueError('Results differ from the frozen comparison')
    panels = []
    for config in configs:
        rows = frame[frame.configuration == config].set_index('entry_id').sort_index()
        if rows.index.duplicated().any() or set(rows.index) != set(plan['entries']):
            raise ValueError('Comparison does not cover the complete frozen cohort')
        panels.append(rows)
    baseline, candidate = panels
    if not baseline.group_id.equals(candidate.group_id):
        raise ValueError('Related-group assignments differ between configurations')
    paired = pd.read_csv(args.paired)
    identities = paired.policy + ':' + paired.rollouts.astype(str) + ':' + paired.method
    if set(identities) != {configs[1]} or set(paired.baseline) != {configs[0]}:
        raise ValueError('Paired effects differ from the frozen comparison')
    effects = paired.set_index('metric').loc[list(METRIC_LABELS)]
    if effects.index.duplicated().any():
        raise ValueError('Repeated paired metric')
    for metric, row in effects.iterrows():
        delta = candidate[metric] - baseline[metric]
        if not np.isclose(delta.mean(), row.difference, rtol=0., atol=1e-12):
            raise ValueError('Paired effect does not match the per-protein table')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), gridspec_kw={'width_ratios': [1, 1.6]})
    colors = ['#64748b', '#2563eb']
    x = np.arange(2)
    for i, (rows, label, color) in enumerate(zip(panels, args.labels, colors, strict=True)):
        values = rows[['r_precision', 'long_r_precision']].mean().to_numpy()
        bars = axes[0].bar(x + (i - .5) * .34, values, width=.34, color=color, label=label)
        axes[0].bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
    axes[0].set(xticks=x, xticklabels=['All contacts', 'Long-range'], ylabel='Mean R-precision', ylim=(0, 1))
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].grid(axis='y', alpha=.2)
    y = np.arange(len(effects))
    axes[1].errorbar(effects.difference, y,
        xerr=[effects.difference - effects.low, effects.high - effects.difference],
        fmt='o', capsize=3, color=colors[1])
    axes[1].axvline(0, color=colors[0], linewidth=1)
    axes[1].set(yticks=y, yticklabels=list(METRIC_LABELS.values()),
                xlabel='Candidate − baseline (95% paired group bootstrap)')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=.2)
    costs = [rows.elapsed_seconds.mean() for rows in panels]
    hardware = '; '.join(sorted(set(frame.hardware)))
    note = (f'{len(baseline):,} structures / {baseline.group_id.nunique()} related groups · {hardware}\n'
            f'Mean warm GPU seconds: {args.labels[0]} {costs[0]:.3f}; '
            f'{args.labels[1]} {costs[1]:.3f} ({costs[1] / costs[0]:.3f}×)')
    fig.suptitle(args.title)
    fig.text(.5, .02, note, ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .95))
    save_plot_with_meta(fig, args.out, dpi=180, caption=(
        'Only the two configurations in the frozen comparison plan are shown. '
        'Structures are weighted equally; paired 95% intervals resample related groups. '
        'R-precision is primary; the other metrics are secondary and intervals are not multiplicity-adjusted. '
        'Long-range metrics require sequence separation at least 24 and exclude targets without such contacts. '
        'GPU-stage times exclude model loading, staging, CPU aggregation and failed/preempted work. '
        'The reserved set was previously used for the proofreader release; generator pretraining '
        'decontamination is not established.' if plan['split'] == 'test' else
        'Validation-only plotting check. Paired intervals resample related groups; no reserved-test outcomes.'))
    plt.close(fig)


if __name__ == '__main__':
    main()
