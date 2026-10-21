"""Compare complete-cohort contact accuracy on one measured GPU model."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from build_summary import save_plot_with_meta


def main() -> None:
    """Plot frozen guided policies and ordinary baselines only when complete."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--title', required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    entries = set(plan['entries'])
    frame = pd.concat([pd.read_csv(path) for path in args.input], ignore_index=True)
    frame = frame[frame.entry_id.isin(entries)]
    keys = ['policy', 'rollouts', 'method']
    if frame.duplicated(keys + ['entry_id']).any():
        raise ValueError('Duplicate map metrics across inputs')
    ordinary = frame.policy.isin(['ordinary', 'ordinary-direct'])
    frame = frame[(~ordinary & (frame.method == 'sumq')) |
                  (ordinary & frame.method.isin(['sumq', 'frequency', 'f1_weighted_frequency']))]
    if frame.hardware.isna().any() or frame.hardware.nunique() != 1 or frame.hardware.str.contains(';').any():
        raise ValueError('A compute frontier must use one GPU model throughout; plot hardware cohorts separately')
    hardware = frame.hardware.iloc[0]
    expected = {policy['name'] for policy in plan['policies']}
    if not expected.issubset(set(frame.policy)):
        raise ValueError('A planned guided policy has no results')
    for key, rows in frame.groupby(keys):
        if set(rows.entry_id) != entries:
            raise ValueError('Incomplete cohort for ' + str(key))
    summary = frame.groupby(keys)[['elapsed_seconds', 'r_precision', 'long_r_precision']].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for index, ((policy, method), rows) in enumerate(summary.groupby(['policy', 'method'])):
        rows = rows.sort_values('rollouts')
        aggregation = {'frequency': 'frequency', 'sumq': 'sum q',
                       'f1_weighted_frequency': 'F1-weighted votes'}[method]
        label = policy + ' / ' + aggregation
        color = plt.get_cmap('tab20')(index)
        marker = 's' if policy.startswith('ordinary') else 'o'
        for axis, metric, title in zip(axes, ['r_precision', 'long_r_precision'],
                                       ['All-contact R-precision', 'Long-range R-precision'], strict=True):
            axis.plot(rows.elapsed_seconds, rows[metric], marker=marker, color=color, label=label,
                      linewidth=1.5, markersize=4)
            axis.set(xscale='log', xlabel=f'Mean warm GPU seconds per contact map\n({hardware})', ylabel=title)
            axis.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=7.5, frameon=False)
    fig.suptitle(args.title)
    fig.tight_layout(rect=(0, .23, 1, .96))
    save_plot_with_meta(fig, args.out, dpi=180,
        caption=f'GPU model: {hardware}. Each point averages the same planned structures. Guided curves use sum-q aggregation over increasing '
        'independent trajectories; ordinary curves show frequency, sum q and predicted-rollout-F1-weighted votes. '
        'The F1 rule is an exploratory addition after the ordinary-pool pilot. Search cost includes discarded '
        'proposals, lookahead, population branches, repair and final scoring. Ordinary frequency omits proofreader cost. '
        'GPU resource seconds sum the work for all trajectories; they are not parallel serving latency. Model staging '
        'and loading are excluded. These are measured compute envelopes, not exactly cost-matched experiments.')
    plt.close(fig)


if __name__ == '__main__':
    main()
