"""Plot map precision/recall and the calibration of emitted contact judgments."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from build_summary import save_plot_with_meta


def main() -> None:
    """Compare explicit configurations on one complete structure cohort."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', type=Path, required=True)
    parser.add_argument('--spec', nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--title', required=True)
    args = parser.parse_args()
    tables = {name: pd.concat([pd.read_csv(path / f'{name}.csv.gz') for path in args.data], ignore_index=True)
              for name in ['quality', 'calibration', 'curves']}
    selected = []
    cohort = None
    for spec in args.spec:
        policy, count, method = spec.split(':')
        rows = {name: frame[(frame.policy == policy) & (frame.rollouts == int(count)) & (frame.method == method)]
                for name, frame in tables.items()}
        quality = rows['quality']
        entries = set(quality.entry_id)
        if not entries or quality.entry_id.duplicated().any():
            raise ValueError('Missing or duplicated map configuration: ' + spec)
        if cohort is not None and entries != cohort:
            raise ValueError('Diagnostic plots require the same structure cohort')
        cohort = entries
        selected.append((spec, rows))
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9))
    coverage_points = {}
    for index, (spec, rows) in enumerate(selected):
        color = plt.get_cmap('tab10')(index)
        for axis, separation in zip(axes[0], [6, 24], strict=True):
            frame = rows['curves']
            frame = frame[frame.separation == separation]
            curve = frame.groupby('budget_L')[['precision', 'recall']].mean()
            axis.plot(curve.recall, curve.precision, 'o-', color=color, label=spec, markersize=4)
            axis.set(xlabel='Recall at fixed L-based prediction budgets', ylabel='Precision', xlim=(0, 1), ylim=(0, 1),
                     title='All eligible contacts' if separation == 6 else 'Long-range contacts')
        bins = rows['calibration'].groupby('bin')[['count', 'sum_probability', 'positives']].sum()
        bins = bins[bins['count'] > 0]
        axes[1, 0].plot(bins.sum_probability / bins['count'], bins.positives / bins['count'], 'o-', color=color,
                       label=spec, markersize=4)
        quality = rows['quality']
        point = (quality.pairwise_jaccard.mean(), quality.oracle_coverage.mean())
        if point not in coverage_points:
            axes[1, 1].scatter(*point, color=color, label=spec, s=65)
            coverage_points[point] = []
        coverage_points[point].append(str(index + 1))
    for point, labels in coverage_points.items():
        axes[1, 1].annotate(', '.join(labels), point, xytext=(6, 5), textcoords='offset points')
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].plot([0, 1], [0, 1], '--', color='.5', linewidth=1)
    axes[1, 0].set(xlabel='Predicted contact probability', ylabel='Observed contact accuracy',
                   title='Emitted-contact calibration', xlim=(0, 1), ylim=(0, 1))
    axes[1, 1].set(xlabel='Mean trajectory-pair Jaccard overlap', ylabel='Native contacts appearing in the pool',
                   title='Diversity and oracle coverage', xlim=(0, 1), ylim=(0, 1))
    for axis in axes.flat:
        axis.grid(alpha=.2)
    fig.suptitle(args.title)
    fig.tight_layout()
    save_plot_with_meta(fig, args.out, dpi=180,
        caption='Curves average over the same structures at budgets 0.1, 0.25, 0.5, 1, 2, 4, 8 and 16 times sequence length. '
        'Unfilled slots count as misses; curves do not use native contact counts to choose the prediction budget. '
        'Calibration pools first occurrences within each rollout and is contact-weighted. '
        'Diversity and coverage are structure means; at most 128 deterministically sampled trajectory pairs measure overlap. '
        'Numbers in the coverage panel follow legend order; multiple aggregators of the same pool share calibration and diversity.')
    plt.close(fig)


if __name__ == '__main__':
    main()
