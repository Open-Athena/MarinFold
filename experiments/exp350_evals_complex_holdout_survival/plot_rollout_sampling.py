"""Plot oracle interface sampling curves from the frozen 17-target test set."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Render best-map quality, R-precision and pooled contact recovery."""
    with (HERE / 'data/sampling_v1/summary.csv').open() as handle:
        rows = [row for row in csv.DictReader(handle) if row['split'] == 'test']
    budgets = np.array([int(row['k']) for row in rows])
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for axis, metric, title in zip(
        axes[:2], ('expected_best_f1', 'expected_best_r_precision'),
        ('Best individual interface: F1', 'Best individual interface: R-precision'), strict=True,
    ):
        values = np.array([float(row['mean_' + metric]) for row in rows])
        low = np.array([float(row[metric + '_95_low']) for row in rows])
        high = np.array([float(row[metric + '_95_high']) for row in rows])
        axis.plot(budgets, values, 'o-', color='#37679e', label='Oracle best sample')
        axis.fill_between(budgets, low, high, color='#37679e', alpha=.15)
        for k, value in zip(budgets, values, strict=True):
            axis.annotate(f'{value:.3f}', (k, value), xytext=(0, 8),
                          textcoords='offset points', ha='center', fontsize=9)
        axis.set_title(title)
    axes[0].plot(budgets, [float(row['mean_random_expected_best_f1']) for row in rows],
                 's--', color='#888888', label='Matched random best sample')
    axes[1].plot(budgets, [float(row['mean_prefix_consensus_r_precision']) for row in rows],
                 's--', color='#ce8647', label='Consensus of first k samples')
    coverage = [float(row['mean_expected_union_recall']) for row in rows]
    axes[2].plot(budgets, coverage, 'o-', color='#54866b')
    axes[2].plot(budgets, [float(row['mean_random_expected_union_recall']) for row in rows],
                 's--', color='#888888', label='Matched random pool')
    axes[2].legend(loc='upper left', fontsize=8)
    for k, value in zip(budgets, coverage, strict=True):
        axes[2].annotate(f'{value:.1%}', (k, value), xytext=(0, 8),
                        textcoords='offset points', ha='center', fontsize=9)
    axes[2].set_title('True contacts seen anywhere in pool')
    axes[2].set_ylabel('Mean fraction of experimental contacts')
    axes[2].set_ylim(0, 1)
    for axis in axes:
        axis.set_xscale('log')
        axis.set_xticks(budgets, [str(k) for k in budgets])
        axis.set_xlim(.8, 1250)
        axis.set_xlabel('Number of attempted rollouts')
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(alpha=.15)
    for axis in axes[:2]:
        axis.set_ylim(bottom=0)
        axis.set_ylabel('Mean across 17 test targets')
        axis.legend(loc='upper left', fontsize=8)
    figure.suptitle('MarinFold inter-chain sampling: oracle diagnostic on pair-held-out FoldBench', fontsize=13)
    figure.tight_layout()
    save_plot_with_meta(
        figure, HERE / 'plots/rollout_sampling.png',
        caption=(
            'Exploratory follow-up on the frozen 17-target test split, 1000 attempted rollouts per target. '
            'Oracle curves average exact best-of-k expectations over subsets of each saved pool; '
            'shading is the 95% homology-group bootstrap interval. F1 and R-precision are maximized '
            'separately; single-map R-precision averages tied binary scores. Consensus uses fixed '
            'prefixes and the original stable top-R convention. The matched random F1 control '
            'preserves each sample contact count. Union recall pools contacts across samples and '
            'does not imply a single correct map. Unfinished attempts receive zero credit.'
        ),
    )


if __name__ == '__main__':
    main()
