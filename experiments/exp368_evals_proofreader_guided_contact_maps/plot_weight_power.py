"""Plot paired pilot effects of emphasizing predicted rollout quality."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta


def main() -> None:
    """Keep hardware cohorts separate while comparing the same weight exponents."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--h100', type=Path, required=True)
    parser.add_argument('--gb200', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    methods = ['f1_power2', 'f1_power4', 'f1_power8']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    y = np.arange(3)
    for directory, label, color, offset in [(args.h100, 'H100 rollout pools', '#2563eb', -.12),
                                            (args.gb200, 'GB200 rollout pools', '#d97706', .12)]:
        complete = pd.read_csv(directory / 'completeness.csv')
        if len(complete) != 32 or not complete.complete.all():
            raise ValueError('Power comparison requires all 32 committed pilot pools')
        table = pd.read_csv(directory / 'paired_vs_f1.csv')
        if set(table.baseline) != {'ordinary-direct:128:f1_weighted_frequency'}:
            raise ValueError('Unexpected power-comparison baseline')
        for axis, metric, title in zip(axes, ['r_precision', 'long_r_precision'],
                                      ['All-contact R-precision', 'Long-range R-precision'], strict=True):
            rows = table[table.metric == metric].set_index('method').loc[methods]
            if set(rows.structures) != {32 if metric == 'r_precision' else 31}:
                raise ValueError('Incomplete power comparison')
            axis.errorbar(rows.difference, y + offset,
                xerr=np.stack([rows.difference - rows.low, rows.high - rows.difference]),
                fmt='o', color=color, capsize=3, label=label)
            axis.axvline(0, color='.5', linestyle='--', linewidth=1)
            axis.set(title=title, xlabel='Difference from ordinary F1-weighted votes (power 1)')
            axis.grid(axis='x', alpha=.2)
    axes[0].set_yticks(y, ['F1 squared', 'F1 to power 4', 'F1 to power 8'])
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8)
    fig.suptitle('Exploratory weighting refinement: same 128 rollouts per protein')
    fig.tight_layout()
    save_plot_with_meta(fig, args.out, dpi=180, caption=(
        'Paired 95% homology-group bootstrap intervals on the complete 32-protein pilot. '
        'The same proteins are used on both hardware/runtime phases; they are not independent biological cohorts. '
        'Powers 2, 4, 8 were fixed after observing the power-1 validation result, before computing these comparisons. '
        'All variants reuse identical generated pools and reader outputs within each runtime. '
        'Intervals are exploratory and unadjusted for multiple comparisons; no test metrics are used.'))
    plt.close(fig)


if __name__ == '__main__':
    main()
