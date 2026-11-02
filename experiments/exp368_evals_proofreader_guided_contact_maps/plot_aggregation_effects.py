"""Plot paired aggregation effects on a complete validation cohort."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from build_summary import save_plot_with_meta

LABELS = {
    'sumq': 'Sum contact confidence',
    'meanq': 'Mean contact confidence',
    'sqrt_frequency_meanq': 'Sqrt(votes) × mean confidence',
    'diversity_sumq': 'Diversity-discounted sum',
    'frequency_then_meanq': 'Confidence breaks vote ties',
    'f1_weighted_frequency': 'Votes weighted by rollout F1',
}


def main() -> None:
    """Compare every aggregation rule with frequency on the same generated pool."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--title', required=True)
    args = parser.parse_args()
    completeness = pd.read_csv(args.data / 'completeness.csv')
    if not completeness.complete.all():
        raise ValueError('Aggregation effects require the complete planned cohort')
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)
    for column, size in enumerate([8, 128]):
        paired = pd.read_csv(args.data / f'paired_vs_frequency{size}.csv')
        paired = paired[paired.rollouts == size]
        for row, metric in enumerate(['r_precision', 'long_r_precision']):
            values = paired[paired.metric == metric].set_index('method')
            if set(values.index) != set(LABELS) or values.structures.nunique() != 1:
                raise ValueError('Missing aggregation methods or inconsistent comparison cohort')
            values = values.loc[list(LABELS)]
            ax = axes[row, column]
            color = '#2563eb' if row == 0 else '#15803d'
            ax.errorbar(values.difference, range(len(values)),
                xerr=[values.difference - values.low, values.high - values.difference],
                fmt='o', color=color, capsize=3)
            ax.axvline(0, color='#64748b', lw=1)
            ax.set_yticks(range(len(values)), LABELS.values())
            ax.set_title(f'{size} rollouts · ' + ('all contacts' if row == 0 else 'long-range contacts'))
            ax.grid(axis='x', alpha=.2)
            if row == 1:
                ax.set_xlabel('R-precision difference from frequency ranking')
    axes[0, 0].invert_yaxis()
    fig.suptitle(args.title)
    fig.tight_layout()
    save_plot_with_meta(fig, args.out, dpi=180,
        caption='Paired differences on identical generated pools. Error bars are 95% homology-group '
        'bootstrap intervals, without multiplicity adjustment. All planned structures are included; '
        'long-range metrics exclude structures with no native contact at sequence separation ≥24. '
        'F1 weighting and confidence tie-breaking were added after the ordinary-pool pilot and remain exploratory. '
        'No new reserved-test scores are shown.')
    plt.close(fig)


if __name__ == '__main__':
    main()
