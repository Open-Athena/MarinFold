"""Plot validation-only candidate coverage and aggregation baselines."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Render figures from committed structure-bootstrap summary CSVs."""
    frame = pd.read_csv(HERE / 'data/existing_validation_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    for metric, label in [('oracle_coverage', 'Native contacts in candidate pool'),
                          ('r_precision', 'Final map R-precision')]:
        rows = frame[(frame.method == 'sumq') & (frame.metric == metric)].sort_values('rollouts')
        axes[0].plot(rows.rollouts, rows['mean'], 'o-', label=label)
        axes[0].fill_between(rows.rollouts, rows.ci_low, rows.ci_high, alpha=.15)
    axes[0].set(xlabel='Independent rollouts', ylabel='Fraction', ylim=(0, 1), xticks=[1, 2, 4, 8],
                title='Pool coverage leaves ranking headroom')
    axes[0].legend(fontsize=8)
    labels = dict(frequency='Frequency', sumq='Sum q', meanq='Mean q',
        sqrt_frequency_meanq='√frequency × mean q', diversity_sumq='Diversity-weighted sum q')
    for i, metric in enumerate(['r_precision', 'long_r_precision']):
        rows = frame[(frame.rollouts == 8) & (frame.metric == metric)].set_index('method').loc[list(labels)]
        axes[1].barh([j + .35 * i for j in range(len(rows))], rows['mean'], height=.32,
                     label='All eligible' if i == 0 else 'Long-range')
    axes[1].set(yticks=[j + .175 for j in range(len(labels))], yticklabels=list(labels.values()),
                xlabel='R-precision', xlim=(0, .65), title='Eight-rollout aggregation')
    axes[1].legend(fontsize=8)
    fig.suptitle('Existing validation rollouts · 1,552 structures · 287 homology groups', fontsize=12)
    save_plot_with_meta(fig, HERE / 'plots/existing_validation.png',
        caption='Validation-only reanalysis. Bands resample homology groups (95% intervals). '
        'Oracle coverage uses labels only to bound achievable recall. The small number of structures '
        'with fewer than eight usable rollouts retains all available samples. Long-range means separation ≥24.')


if __name__ == '__main__':
    main()
