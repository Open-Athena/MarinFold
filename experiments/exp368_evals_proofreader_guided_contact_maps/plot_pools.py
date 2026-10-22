"""Plot measured ordinary-pool accuracy, candidate coverage and inference cost."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from build_summary import save_plot_with_meta


def main() -> None:
    """Render complete-cohort pool comparisons with uncertainty intervals."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--title', default='32 validation proteins: ordinary rollout pools')
    methods = ['frequency', 'sumq', 'meanq', 'f1_weighted_frequency']
    parser.add_argument('--methods', nargs='+', choices=methods,
                        default=['frequency', 'sumq', 'meanq'])
    parser.add_argument('--coverage-method', choices=methods, default='sumq')
    args = parser.parse_args()
    summary = pd.read_csv(args.data / 'summary.csv')
    if summary.structures.nunique() != 1:
        raise ValueError('Pool plot requires a complete common cohort')
    completeness = pd.read_csv(args.data / 'completeness.csv')
    ordinary = completeness[completeness.policy.str.startswith('ordinary-')]
    if ordinary.empty or not ordinary.complete.all():
        raise ValueError('Ordinary generation/scoring has not completed its planned cohort')
    if not set(args.methods + [args.coverage_method]) <= set(summary.method):
        raise ValueError('Requested plot methods are absent from the summary')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    colors = {'frequency': '#2563eb', 'sumq': '#d97706', 'meanq': '#64748b',
              'f1_weighted_frequency': '#9333ea'}
    for method, label in [('frequency', 'Frequency'), ('sumq', 'Sum proofreader q'),
                          ('meanq', 'Mean q'), ('f1_weighted_frequency', 'Rollout F1-weighted votes')]:
        if method not in args.methods:
            continue
        rows = summary[summary.method == method].sort_values('rollouts')
        for ax, metric, title in zip(axes[:2], ['r_precision', 'long_r_precision'],
                                     ['All-contact R-precision', 'Long-range R-precision'], strict=True):
            ax.errorbar(rows.rollouts, rows[metric],
                yerr=[rows[metric] - rows[metric + '_low'], rows[metric + '_high'] - rows[metric]],
                marker='o', color=colors[method], label=label, capsize=3)
            ax.set(xscale='log', xlabel='Number of independent rollouts', ylabel=title)
            ax.set_xticks(rows.rollouts, rows.rollouts.astype(str))
            ax.grid(alpha=.2)
    rows = summary[summary.method == args.coverage_method].sort_values('rollouts')
    axes[2].plot(rows.rollouts, rows.oracle_coverage, 'o-', color='#15803d', label='Native contacts in pool')
    axes[2].plot(rows.rollouts, rows.r_precision, 'o-', color=colors[args.coverage_method], label='Recovered at R predictions')
    axes[2].set(xscale='log', xlabel='Number of independent rollouts', ylabel='Fraction of native contacts', ylim=(0, 1))
    axes[2].set_xticks(rows.rollouts, rows.rollouts.astype(str))
    axes[2].grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.suptitle(args.title)
    fig.tight_layout()
    save_plot_with_meta(fig, args.out, dpi=180,
        caption='Mean over the same validation structures. Error bars are 95% homology-group bootstrap intervals. '
        'Oracle coverage measures whether a native contact was proposed; it is not an attainable ranking result. '
        'Long-range means sequence separation at least 24. No test-set results appear here.')
    plt.close(fig)


if __name__ == '__main__':
    main()
