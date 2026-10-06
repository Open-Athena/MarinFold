"""Plot amino-acid and residue-degree controls for oracle interface sampling."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Compare the same frozen test maps against both conditional nulls."""
    with (HERE / 'data/sampling_controls_v1/summary.csv').open() as handle:
        rows = [row for row in csv.DictReader(handle) if row['split'] == 'test']
    with (HERE / 'data/sampling_v1/summary.csv').open() as handle:
        original = [row for row in csv.DictReader(handle) if row['split'] == 'test']
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    styles = [('amino_acid_pair', 'Amino-acid pair counts', '#da9448'),
              ('residue_degree', 'Each residue contact count', '#885ca6')]
    for axis, metric, title in zip(axes, ('f1', 'r_precision'), ('Interface F1', 'R-precision'), strict=True):
        model = [row for row in rows if row['arm'] == 'amino_acid_pair' and row['metric'] == metric]
        budgets = [int(row['k']) for row in model]
        values = [float(row['mean_model']) for row in model]
        axis.plot(budgets, values, 'o-', color='#37679e', label='MarinFold oracle best')
        for k, value in zip(budgets, values, strict=True):
            axis.annotate(f'{value:.3f}', (k, value), xytext=(0, 8),
                          textcoords='offset points', ha='center', fontsize=9)
        for arm, label, color in styles:
            values = [float(row['mean_null']) for row in rows if row['arm'] == arm and row['metric'] == metric]
            axis.plot(budgets, values, 's--', label=label, color=color)
        if metric == 'f1':
            axis.plot(budgets, [float(row['mean_random_expected_best_f1']) for row in original],
                      '^:', color='#888888', label='Uniform pairs, matched size')
        axis.set_xscale('log')
        axis.set_xticks(budgets, [str(k) for k in budgets])
        axis.set_ylim(bottom=0)
        axis.set_xlim(.8, 1250)
        axis.set_title(title)
        axis.set_xlabel('Number of attempted rollouts')
        axis.set_ylabel('Mean oracle best across 17 test targets')
        axis.legend(loc='upper left', fontsize=8)
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(alpha=.15)
    figure.suptitle('How much of best-of-many comes from simple contact biases?', fontsize=13)
    figure.tight_layout()
    save_plot_with_meta(
        figure, HERE / 'plots/sampling_controls.png',
        caption=(
            'Same 17 frozen test targets and 1000 attempted maps per target. '
            'All curves use ground-truth oracle selection; F1 and R-precision are maximized separately. '
            'The AA control exactly preserves each chain-A/chain-B residue-type pair count. '
            'The degree control preserves every residue degree using a uniform-stationary bipartite '
            'switch chain (100 proposals per edge burn-in, 20 between draws; 1000 null pools). '
            'The latter is approximate and checked against longer chains, as recorded in the result tables. '
            'Ten unfinished attempts retain zero credit. Degree conditioning also preserves any genuine '
            'interface-residue localization, so it specifically tests residue-pairing information.'
        ),
    )


if __name__ == '__main__':
    main()
