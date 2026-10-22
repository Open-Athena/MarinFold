"""Plot complete validation threshold mixtures and their fixed practical controls."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta


def main() -> None:
    """Separate fixed-rollout comparisons from larger pooled mixtures."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    entries = set(json.loads(args.plan.read_text())['entries'])
    frame = pd.concat([pd.read_csv(path) for path in args.input], ignore_index=True)
    choices = [
        ('control', 4, 'sumq', 'Ordinary control · 4'),
        ('top50-k16', 4, 'sumq', 'Top 50% · 4'),
        ('mix-thresholds', 4, 'sumq', '10/25/50% mixture · 4'),
        ('mix-control-thresholds', 4, 'sumq', 'Ordinary + thresholds · 4'),
        ('mix-thresholds', 12, 'sumq', '10/25/50% full pool · 12'),
        ('mix-control-thresholds', 16, 'f1_power2', 'Ordinary + thresholds · 16 (F1²)'),
        ('ordinary-direct', 128, 'frequency', 'Fast ordinary pool · 128 (votes)'),
    ]
    rows = []
    for policy, count, method, label in choices:
        subset = frame[(frame.policy == policy) & (frame.rollouts == count) &
                       (frame.method == method) & frame.entry_id.isin(entries)]
        if subset.entry_id.duplicated().any() or set(subset.entry_id) != entries:
            raise ValueError('Incomplete or duplicated cohort: ' + label)
        if subset.hardware.nunique() != 1 or subset.hardware.str.contains(';').any():
            raise ValueError('Mixed hardware in ' + label)
        rows.append(dict(label=label, hardware=subset.hardware.iloc[0],
            **subset[['r_precision', 'long_r_precision', 'oracle_coverage', 'elapsed_seconds']].mean()))
    summary = pd.DataFrame(rows)
    if summary.hardware.nunique() != 1:
        raise ValueError('Compute comparisons require the same GPU model')
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.1), gridspec_kw={'width_ratios': [1.45, 1, 1]})
    y = np.arange(len(summary))
    axes[0].barh(y - .17, summary.r_precision, height=.32, color='#2962a3', label='All contacts')
    axes[0].barh(y + .17, summary.long_r_precision, height=.32, color='#dc8a32', label='Long range')
    axes[0].set(xlim=(0, .69), xlabel='R-precision', yticks=y, yticklabels=summary.label)
    axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=2, fontsize=9, frameon=False)
    axes[1].barh(y, summary.oracle_coverage, height=.62, color='#4f8a65')
    axes[1].set(xlim=(0, 1.1), xlabel='Native-contact coverage')
    axes[2].barh(y, summary.elapsed_seconds, height=.62, color='#7b668e')
    axes[2].set(xlim=(0, 225), xlabel='Mean warm H100 seconds')
    for axis in axes:
        axis.invert_yaxis()
        axis.axhline(3.5, color='.5', linestyle='--', linewidth=1)
        axis.grid(axis='x', alpha=.2)
        axis.set_axisbelow(True)
    for axis in axes[1:]:
        axis.set(yticks=y, yticklabels=[])
    for index, row in summary.iterrows():
        axes[0].text(row.r_precision + .006, index - .17, f'{row.r_precision:.3f}', va='center', fontsize=8)
        axes[0].text(row.long_r_precision + .006, index + .17, f'{row.long_r_precision:.3f}', va='center', fontsize=8)
        axes[1].text(row.oracle_coverage + .014, index, f'{row.oracle_coverage:.3f}', va='center', fontsize=9)
        axes[2].text(row.elapsed_seconds + 3, index, f'{row.elapsed_seconds:.1f}', va='center', fontsize=9)
    fig.suptitle('Threshold mixtures: fixed-budget and larger-pool validation comparisons')
    fig.text(.5, .02, '32 proteins · dashed line separates four-rollout comparisons from larger pools · rotations averaged within protein',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .95))
    save_plot_with_meta(fig, args.out, dpi=180,
        caption='Exploratory validation-only reuse of top-10%, top-25%, top-50% and ordinary constrained trajectories. '
        'Four-rollout mixtures rotate component assignments over four distinct sample identities and average metrics '
        'within each protein; rotations reuse trajectories and are not independent observations. Full pools include '
        'four trajectories per component (12 or 16). Sum q is used except where labeled. The full 16-rollout F1-squared '
        'point is the highest all-contact mixture result among three prespecified aggregation rules; this display '
        'does not constitute a separate confirmation. Warm cost sums the original included-trajectory inference '
        'timings and excludes loading, staging and CPU aggregation. These mixtures add no GPU work. Top-20% and '
        'within-rollout threshold changes were not tested. Reserved-test configurations remain unchanged.')
    plt.close(fig)


if __name__ == '__main__':
    main()
