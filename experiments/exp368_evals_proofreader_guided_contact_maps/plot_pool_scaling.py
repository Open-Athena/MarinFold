"""Show protein-level variability, length-stratified gains and measured latency."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta
from metrics import bootstrap


def main() -> None:
    """Compare complete eight- and 128-rollout frequency maps on the frozen panel."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--data-out', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    if 'hardware' not in frame or frame.hardware.isna().any() or frame.hardware.nunique() != 1:
        raise ValueError('A latency comparison requires one recorded GPU model')
    hardware = frame.hardware.iloc[0]
    if ';' in hardware:
        raise ValueError('Mixed hardware cannot define a latency comparison')
    entries = set(json.loads(args.plan.read_text())['entries'])
    maps = {}
    for size in [8, 128]:
        rows = frame[(frame.policy == 'ordinary-direct') & (frame.method == 'frequency') & (frame.rollouts == size)]
        if set(rows.entry_id) != entries or rows.entry_id.duplicated().any():
            raise ValueError('Pool does not cover the exact planned cohort')
        maps[size] = rows.set_index('entry_id').sort_index()
    base, candidate = maps[8], maps[128]
    if not base[['group_id', 'L']].equals(candidate[['group_id', 'L']]):
        raise ValueError('Protein identities differ across pools')
    delta = candidate.r_precision - base.r_precision
    records = []
    bounds = [32, 128, 256, 512, 1001]
    if not base.L.between(bounds[0], bounds[-1] - 1).all():
        raise ValueError('Length strata do not cover this cohort')
    for low, high in zip(bounds[:-1], bounds[1:], strict=True):
        mask = base.L.ge(low) & base.L.lt(high)
        for metric in ['r_precision', 'long_r_precision', 'oracle_coverage', 'average_precision']:
            values = candidate.loc[mask, metric] - base.loc[mask, metric]
            lower, upper = bootstrap(values.tolist(), base.loc[mask, 'group_id'].tolist())
            records.append(dict(length_low=low, length_high=high - 1, metric=metric,
                structures=int(values.notna().sum()), groups=base.loc[mask & values.reindex(base.index).notna(), 'group_id'].nunique(),
                baseline_mean=base.loc[mask, metric].mean(), candidate_mean=candidate.loc[mask, metric].mean(),
                difference=values.mean(), low=lower, high=upper))
    strata = pd.DataFrame(records)
    args.data_out.mkdir(parents=True, exist_ok=True)
    strata.to_csv(args.data_out / 'length_strata.csv', index=False)
    summary = dict(structures=len(base), groups=base.group_id.nunique(),
        improved=int((delta > 0).sum()), unchanged=int((delta == 0).sum()), worse=int((delta < 0).sum()),
        mean_difference=delta.mean(), median_difference=delta.median(),
        latency={str(size): dict(mean=rows.elapsed_seconds.mean(), median=rows.elapsed_seconds.median(),
                    p95=rows.elapsed_seconds.quantile(.95), maximum=rows.elapsed_seconds.max()) for size, rows in maps.items()})
    (args.data_out / 'scaling_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={'width_ratios': [1, 1, 1.25]})
    for size, color in [(8, '#64748b'), (128, '#2563eb')]:
        rows = maps[size]
        axes[0].scatter(rows.L, rows.elapsed_seconds, s=10, alpha=.25, color=color,
                        label=f'MarinFold, {size} rollouts')
    axes[0].set(xscale='log', yscale='log', xlabel='Sequence length (residues)',
                ylabel=f'Warm generation seconds\n({hardware})', title='Measured per-protein latency')
    axes[0].legend(fontsize=8)
    axes[0].grid(which='both', alpha=.2)
    axes[1].hist(delta, bins=40, color='#2563eb', alpha=.8)
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set(xlabel='R-precision difference, 128 minus 8', ylabel='Structures',
                title='Paired differences across structures')
    axes[1].text(.97, .96, f"{summary['improved']} improved\n{summary['unchanged']} unchanged\n{summary['worse']} worse",
                 transform=axes[1].transAxes, ha='right', va='top', fontsize=9)
    for metric, offset, color, label in [('r_precision', -.12, '#2563eb', 'All contacts'),
                                        ('long_r_precision', .12, '#d97706', 'Long-range contacts')]:
        rows = strata[strata.metric == metric]
        axes[2].errorbar(rows.difference, np.arange(len(rows)) + offset,
            xerr=np.stack([rows.difference - rows.low, rows.high - rows.difference]),
            fmt='o', color=color, markersize=4, capsize=3, label=label)
    labels = [f'{row.length_low}–{row.length_high} aa (n={row.structures})'
              for row in strata[strata.metric == 'r_precision'].itertuples()]
    axes[2].set_yticks(np.arange(len(labels)), labels)
    axes[2].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[2].set(xlabel='Paired R-precision difference', title='Gain by protein length')
    axes[2].invert_yaxis()
    axes[2].legend(fontsize=8)
    axes[2].grid(axis='x', alpha=.2)
    fig.suptitle(f'{len(base)} validation structures: eight versus 128 frequency-ranked rollouts')
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_plot_with_meta(fig, args.out, caption=(
        'Each latency point is one (protein, pool-size) inference, excluding model load and artifact writes. '
        'Length strata use fixed boundaries at 32, 128, 256, 512 and 1001 residues. '
        'Intervals resample related groups within each stratum and describe paired differences. '
        'Strata are descriptive; they do not select proteins or tune a test policy.'))
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
