"""Account exactly for true contacts gained and lost by sum-q reranking."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from build_summary import save_plot_with_meta
from metrics import aggregate, bootstrap, map_metrics, rank_key


def main() -> None:
    """Decompose paired R-precision changes on complete independent pools."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plot', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get('pool_mode') != 'independent':
        raise ValueError('This analysis requires independently requested pools')
    targets = {row['entry_id']: row for row in pq.read_table(args.targets,
        filters=[('entry_id', 'in', plan['entries'])]).to_pylist()}
    records = []
    for entry in plan['entries']:
        target = targets[entry]
        if target['split'] != plan['split']:
            raise ValueError('Target split differs from the frozen plan')
        for size in plan.get('pool_sizes', [8, 32, 128]):
            key = args.root / entry / f'pool-{size}'
            if not key.with_suffix('.done.json').exists():
                raise ValueError('Incomplete planned pool: ' + str(key))
            rows = pq.read_table(key.with_suffix('.parquet')).to_pylist()
            if len(rows) != size or {r['sample'] for r in rows} != set(range(size)):
                raise ValueError('Missing or duplicate ordinary rollouts')
            votes, confidence = aggregate(rows, 'frequency'), aggregate(rows, 'sumq')
            if set(votes) != set(confidence):
                raise ValueError('The two rankings must use the identical candidate union')
            for separation in [6, 24]:
                truth = {tuple(pair) for pair in target['contacts'] if pair[1] - pair[0] >= separation}
                if not truth:
                    continue
                selected = []
                for scores in [votes, confidence]:
                    pairs = [pair for pair in scores if pair[1] - pair[0] >= separation]
                    selected.append(set(sorted(pairs, key=lambda pair: (-scores[pair], rank_key(pair)))[:len(truth)]))
                removed, added = selected[0] - selected[1], selected[1] - selected[0]
                if len(removed) != len(added):
                    raise ValueError('Reranking changed the number of filled prediction slots')
                gained = len(added & truth) / len(truth)
                lost = len(removed & truth) / len(truth)
                delta = (map_metrics(confidence, truth, target['L'], separation)['r_precision']
                         - map_metrics(votes, truth, target['L'], separation)['r_precision'])
                if not np.isclose(gained - lost, delta, atol=1e-12, rtol=0):
                    raise ValueError('Contact-level accounting does not reproduce the R-precision change')
                records.append(dict(entry_id=entry, group_id=target['group_id'], L=target['L'],
                    rollouts=size, separation=separation, native_contacts=len(truth), swapped=len(added),
                    swapped_fraction=len(added) / len(truth), gained_correct_fraction=gained,
                    lost_correct_fraction=lost, difference=delta,
                    added_precision=len(added & truth) / len(added) if added else np.nan,
                    removed_precision=len(removed & truth) / len(removed) if removed else np.nan,
                    added_mean_votes=np.mean([votes[pair] for pair in added]) if added else np.nan,
                    removed_mean_votes=np.mean([votes[pair] for pair in removed]) if removed else np.nan))
    frame = pd.DataFrame(records)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / 'reranking_per_protein.csv.gz', index=False)
    metrics = ['swapped_fraction', 'gained_correct_fraction', 'lost_correct_fraction', 'difference',
               'added_precision', 'removed_precision', 'added_mean_votes', 'removed_mean_votes']
    records = []
    for (size, separation), rows in frame.groupby(['rollouts', 'separation']):
        row = dict(rollouts=size, separation=separation, structures=len(rows), groups=rows.group_id.nunique())
        for metric in metrics:
            row[metric] = rows[metric].mean()
            row[metric + '_low'], row[metric + '_high'] = bootstrap(rows[metric].tolist(), rows.group_id.tolist())
        records.append(row)
    summary = pd.DataFrame(records)
    summary.to_csv(args.out / 'reranking_summary.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for axis, separation, title in zip(axes, [6, 24], ['All contacts', 'Long-range contacts'], strict=True):
        rows = summary[summary.separation == separation].sort_values('rollouts')
        x = np.arange(len(rows))
        axis.bar(x, rows.gained_correct_fraction, color='#4393c3', label='Correct contacts gained / R')
        axis.bar(x, -rows.lost_correct_fraction, color='#d6604d', label='Correct contacts lost / R')
        axis.errorbar(x, rows.difference, yerr=np.stack([rows.difference - rows.difference_low,
                      rows.difference_high - rows.difference]), fmt='ko', capsize=3,
                      label='Net R-precision change, 95% interval')
        axis.axhline(0, color='#555555', linewidth=.8)
        axis.set_xticks(x, [str(n) for n in rows.rollouts])
        axis.set(xlabel='Independent rollouts per protein', title=title)
        axis.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Change when ranking by sum q instead of frequency')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8, frameon=False)
    fig.suptitle('What changes when the proofreader reranks an ordinary pool?')
    fig.tight_layout(rect=(0, .10, 1, .96))
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    save_plot_with_meta(fig, args.plot, dpi=180, caption=(
        'Both rankings use exactly the same sampled candidate union and deterministic label-independent tie breaks. '
        'Bars count true contacts entering and leaving the top R; their sum equals the observed R-precision change, '
        'verified for every input. Proteins are weighted equally; intervals resample homology groups. '
        'R uses native contact count only for retrospective evaluation. This is an adaptive diagnostic, not a new policy.'))
    plt.close(fig)
    print(summary[['rollouts', 'separation', 'structures', 'swapped_fraction', 'difference',
                   'difference_low', 'difference_high']].to_string(index=False))


if __name__ == '__main__':
    main()
