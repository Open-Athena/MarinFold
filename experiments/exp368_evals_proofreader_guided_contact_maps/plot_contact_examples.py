"""Show fixed-L contact maps for three proteins chosen before accuracy inspection."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from build_summary import save_plot_with_meta
from metrics import aggregate, rank_key


def collect(root: Path, targets_path: Path, selection_path: Path, data: Path) -> None:
    """Save reference pairs and top-L predictions as standalone figure inputs."""
    selection = json.loads(selection_path.read_text())
    entries = [item['entry_id'] for item in selection['references']]
    targets = {row['entry_id']: row for row in pq.read_table(targets_path,
        filters=[('entry_id', 'in', entries)]).to_pylist()}
    records, sources = [], {}
    for entry in entries:
        target = targets[entry]
        if target['split'] != 'validation':
            raise ValueError('Examples must use validation proteins')
        path = root / entry / 'pool-128.parquet'
        if not path.with_suffix('.done.json').exists():
            raise ValueError('Example requires a complete ordinary pool')
        rows = pq.read_table(path).to_pylist()
        if len(rows) != 128 or {row['sample'] for row in rows} != set(range(128)):
            raise ValueError('Incomplete or duplicated example rollout identities')
        sources[entry] = hashlib.sha256(path.read_bytes()).hexdigest()
        pairs = {'reference': sorted(map(tuple, target['contacts']))}
        for method in ['frequency', 'f1_weighted_frequency']:
            scores = aggregate(rows, method)
            pairs[method] = sorted(scores, key=lambda pair: (-scores[pair], rank_key(pair)))[:target['L']]
        for method, selected in pairs.items():
            records.extend(dict(entry_id=entry, L=target['L'], method=method, i=i, j=j)
                           for i, j in selected)
    data.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(data, index=False)
    data.with_suffix('.provenance.json').write_text(json.dumps(dict(
        selection=selection['selection'], selection_file=str(selection_path),
        source_pool_sha256=sources, prediction_budget='L unique eligible pairs',
        cohort='Three prespecified numerical-check proteins from the complete validation pilot'), indent=2) + '\n')


def main() -> None:
    """Collect inputs once or render entirely from the published coordinate CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path)
    parser.add_argument('--targets', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.root:
        if not args.targets or not args.selection:
            raise ValueError('Collection requires targets and the prespecified selection')
        collect(args.root, args.targets, args.selection, args.data)
    frame = pd.read_csv(args.data)
    entries = frame[['entry_id', 'L']].drop_duplicates().sort_values(['L', 'entry_id'])
    if len(entries) != 3 or frame.duplicated(['entry_id', 'method', 'i', 'j']).any():
        raise ValueError('Expected three examples without duplicate pairs')
    methods = ['reference', 'frequency', 'f1_weighted_frequency']
    names = ['Experimental reference', '128 rollouts: frequency', '128 rollouts: F1-weighted votes']
    cmap = ListedColormap(['white', '#999999', '#2166ac', '#d6604d'])
    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    for row, target in enumerate(entries.itertuples(index=False)):
        observed = frame[frame.entry_id == target.entry_id]
        native = set(map(tuple, observed[observed.method == 'reference'][['i', 'j']].to_numpy()))
        for column, method in enumerate(methods):
            selected = set(map(tuple, observed[observed.method == method][['i', 'j']].to_numpy()))
            ax = axes[row, column]
            layers = [(native, 1)]
            if method != 'reference':
                layers.extend([(selected & native, 2), (selected - native, 3)])
            for pairs, color in layers:
                if not pairs:
                    continue
                coordinates = np.array(sorted(pairs))
                ax.scatter(np.r_[coordinates[:, 0], coordinates[:, 1]],
                    np.r_[coordinates[:, 1], coordinates[:, 0]], color=cmap(color),
                    s=max(1.8, (220 / target.L) ** 2), marker='s', linewidths=0)
            ax.set(xlim=(-.5, target.L - .5), ylim=(-.5, target.L - .5), aspect='equal')
            label = f'{target.entry_id} · L={target.L}'
            if method != 'reference':
                label += f' · P@L={len(selected & native) / target.L:.3f}'
            ax.set_title(label, fontsize=10)
            if row == 0:
                ax.text(.5, 1.15, names[column], transform=ax.transAxes, ha='center', fontsize=11)
            if column == 0:
                ax.set_ylabel('Residue index')
            if row == 2:
                ax.set_xlabel('Residue index')
    fig.legend(handles=[Patch(color=cmap(i), label=label) for i, label in
        [(1, 'Reference / missed native contact'), (2, 'Correct prediction'), (3, 'Incorrect prediction')]],
        loc='lower center', ncol=3, frameon=False)
    fig.suptitle('Concrete contact maps at a sequence-length prediction budget', y=.995)
    fig.tight_layout(rect=(0, .035, 1, .975))
    save_plot_with_meta(fig, args.out, dpi=300,
        args=['--data', str(args.data), '--out', str(args.out)],
        caption='Three validation proteins chosen as the first, middle and last length ranks for earlier '
        'numerical checks, without accuracy selection. Both methods rank the same 128 H100 rollouts and '
        'predict L unique pairs. Matrices are mirrored only for visualization; each pair counts once. '
        'Gray native contacts remain visible where not selected. Point markers are enlarged for legibility '
        'on long sequences. These examples are descriptive and do '
        'not estimate an average effect. References use the inherited exp222 native-only pyconfind definition, '
        'with sequence separation at least six. The published coordinate CSV reproduces this figure.')
    plt.close(fig)


if __name__ == '__main__':
    main()
