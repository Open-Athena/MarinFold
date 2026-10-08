"""Build a visual diagnosis of inter-chain contact predictions from saved rollouts.

Run once with --raw-root to export compact, reproducible plotting CSVs. Later
invocations render from those CSVs and the frozen target table without GPU work
or network access. Cases are selected to illustrate distinct observed failure
patterns; the cover and cohort plots summarize all 17 test targets.
"""

import argparse
import csv
import hashlib
import json
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.image import AxesImage
from matplotlib.patches import Patch

from build_summary import save_plot_with_meta
from score_foldbench_contacts import target_r_precision
from score_rollout_sampling import write_csv

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data/contact_atlas_v1'
PAGES = HERE / 'plots/contact_atlas_v1'
PDF = HERE / 'plots/contact_prediction_atlas.pdf'
TARGETS = HERE / 'data/foldbench_complex_contact_eval_targets.parquet'
CASES = {
    '8wk1-assembly1': 'Often emits no interface contacts',
    '8cqm-assembly1': 'Almost always predicts the wrong interface',
    '7ytu-assembly1': 'Correct contacts appear, but lose the vote',
    '8onf-assembly1': 'Strongest consensus, still many wrong pairs',
    '8jca-assembly1': 'Best individual sample is still incomplete',
    '8smq-assembly1': 'Large interface; correct contacts are scattered',
}
INTERPRETATIONS = {
    '8wk1-assembly1': 'Frequent silence combines with poorly localized contacts when an interface is emitted.',
    '8cqm-assembly1': 'Top-ranked contact endpoints miss the experimental interface on both chains.',
    '7ytu-assembly1': '35 of 37 true pairs appear somewhere, but incorrect pairs win the consensus ranking.',
    '8onf-assembly1': 'Correct contacts occur often, but votes concentrate on a limited and partly wrong region.',
    '8jca-assembly1': "Top-ranked residues often hit chain A's interface, while chain B localization is weaker.",
    '8smq-assembly1': 'Many predicted endpoints lie on the interface, but most selected residue pairings are wrong.',
}
STATES = ('none', 'unresolved_only', 'false_only', 'some_true', 'unfinished')
STATE_LABELS = ('No inter-chain contacts', 'Only unscorable inter-chain contacts',
                'Scorable contacts; zero correct', 'At least one correct contact', 'Unfinished')
STATE_COLORS = ('#b9c1ca', '#b19fc7', '#dc8352', '#379c88', '#333d4b')
TP_COLOR, FP_COLOR, FN_COLOR = '#00866f', '#e17a48', '#557cae'
PAPER, INK, GRAY = '#ffffff', '#203047', '#e4e8eb'


def read_csv(path: Path) -> list[dict]:
    """Read a small plotting table."""
    with path.open() as handle:
        return list(csv.DictReader(handle))


def top_interface(target: dict, votes: Counter) -> set[tuple[int, int]]:
    """Select stable top-R pairs from the frozen resolved universe."""
    left, right = target['resolved_positions_by_chain']
    pairs = [(i, j) for i in left for j in right]
    pairs.sort(key=lambda pair: -votes[pair])
    return set(pairs[:len(target['gt_contacts'])])


def summarize_target(target: dict, samples: list[dict]) -> tuple[dict, Counter, list[dict], list[dict]]:
    """Separate missing predictions, unscorable predictions and wrong predictions."""
    if [row['rollout'] for row in samples] != list(range(1000)):
        raise ValueError(f"{target['stem']}: expected rollout indices 0..999")
    left, right = map(set, target['resolved_positions_by_chain'])
    truth = {tuple(pair) for pair in target['gt_contacts']}
    boundary = target['chain_lengths'][0]
    votes, categories = Counter(), Counter()
    rows, maps = [], []
    for sample in samples:
        complete = sample['finish_reason'] == 'stop'
        all_pairs = {tuple(pair) for pair in sample['contacts']} if complete else set()
        inter = {pair for pair in all_pairs if pair[0] < boundary <= pair[1]}
        scored = {pair for pair in inter if pair[0] in left and pair[1] in right}
        correct = len(scored & truth)
        if not complete:
            state = 'unfinished'
        elif not inter:
            state = 'none'
        elif not scored:
            state = 'unresolved_only'
        elif not correct:
            state = 'false_only'
        else:
            state = 'some_true'
        categories[state] += 1
        votes.update(all_pairs)
        maps.append(scored)
        rows.append({'target_id': target['stem'], 'rollout': sample['rollout'],
                     'complete': int(complete), 'state': state, 'n_all': len(all_pairs),
                     'n_inter': len(inter), 'n_scored': len(scored), 'n_correct': correct,
                     'f1': 2 * correct / (len(scored) + len(truth))})
    completed = [row for row in rows if row['complete']]
    median_count = float(np.median([row['n_scored'] for row in completed]))
    typical = min(completed, key=lambda row: (abs(row['n_scored'] - median_count), row['rollout']))
    oracle = max(completed, key=lambda row: (row['f1'], -row['rollout']))
    top = top_interface(target, votes)
    n_correct_top = len(top & truth)
    if n_correct_top != target_r_precision(target, votes)['n_correct']:
        raise ValueError('Atlas top-R differs from the primary evaluator')
    # In these saved pools, top-R must never manufacture a zero-vote prediction.
    if any(votes[pair] == 0 for pair in top):
        raise ValueError('Atlas top-R would include pairs never emitted by the model')
    endpoint_precision = []
    for side in (0, 1):
        true_residues = {pair[side] for pair in truth}
        predicted_residues = {pair[side] for pair in top}
        endpoint_precision.append(len(true_residues & predicted_residues) / len(predicted_residues))
    total_inter = sum(row['n_inter'] for row in rows)
    total_scored = sum(row['n_scored'] for row in rows)
    total_all = sum(row['n_all'] for row in rows)
    correct_events = sum(row['n_correct'] for row in rows)
    summary = {'target_id': target['stem'], 'group_id': target['group_id'],
               'complex_type': target['complex_type'], 'length_a': target['chain_lengths'][0],
               'length_b': target['chain_lengths'][1], 'n_true': len(truth),
               **{state: categories[state] for state in STATES},
               'n_predicted_events_all': total_all, 'n_predicted_events_inter': total_inter,
               'n_predicted_events_scored': total_scored, 'n_correct_events': correct_events,
               'inter_fraction': total_inter / total_all, 'scored_event_precision': correct_events / total_scored,
               'median_scored_contacts': median_count, 'top_correct': n_correct_top,
               'consensus_r_precision': n_correct_top / len(truth),
               'max_pair_frequency': max(votes[pair] for pair in votes if pair[0] in left and pair[1] in right) / 1000,
               'union_true': sum(votes[pair] > 0 for pair in truth),
               'endpoint_precision_a': endpoint_precision[0], 'endpoint_precision_b': endpoint_precision[1],
               'typical_rollout': typical['rollout'], 'oracle_rollout': oracle['rollout'],
               'oracle_f1': oracle['f1'], 'oracle_correct': oracle['n_correct'],
               'oracle_predicted': oracle['n_scored'],
               'selected_reason': CASES.get(target['stem'], '')}
    examples = [{'mode': mode, 'rollout': row['rollout'], 'i': i, 'j': j}
                for mode, row in (('typical', typical), ('oracle', oracle))
                for i, j in sorted(maps[row['rollout']])]
    return summary, votes, rows, examples


def prepare(raw_root: Path, targets: list[dict]) -> None:
    """Validate source artifacts and save all data needed for offline plotting."""
    source_path = HERE / 'data/sampling_v1/manifest.json'
    manifest = json.loads(source_path.read_text())
    if hashlib.sha256(TARGETS.read_bytes()).hexdigest() != manifest['targets_sha256']:
        raise ValueError('Frozen target file hash changed')
    for name, expected in manifest['raw_files'].items():
        if hashlib.sha256((raw_root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Rollout artifact hash mismatch: {name}')
    samples = defaultdict(list)
    for path in sorted((raw_root / 'rollouts').glob('*.parquet')):
        for row in pq.read_table(path).to_pylist():
            samples[row['stem']].append(row)
    summaries, counts = [], []
    for target in targets:
        rows = sorted(samples[target['stem']], key=lambda row: row['rollout'])
        summary, votes, rollout_rows, examples = summarize_target(target, rows)
        summaries.append(summary)
        if target['stem'] not in CASES:
            continue
        counts.extend(rollout_rows)
        write_csv(DATA / (target['stem'] + '_votes.csv'),
                  [{'i': i, 'j': j, 'votes': count} for (i, j), count in sorted(votes.items())])
        write_csv(DATA / (target['stem'] + '_examples.csv'), examples)
    write_csv(DATA / 'summary.csv', summaries)
    write_csv(DATA / 'rollout_counts.csv', counts)
    metadata = {'n_test_targets': len(targets), 'attempts_per_target': 1000,
                'checkpoint': manifest['checkpoint'], 'selected_cases': CASES,
                'case_selection': 'purposeful diagnostic cases, not a random sample',
                'typical_selection': 'completed map closest to median resolved inter-chain contact count; lowest index breaks ties',
                'oracle_selection': 'maximum experimental inter-chain F1; lowest index breaks ties',
                'frequency_denominator': 'all 1000 attempted rollouts, including unfinished attempts',
                'source_manifest_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
                'source_public_prefix': manifest['public_prefix'],
                'targets_sha256': manifest['targets_sha256'],
                'csv_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.glob('*.csv'))}}
    (DATA / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')


def new_page(title: str, subtitle: str = '') -> plt.Figure:
    """Create a landscape page with a consistent title area."""
    figure = plt.figure(figsize=(13.33, 7.5), facecolor=PAPER)
    figure.text(.045, .947, title, fontsize=22, weight='bold', color=INK, va='top')
    if subtitle:
        figure.text(.045, .895, subtitle, fontsize=10.5, color='#526277', va='top')
    return figure


def finish(figure: plt.Figure, pdf: PdfPages, number: int, stem: str, caption: str) -> None:
    """Save a PDF page plus a preview and reproducibility sidecar."""
    figure.text(.045, .037, 'exp350  |  exp343 step 280154  |  1,000 attempts per complex', fontsize=8, color='#627080')
    figure.text(.045, .015, 'Rebuild: uv run python build_contact_atlas.py', fontsize=7, color='#627080')
    figure.text(.965, .025, str(number), ha='right', fontsize=9, color='#627080')
    pdf.savefig(figure, dpi=300)
    save_plot_with_meta(figure, PAGES / f'{number:02d}_{stem}.png', caption=caption,
                        script='build_contact_atlas.py', args=[], dpi=130, bbox_inches=None)
    plt.close(figure)


def load_case(target: dict) -> tuple[Counter, dict[str, set], list[dict]]:
    """Load the compact plot data for one selected complex."""
    stem = target['stem']
    votes = Counter({(int(row['i']), int(row['j'])): int(row['votes'])
                     for row in read_csv(DATA / (stem + '_votes.csv'))})
    examples = {'typical': set(), 'oracle': set()}
    for row in read_csv(DATA / (stem + '_examples.csv')):
        examples[row['mode']].add((int(row['i']), int(row['j'])))
    counts = [row for row in read_csv(DATA / 'rollout_counts.csv') if row['target_id'] == stem]
    return votes, examples, counts


def valid_mask(target: dict) -> np.ndarray:
    """Mask unresolved residues while retaining full input-sequence coordinates."""
    a, b = target['chain_lengths']
    left, right = target['resolved_positions_by_chain']
    mask = np.zeros((a, b), dtype=bool)
    mask[np.ix_(left, np.array(right) - a)] = True
    return mask


def map_axes(axis: plt.Axes, target: dict) -> None:
    """Label chain-local, one-based input positions without compressing gaps."""
    a, b = target['chain_lengths']
    axis.set_xlim(.5, b + .5)
    axis.set_ylim(a + .5, .5)
    axis.set_xlabel(f"Chain {target['chain_ids'][1]} input position", fontsize=9)
    axis.set_ylabel(f"Chain {target['chain_ids'][0]} input position", fontsize=9)
    axis.tick_params(labelsize=8)


def binary_map(axis: plt.Axes, target: dict, predicted: set | None, title: str) -> None:
    """Draw exact residue-pair cells; orange is false, blue missed, green correct."""
    a, b = target['chain_lengths']
    truth = {tuple(pair) for pair in target['gt_contacts']}
    values = np.zeros((a, b), dtype=np.uint8)
    values[~valid_mask(target)] = 1
    for i, j in truth:
        values[i, j - a] = 4 if predicted is None else 2
    if predicted is not None:
        for i, j in predicted:
            values[i, j - a] = 4 if (i, j) in truth else 3
    colors = ListedColormap([PAPER, GRAY, FN_COLOR, FP_COLOR, TP_COLOR])
    axis.imshow(values, cmap=colors, vmin=0, vmax=4, interpolation='nearest', aspect='auto',
                extent=(.5, b + .5, a + .5, .5))
    axis.set_title(title, fontsize=11, loc='left', pad=10)
    map_axes(axis, target)


def frequency_map(axis: plt.Axes, target: dict, votes: Counter) -> AxesImage:
    """Plot aggregate frequency on a shared logarithmic scale, never smoothed."""
    a, b = target['chain_lengths']
    mask = valid_mask(target)
    values = np.zeros((a, b))
    for (i, j), count in votes.items():
        if i < a <= j:
            values[i, j - a] = count / 1000
    axis.imshow(~mask, cmap=ListedColormap([PAPER, GRAY]), vmin=0, vmax=1,
                interpolation='nearest', aspect='auto', extent=(.5, b + .5, a + .5, .5))
    color = plt.get_cmap('magma_r').copy()
    color.set_bad((0, 0, 0, 0))
    im = axis.imshow(np.ma.masked_where((values == 0) | ~mask, values), cmap=color,
                     norm=LogNorm(.001, 1), interpolation='nearest', aspect='auto',
                     extent=(.5, b + .5, a + .5, .5))
    axis.set_title('MarinFold contact frequency\nAll 1,000 attempted rollouts', fontsize=11, loc='left', pad=10)
    map_axes(axis, target)
    return im


def error_legend(figure: plt.Figure, y: float) -> None:
    """Explain exact-pair error colors on every diagnostic map page."""
    figure.legend(handles=[Patch(color=TP_COLOR, label='Correct predicted pair'),
                           Patch(color=FP_COLOR, label='Wrong predicted pair'),
                           Patch(color=FN_COLOR, label='Missed experimental pair'),
                           Patch(color=GRAY, label='Unresolved / unscorable')],
                  loc='center', bbox_to_anchor=(.5, y), ncol=4, frameon=False, fontsize=9)


def page_cover(pdf: PdfPages, summaries: list[dict]) -> None:
    """Lead with the answer and make selection and oracle limitations explicit."""
    totals = {state: sum(int(row[state]) for row in summaries) for state in STATES}
    n = len(summaries) * 1000
    figure = new_page('Where MarinFold goes wrong on protein complexes',
                      'A contact-map atlas  |  17 FoldBench test complexes  |  Six diagnostic case studies')
    figure.text(.055, .785, 'Usually wrong contacts; sometimes no interface prediction.', fontsize=21, weight='bold', color=INK)
    cards = [(totals['none']/n, 'emit no inter-chain contacts', STATE_COLORS[0]),
             (totals['false_only']/n, 'emit scorable contacts, all wrong', FP_COLOR),
             (totals['some_true']/n, 'include at least one correct pair', TP_COLOR)]
    for x, (fraction, label, color) in zip((.055, .37, .685), cards, strict=True):
        figure.text(x, .675, f'{fraction:.1%}', fontsize=39, weight='bold', color=color)
        figure.text(x, .625, label, fontsize=12, color=INK)
    figure.text(.055, .57, f"Other attempts: {totals['unresolved_only']/n:.1%} predict only unscorable interface pairs; "
                f"{totals['unfinished']} / {n:,} are unfinished.\n"
                '“At least one correct” can still mean a largely wrong map. Categories refer to whole rollouts.',
                fontsize=11, linespacing=1.7, color=INK, va='top')
    figure.text(.055, .445, 'What to look for', fontsize=15, weight='bold', color=INK)
    figure.text(.055, .402,
                '1. Experimental vs frequency heatmaps: do frequent predictions land on the right interface patch?\n'
                '2. Top-R error maps: which high-vote pairs are wrong, and which true pairs are missed?\n'
                '3. A typical rollout vs oracle best@1000: do correct contacts ever occur together?',
                fontsize=12, linespacing=1.8, color=INK, va='top')
    figure.text(.055, .215,
                'Six cases are purposefully chosen to span observed behaviors; they are not a random sample.\n'
                'Cohort summaries always include all 17 test complexes. No additional inference was run.\n'
                'The best@1000 map uses experimental truth for selection. It is a diagnostic ceiling.',
                fontsize=11, linespacing=1.8, color='#526277', va='top')
    finish(figure, pdf, 1, 'cover', 'All-test rollout-level failure categories and guide to the diagnostic atlas.')


def page_cohort(pdf: PdfPages, summaries: list[dict]) -> None:
    """Show every target, preventing the case-study choices from hiding variation."""
    figure = new_page('How often is a rollout empty, wrong, or partly right?',
                      'Each bar represents 1,000 attempts. A partly right map contains at least one true contact; it can contain many false contacts.')
    rows = sorted(summaries, key=lambda row: int(row['some_true']))
    axis = figure.add_axes([.12, .2, .79, .62])
    offset = np.zeros(len(rows))
    for state, label, color in zip(STATES, STATE_LABELS, STATE_COLORS, strict=True):
        values = np.array([int(row[state])/10 for row in rows])
        axis.barh(np.arange(len(rows)), values, left=offset, color=color, label=label, height=.72)
        offset += values
    axis.set_yticks(np.arange(len(rows)), [r['target_id'][:4] + (' *' if r['target_id'] in CASES else '') for r in rows])
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel('Percentage of all attempted rollouts')
    axis.spines[['top', 'right', 'left']].set_visible(False)
    axis.grid(axis='x', alpha=.15)
    axis.legend(loc='upper center', bbox_to_anchor=(.5, -.11), ncol=3, fontsize=8, frameon=False)
    figure.text(.12, .85, '* Included in the detailed case studies', fontsize=9, color='#526277')
    save_plot_with_meta(figure, HERE / 'plots/contact_failure_modes.png',
                        caption='All 17 test complexes, categorized by raw inter-chain emissions, resolved scoring and experimental overlap.',
                        script='build_contact_atlas.py', args=[], dpi=150, bbox_inches=None)
    finish(figure, pdf, 2, 'cohort', 'Complete test cohort: zero raw interface contacts are distinguished from unscorable-only emissions.')


def page_full_maps(pdf: PdfPages, cases: list[dict], summaries: dict) -> None:
    """Put the inter-chain problem in the context of the model's entire output."""
    figure = new_page('The model emits many more intra-chain than inter-chain contacts',
                      'Full two-chain prediction matrices; dashed boundaries separate chains. These panels show output allocation, not intra-chain accuracy.')
    color = plt.get_cmap('magma_r').copy()
    color.set_bad(PAPER)
    image = None
    for index, target in enumerate(cases):
        row, col = divmod(index, 3)
        axis = figure.add_axes([.055 + col*.30, .515 - row*.355, .255, .255])
        votes, _, _ = load_case(target)
        values = np.zeros((target['L'], target['L']))
        for (i, j), count in votes.items():
            values[i, j] = values[j, i] = count / 1000
        image = axis.imshow(np.ma.masked_equal(values, 0), cmap=color, norm=LogNorm(.001, 1),
                            interpolation='nearest', extent=(.5, target['L']+.5, target['L']+.5, .5), aspect='auto')
        boundary = target['chain_lengths'][0] + .5
        axis.axvline(boundary, linestyle='--', linewidth=.8, color='#608388')
        axis.axhline(boundary, linestyle='--', linewidth=.8, color='#608388')
        fraction = float(summaries[target['stem']]['inter_fraction'])
        axis.set_title(f"{target['stem'][:4]}  |  {fraction:.1%} of contact emissions cross chains", fontsize=9, loc='left')
        axis.set_xlabel('Concatenated input position', fontsize=8)
        axis.set_ylabel('Concatenated input position', fontsize=8)
        axis.tick_params(labelsize=7)
    bar = figure.colorbar(image, cax=figure.add_axes([.947, .195, .011, .56]), ticks=[.001, .01, .1, 1])
    bar.ax.set_yticklabels(['0.1%', '1%', '10%', '100%'])
    bar.set_label('Fraction of attempts emitting each pair', fontsize=8)
    bar.ax.yaxis.set_label_position('left')
    figure.text(.055, .078, 'All emitted contacts are shown here, including positions unresolved in the experiment. The next pages score resolved cross-chain pairs only.',
                fontsize=9, color='#526277')
    finish(figure, pdf, 3, 'full_maps', 'Full prediction maps retain all input positions; inter-chain fractions count unique contacts per completed rollout.')


def page_consensus(pdf: PdfPages, target: dict, summary: dict, number: int) -> None:
    """Compare truth and votes, and distinguish wrong residues from wrong pairing."""
    stem = target['stem']
    title = f"{stem[:4]}  |  {CASES[stem]}"
    subtitle = textwrap.shorten(target['title'], width=155, placeholder='...')
    figure = new_page(title, subtitle)
    a, b = target['chain_lengths']
    votes, _, _ = load_case(target)
    truth = {tuple(pair) for pair in target['gt_contacts']}
    top = top_interface(target, votes)
    r, tp = len(truth), len(top & truth)
    figure.text(.045, .844, f"{target['complex_type']}  |  Chains {a} + {b} residues  |  {r} experimental interface contacts  |  Test split", fontsize=10, color=INK)
    axes = [figure.add_axes([x, .395, .24, .375]) for x in (.055, .375, .715)]
    binary_map(axes[0], target, None, f'Experimental interface\n{r} true residue pairs')
    im = frequency_map(axes[1], target, votes)
    cbar = figure.colorbar(im, cax=figure.add_axes([.625, .42, .009, .32]), ticks=[.001, .01, .1, 1])
    cbar.ax.set_yticklabels(['0.1%', '1%', '10%', '100%'])
    cbar.ax.tick_params(labelsize=7)
    binary_map(axes[2], target, top, f'Highest-vote {r} pairs (top-R)\n{tp} correct, {r-tp} wrong, {r-tp} missed')
    error_legend(figure, .322)
    for side, (length, offset, x) in enumerate(((a, 0, .075), (b, a, .555))):
        axis = figure.add_axes([x, .172, .39, .086])
        true_degree = np.bincount([pair[side]-offset for pair in truth], minlength=length)
        predicted_degree = np.bincount([pair[side]-offset for pair in top], minlength=length)
        axis.step(np.arange(length)+1, true_degree, where='mid', color=FN_COLOR, label='Experimental')
        axis.step(np.arange(length)+1, predicted_degree, where='mid', color=FP_COLOR, label='Top-R prediction', alpha=.85)
        axis.set_xlim(.5, length+.5)
        axis.set_ylabel('Contacts', fontsize=8)
        axis.tick_params(labelsize=7)
        axis.set_title(f'Chain {target["chain_ids"][side]}: where the selected contact endpoints land', fontsize=9, loc='left')
        axis.spines[['top', 'right']].set_visible(False)
        if side == 0:
            axis.legend(loc='upper right', fontsize=7, frameon=False)
    figure.text(.055, .116, INTERPRETATIONS[stem], fontsize=10, weight='bold', color=INK)
    figure.text(.055, .078,
                f"Top-R precision = recall = {tp/r:.1%}.  Predicted endpoint residues on the true interface: "
                f"A {float(summary['endpoint_precision_a']):.0%}, B {float(summary['endpoint_precision_b']):.0%}.  "
                f"True pairs ever sampled: {int(summary['union_true'])}/{r}.", fontsize=9, color=INK)
    finish(figure, pdf, number, stem[:4] + '_consensus',
           'Exact-pair heatmaps use full chain-local one-based input coordinates, unresolved positions in gray; top-R uses the evaluator tie convention.')


def page_samples(pdf: PdfPages, target: dict, summary: dict, number: int) -> None:
    """Show a truth-blind typical sample, an explicitly oracle-selected map and the pool."""
    stem = target['stem']
    figure = new_page(f"{stem[:4]}  |  What happens inside individual rollouts?",
                      'Typical = closest to median scorable contact count, selected without truth. Oracle = highest experimental F1 among 1,000 attempts.')
    _, examples, counts = load_case(target)
    truth = {tuple(pair) for pair in target['gt_contacts']}
    for mode, x in (('typical', .055), ('oracle', .375)):
        axis = figure.add_axes([x, .385, .267, .375])
        predicted = examples[mode]
        tp = len(predicted & truth)
        precision = tp/len(predicted) if predicted else 0
        label = 'Contact-count typical' if mode == 'typical' else 'Oracle best@1000'
        binary_map(axis, target, predicted,
                   f"{label}  |  rollout {summary[mode+'_rollout']}\n"
                   f"{tp}/{len(predicted)} predicted correct; {tp}/{len(truth)} true recovered")
        figure.text(x, .301, f'Precision {precision:.1%}  |  Recall {tp/len(truth):.1%}', fontsize=10, color=INK)
    axis = figure.add_axes([.72, .385, .23, .375])
    completed = [row for row in counts if int(row['complete'])]
    for correct, color in ((False, FP_COLOR), (True, TP_COLOR)):
        rows = [row for row in completed if (int(row['n_correct']) > 0) == correct]
        axis.scatter([int(row['n_scored']) for row in rows], [int(row['n_correct']) for row in rows],
                     s=12, alpha=.18, color=color, linewidths=0)
    for mode, marker, color in (('typical', 'x', '#1b2635'), ('oracle', '*', '#6d3d96')):
        selected = counts[int(summary[mode+'_rollout'])]
        axis.scatter(int(selected['n_scored']), int(selected['n_correct']), marker=marker,
                     s=90, color=color, zorder=4, label=mode)
    axis.set_xscale('symlog', linthresh=5)
    axis.set_xlabel('Scorable inter-chain contacts emitted\n(symlog axis; zero included)', fontsize=8)
    axis.set_ylabel('Correct experimental pairs', fontsize=9)
    axis.set_title('All completed rollouts\nDots overlap; see category counts below', fontsize=10, loc='left', pad=10)
    axis.spines[['top', 'right']].set_visible(False)
    axis.tick_params(labelsize=8)
    axis.legend(fontsize=8, frameon=False)
    error_legend(figure, .268)
    for x, state, label, color in zip((.055, .24, .45, .665, .86), STATES,
                                     ('No interface\ncontacts', 'Unscorable\ncontacts only',
                                      'Scorable, but\nall wrong', 'At least one\ncorrect pair', 'Unfinished'), STATE_COLORS, strict=True):
        figure.text(x, .202, f"{int(summary[state])/10:.1f}%", fontsize=23, weight='bold', color=color)
        figure.text(x, .165, label, fontsize=10, va='top', color=INK)
    figure.text(.055, .073, f"All-rollout median scorable contact count (completed only): {float(summary['median_scored_contacts']):g}.  "
                f"Experimental contact count: {len(truth)}.  "
                'A green pair is correct; a green-containing rollout is not necessarily a good map.', fontsize=9, color='#526277')
    finish(figure, pdf, number, stem[:4] + '_samples',
           'Typical and oracle examples use exact emitted contacts, not thresholded consensus; all categories retain the 1000-attempt denominator.')


def page_methods(pdf: PdfPages, number: int) -> None:
    """Keep the report scientifically interpretable without consulting the code."""
    figure = new_page('Reading the maps and reproducing the atlas',
                      'These are diagnostics of contact generation and ranking; no structures were refolded for this report.')
    paragraphs = [
        ('Experimental truth', ('The frozen FoldBench pyconfind interface contacts use degree >=0.001. '
         'Scoring uses only residue pairs with resolved experimental coordinates. This is the same truth and mask as the reported contact evaluation.')),
        ('Coordinates and colors', ('Axes are one-based positions within the full input sequence of each chain, not PDB author residue numbers. '
         'Unresolved rows/columns are gray. White marks eligible cells with no signal in that panel. Cells are unsmoothed; zoom into the PDF to inspect exact residue pairs. '
         'Homodimers use the frozen chain assignment.')),
        ('Frequency and selection', ('Frequency is the fraction of all 1,000 attempts that completed and emitted that pair; it is not calibrated confidence. '
         'Every frequency panel shares a logarithmic 0.1%-100% scale. Top-R selects exactly as many pairs as there are experimental contacts, '
         'with the original stable tie convention. The full-output panels also show unscorable positions.')),
        ('Individual samples', ('The typical map is the completed rollout nearest the median scorable contact count; its choice does not inspect correctness. '
         'The oracle map maximizes experimental F1, so it is unavailable at inference time. Neither map removes its false positives. '
         'Unfinished attempts retain zero credit; their partial maps are excluded.')),
        ('Interpretation', ('An absence of inter-chain contacts differs from contacts aimed at unresolved residues, a wrong interface patch, '
         'or wrong residue pairing within a partly correct patch. The cohort bars, marginal contact counts and exact-pair error colors separate these cases. '
         'The observed six-case patterns are descriptive, not additional blinded benchmark comparisons.')),
    ]
    y = .80
    for title, body in paragraphs:
        figure.text(.055, y, title, fontsize=12, weight='bold', color=INK)
        lines = textwrap.fill(body, width=145)
        figure.text(.055, y-.03, lines, fontsize=10, va='top', linespacing=1.4, color=INK)
        y -= .132
    figure.text(.055, .09, 'Source: public MarinFold bucket, exp350_foldbench_pair_holdout/contact_eval_v1/sampling_v1.\n'
                'Rebuild offline from the committed plotting CSVs: uv run python build_contact_atlas.py', fontsize=9, color='#526277')
    finish(figure, pdf, number, 'methods', 'Coordinate, truth, frequency, case-selection and oracle conventions for this report.')


def main() -> None:
    """Optionally prepare data, then assemble the 16-page contact atlas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path)
    args = parser.parse_args()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'axes.labelcolor': INK,
                         'xtick.color': INK, 'ytick.color': INK, 'text.color': INK})
    targets = [row for row in pq.read_table(TARGETS).to_pylist() if row['split'] == 'test']
    if args.raw_root:
        prepare(args.raw_root, targets)
    provenance = json.loads((DATA / 'manifest.json').read_text())
    if hashlib.sha256(TARGETS.read_bytes()).hexdigest() != provenance['targets_sha256']:
        raise ValueError('Frozen target file hash changed since atlas export')
    for name, expected in provenance['csv_sha256'].items():
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Atlas plot-data hash mismatch: {name}')
    summaries = read_csv(DATA / 'summary.csv')
    by_id = {row['target_id']: row for row in summaries}
    by_target = {row['stem']: row for row in targets}
    selected = [by_target[stem] for stem in CASES]
    PAGES.mkdir(parents=True, exist_ok=True)
    with PdfPages(PDF, metadata={'Title': 'MarinFold contact predictions versus experimental truth',
                                 'Subject': 'Exp350 inter-chain contact failure analysis',
                                 'Author': 'Open Athena / MarinFold'}) as pdf:
        page_cover(pdf, summaries)
        page_cohort(pdf, summaries)
        page_full_maps(pdf, selected, by_id)
        number = 4
        for target in selected:
            page_consensus(pdf, target, by_id[target['stem']], number)
            page_samples(pdf, target, by_id[target['stem']], number + 1)
            number += 2
        page_methods(pdf, number)
    print(f'Wrote {PDF} ({PDF.stat().st_size:,} bytes, {number} pages)')


if __name__ == '__main__':
    main()
