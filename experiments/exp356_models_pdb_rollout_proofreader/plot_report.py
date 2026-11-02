"""Render the written report's figures from saved experiment measurements."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from build_summary import save_plot_with_meta


HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
PLOTS = HERE / 'plots'
BLUE = '#2166ac'
TEAL = '#148a83'
ORANGE = '#cf6b27'
GRAY = '#77818b'
PREFIXES = ['1', '2', '4', '8', '16', '32', '64', 'full']


def read_json(name: str) -> dict:
    """Read an immutable local result table."""
    return json.loads((DATA / name).read_text())


def save(fig: plt.Figure, name: str, caption: str) -> None:
    """Save a figure and its exact reproducibility metadata."""
    save_plot_with_meta(fig, PLOTS / f'report_{name}.png', caption=caption,
                       script='plot_report.py', args=[], dpi=180)
    plt.close(fig)


def intervals(ax: plt.Axes, records: list[dict], color: str, label: str,
              scale: float = 1.0, offset: float = 0.0) -> None:
    """Draw point estimates with their saved group-bootstrap intervals."""
    means = np.array([x['mean'] for x in records]) * scale
    lows = np.array([x['low'] for x in records]) * scale
    highs = np.array([x['high'] for x in records]) * scale
    ax.errorbar(np.arange(len(records)) + offset, means,
                yerr=[means-lows, highs-means], fmt='o-', color=color,
                capsize=3, linewidth=1.6, markersize=4.5, label=label)


def architecture() -> None:
    """Explain how future supplied contacts can change earlier predictions."""
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.set(xlim=(0, 10.5), ylim=(-.12, 6.2))
    ax.axis('off')

    def box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06',
                                   facecolor=color, edgecolor='#8c99a5', linewidth=.8))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=10)

    def arrow(a: tuple[float, float], b: tuple[float, float]) -> None:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=13,
                                    color='#4d5c68', linewidth=1.3))

    box(.45, 5.2, 9.6, .65,
        'Sequence prompt + actual emitted contacts 1 … k\nPhysically crop the input at the supplied boundary', '#edf2f7')
    box(.45, 3.85, 9.6, .8,
        'Frozen exp277 causal backbone\nToken representations retain the pretrained generator features', '#e1edf8')
    arrow((5.25, 5.15), (5.25, 4.72))
    box(.45, 2.35, 5.8, .95,
        'One vector per emitted contact\n3 token states → 512 dimensions\nAdd learned emission-order embedding', '#e3f2f0')
    box(6.65, 2.35, 3.4, .95,
        'Global token\nPrompt and last-token states\n+ prefix-size features', '#e3f2f0')
    arrow((3.35, 3.8), (3.35, 3.37))
    arrow((8.35, 3.8), (8.35, 3.37))
    box(.45, 1.0, 9.6, .8,
        'Trainable bidirectional contact transformer\n4 layers · 512 dimensions · 8 heads · all supplied contacts attend to each other', '#cce8e4')
    arrow((3.35, 2.3), (3.35, 1.87))
    arrow((8.35, 2.3), (8.35, 1.87))
    box(.45, .02, 5.8, .55, 'Contact probabilities → mean over distinct pairs = precision', '#f5eadb')
    box(6.65, .02, 3.4, .55, 'Global head → recall estimate', '#f5eadb')
    arrow((3.35, .95), (3.35, .64))
    arrow((8.35, .95), (8.35, .64))
    fig.tight_layout(pad=.5)
    save(fig, 'architecture', 'Figure 1. The frozen causal backbone feeds a trainable bidirectional contact encoder. '
         'Later supplied contacts can revise earlier scores; unsupplied continuation is absent. '
         'There are 22,181,378 trainable parameters out of 1,487,731,202 total.')


def dataset() -> None:
    """Show sequence lengths and the experimental structure composition."""
    manifest = pd.read_csv(DATA / 'structure_manifest.csv.gz')
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5), layout='constrained')
    bins = np.linspace(0, 1000, 26)
    for split, color in [('train', BLUE), ('validation', ORANGE), ('test', TEAL)]:
        values = manifest.loc[manifest['split'] == split, 'L']
        axes[0].hist(values, bins=bins, weights=np.full(len(values), 100/len(values)),
                     histtype='step', linewidth=1.7, color=color, label=f'{split} (n={len(values):,})')
    axes[0].set(xlabel='Supplied sequence length in residues', ylabel='Structures per bin (%)',
                title='A  Sequence lengths')
    axes[0].legend(frameon=False, fontsize=8)
    methods = manifest['method'].value_counts()
    main = ['X-RAY DIFFRACTION', 'SOLUTION NMR', 'ELECTRON MICROSCOPY']
    values = [int(methods.get(x, 0)) for x in main]
    values.append(len(manifest)-sum(values))
    axes[1].barh(['X-ray diffraction', 'Solution NMR', 'Electron microscopy', 'Other / combined'],
                 np.array(values)/len(manifest)*100, color=[BLUE, TEAL, ORANGE, GRAY])
    axes[1].invert_yaxis()
    axes[1].set(xlabel='Share of all 53,025 structures (%)', xlim=(0, 100),
                title='B  Experimental methods')
    for i, n in enumerate(values):
        axes[1].text(100*n/len(manifest)+1, i, f'{n:,}', va='center', fontsize=9)
    save(fig, 'dataset', 'Figure 2. Supplied sequence lengths and experimental methods in the 53,025-structure corpus. '
         'Length histograms are normalized separately within each split; method counts combine all splits. '
         'These are chain structures, not independent sequences.')


def development() -> None:
    """Separate architecture comparison from training-sample diagnostics."""
    v1 = read_json('validation_step2000_metrics.json')['full']
    v2 = read_json('frozen_validation_step2000_metrics.json')['full']
    learning = pd.read_csv(DATA / 'exp356-exp277-frozen-contact-pdb50k-v2_learning.csv')
    selection = read_json('release_selection.json')
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), layout='constrained')
    labels = ['Full-backbone\nbidirectional', 'Frozen backbone\n+ contact encoder']
    for ax, metric, title in [(axes[0, 0], 'auroc', 'A  Full validation at step 2,000: AUROC ↑'),
                               (axes[0, 1], 'brier', 'B  Full validation at step 2,000: Brier ↓')]:
        values = [v1[metric]['mean'], v2[metric]['mean']]
        ax.bar(labels, values, color=[GRAY, TEAL], width=.6)
        ax.set(title=title, ylim=(0, max(values)*1.25))
        for i, value in enumerate(values):
            ax.text(i, value+max(values)*.035, f'{value:.4f}', ha='center', fontsize=10)
    ax = axes[1, 0]
    ax.plot(learning['step'], learning['contact_loss'], color=BLUE)
    ax.axvline(18000, color=TEAL, linestyle='--', linewidth=1)
    ax.set(xlabel='Optimizer step', ylabel='Mean contact BCE',
           title='C  Fixed validation sample during training')
    ax.text(.03, .04, '256 rollouts × 4 prefix lengths\nDiagnostic sample; not the full validation set',
            transform=ax.transAxes, fontsize=8)
    ax = axes[1, 1]
    steps = ['7,500', '18,000', '18,714']
    values = [x['objective'] for x in selection['candidates']]
    ax.bar(steps, values, color=[GRAY, TEAL, GRAY], width=.6)
    ax.set(xlabel='Checkpoint step', ylabel='BCE + recall MSE', ylim=(0, .55),
           title='D  Full validation selection objective ↓')
    for i, value in enumerate(values):
        ax.text(i, value+.012, f'{value:.4f}', ha='center', fontsize=10)
    save(fig, 'development', 'Figure 3. Development evidence. A–B: matched-step architecture comparison over all validation structures. '
         'C: fixed 256-rollout diagnostic sample, averaging prefixes 1, 4, 16 and full. '
         'D: final selection on all validation structures, averaging eight prefix-specific BCE + recall-MSE objectives. '
         'Step 18,000 was selected before the reserved test was evaluated.')


def quality(metrics: dict) -> None:
    """Show held-out calibration and errors across supplied rollout lengths."""
    calibration = pd.read_csv(DATA / 'final_test_calibration.csv')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), layout='constrained')
    ax = axes[0]
    ax.plot([0, 1], [0, 1], '--', color=GRAY, linewidth=1)
    for prefix, color in [('1', ORANGE), ('8', GRAY), ('64', BLUE), ('full', TEAL)]:
        rows = calibration[calibration['prefix'].astype(str) == prefix].sort_values('predicted')
        ax.plot(rows['predicted'], rows['observed'], 'o-', color=color, label=prefix, markersize=3)
    ax.set(xlabel='Mean predicted correctness', ylabel='Observed correct fraction',
           title='A  Contact calibration', xlim=(0, 1), ylim=(0, 1))
    ax.legend(title='Contacts supplied', frameon=False, fontsize=8, title_fontsize=8)
    for ax, metric, color, title in [
            (axes[1], 'precision_mae', BLUE, 'B  Precision estimation'),
            (axes[2], 'recall_mae', TEAL, 'C  Recall estimation')]:
        intervals(ax, [metrics[k][metric] for k in PREFIXES], color, metric, scale=100)
        ax.set(xticks=np.arange(8), xticklabels=PREFIXES, xlabel='Contacts supplied',
               ylabel='Mean absolute error (percentage points)', title=title, ylim=(0, None))
        ax.tick_params(axis='x', labelsize=9)
    save(fig, 'test_quality', 'Figure 4. Reserved-test quality at the selected step-18,000 checkpoint. '
         'Calibration pools contacts within probability bins. MAE first averages rollouts within each structure and then structures; '
         'bars are saved 95% intervals from 1,000 related-group bootstrap replicates. '
         'Longer fixed prefixes include only sufficiently long rollouts. Full is a separate variable-length condition.')


def context(metrics: dict) -> None:
    """Plot the paired effect of later evidence on the same first contact."""
    keys = PREFIXES[1:]
    records = [metrics['context_by_prefix'][k] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 3.8), layout='constrained')
    intervals(ax, records, TEAL, 'Paired first-contact improvement')
    ax.axhline(0, color=GRAY, linestyle='--', linewidth=1)
    ax.set(xticks=np.arange(len(keys)), xticklabels=keys, xlabel='Contacts supplied to reassess the first contact',
           ylabel='Reduction in first-contact Brier score', ylim=(-.0005, .0100),
           title='Later contacts improve judgments of the same earlier contact')
    for i, record in enumerate(records):
        ax.text(i, -.00028, f"n={record['proteins']:,}", ha='center', fontsize=8, color=GRAY)
    save(fig, 'context', 'Figure 5. Paired first-contact Brier reduction relative to supplying only the first contact. '
         'Positive values mean improvement. Each comparison uses the same eligible trajectories in both conditions; '
         'n denotes structures. Bars are 95% group-bootstrap intervals. '
         'Explicit fixed prefixes exclude the later end marker; full includes a genuine end marker when present.')


def utility(metrics: dict) -> None:
    """Compare contact filtering, rollout selection and multi-rollout aggregation."""
    full, agg = metrics['full'], metrics['aggregation']
    panels = [
        ('A  Keep half the contacts', ['First half\nin emission order', 'Highest-scored\nhalf'],
         [full['precision_emission_0.5'], full['precision_retained_0.5']], 'Contact precision (%)',
         full['precision_gain_half']),
        ('B  Choose one of up to eight rollouts', ['Random', 'Proofreader', 'Oracle'],
         [agg['random_rollout_precision'], agg['selected_rollout_precision'], agg['oracle_rollout_precision']],
         'Selected rollout precision (%)', agg['selection_precision_gain']),
        ('C  Aggregate the rollouts', ['Contact\nfrequency', 'Probability\nweighted'],
         [agg['frequency_r_precision'], agg['weighted_r_precision']], 'R-precision (%)', agg['aggregation_gain']),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), layout='constrained')
    for ax, (title, labels, records, ylabel, gain) in zip(axes, panels):
        means = np.array([x['mean'] for x in records])*100
        errors = np.array([[x['mean']-x['low'] for x in records],
                           [x['high']-x['mean'] for x in records]])*100
        ax.bar(labels, means, color=[GRAY, TEAL, '#a8c9d4'][:len(labels)], width=.58,
               yerr=errors, capsize=3, error_kw={'linewidth': 1})
        ax.set(title=title, ylabel=ylabel, ylim=(0, 78))
        ax.tick_params(axis='x', labelsize=9)
        for i, mean in enumerate(means):
            ax.text(i, 3, f'{mean:.1f}', ha='center', color='white', fontweight='bold')
        ax.text(.5, .96, f"Paired gain: +{100*gain['mean']:.2f} pp\n95% CI {100*gain['low']:.2f}–{100*gain['high']:.2f}",
                transform=ax.transAxes, ha='center', va='top', fontsize=9)
    save(fig, 'utility', 'Figure 6. Reserved-test downstream uses of the released scores. '
         'A compares equal contact budgets within each rollout; B selects a whole rollout by predicted precision, with an unattainable oracle for context; '
         'C ranks pooled distinct pairs by occurrence count or summed correctness probability. '
         'Bars show structure-mean scores and 95% group-bootstrap intervals. '
         'Paired gains and their intervals compare the proofreader with the first baseline in each panel; pp means percentage points.')


def main() -> None:
    """Build six figures without rerunning training or inference."""
    matplotlib.use('Agg')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'axes.titlesize': 10,
                         'axes.labelsize': 9, 'savefig.facecolor': 'white'})
    metrics = read_json('final_test_metrics.json')
    architecture()
    dataset()
    development()
    quality(metrics)
    context(metrics)
    utility(metrics)


if __name__ == '__main__':
    main()
