"""Consolidate held-out predictions into protein- and family-aware quality reports."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from build_summary import save_plot_with_meta
from records import stable_seed
from storage import stage_directory

HERE = Path(__file__).resolve().parent


def interval(frame: pd.DataFrame, column: str) -> dict:
    """Report a protein mean with a bootstrap that resamples whole related groups."""
    protein = frame.dropna(subset=[column]).groupby(['group_id','entry_id'],as_index=False)[column].mean()
    groups = protein.groupby('group_id')[column].agg(['sum','count'])
    rng = np.random.default_rng(356)
    values = []
    for _ in range(1000):
        sample = groups.iloc[rng.integers(0,len(groups),len(groups))]
        values.append(sample['sum'].sum()/sample['count'].sum())
    return dict(mean=float(protein[column].mean()),low=float(np.quantile(values,.025)),
        high=float(np.quantile(values,.975)),proteins=len(protein),groups=len(groups))


def read_pattern(directory: Path, pattern: str) -> pd.DataFrame:
    """Combine completed rank-local parquet reports."""
    return pd.concat([pq.read_table(p).to_pandas() for p in sorted(directory.glob(pattern))],ignore_index=True)


def aggregate_contacts(rows: list[dict]) -> list[dict]:
    """Compare frequency aggregation with probability-weighted frequency at true R."""
    proteins = defaultdict(list)
    for row in rows:
        proteins[row['entry_id']].append(row)
    results = []
    for entry_id, rollouts in proteins.items():
        candidates = {}
        precision, recall, estimated_precision, estimated_recall = [], [], [], []
        for row in rollouts:
            labels = np.asarray(row['labels'])
            probabilities = np.asarray(row['probabilities'])
            precision.append(labels.mean())
            recall.append(labels.sum()/row['gt_count'])
            estimated_precision.append(probabilities.mean())
            estimated_recall.append(row['predicted_recall'])
            for pair, truth, probability in zip(row['pairs'],labels,probabilities,strict=True):
                pair = tuple(pair)
                if pair not in candidates:
                    candidates[pair] = [0.,0.,float(truth)]
                if candidates[pair][2] != truth:
                    raise ValueError(f'Inconsistent reference labels for {entry_id}, {pair}')
                candidates[pair][0] += 1
                candidates[pair][1] += probability
        reference_count = rollouts[0]['gt_count']
        ordered = sorted(candidates,key=lambda pair:stable_seed(f'{entry_id}:{pair}'))
        measured = {}
        for field, index in [('frequency_r_precision',0),('weighted_r_precision',1)]:
            ranking = sorted(ordered,key=lambda pair:-candidates[pair][index])[:reference_count]
            measured[field] = sum(candidates[p][2] for p in ranking)/reference_count
        p,r,ep,er = map(np.asarray,(precision,recall,estimated_precision,estimated_recall))
        f1 = 2*p*r/np.maximum(p+r,1e-12)
        estimated_f1 = 2*ep*er/np.maximum(ep+er,1e-12)
        results.append(dict(entry_id=entry_id,group_id=rollouts[0]['group_id'],
            random_rollout_precision=float(p.mean()),selected_rollout_precision=float(p[np.argmax(ep)]),
            oracle_rollout_precision=float(p.max()),random_rollout_f1=float(f1.mean()),
            selected_rollout_f1=float(f1[np.argmax(estimated_f1)]),oracle_rollout_f1=float(f1.max()),**measured))
    return results


def main() -> None:
    """Download one bounded evaluation report and produce reproducible tables/plots."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--label',required=True)
    args = parser.parse_args()
    local = HERE/'_cache/evaluation'/args.label
    stage_directory(args.source,local)
    markers = [json.loads(p.read_text()) for p in sorted(local.glob('rank-*.complete.json'))]
    if not markers or {m['rank'] for m in markers} != set(range(markers[0]['world'])):
        raise ValueError('Evaluation is missing completed ranks')
    frame = read_pattern(local,'per_rollout-*.parquet')
    if frame.duplicated(['identity','prefix']).any():
        raise ValueError('Duplicate evaluation examples')
    frame['precision_mae'] = (frame.predicted_precision-frame.precision).abs()
    frame['recall_mae'] = (frame.predicted_recall-frame.recall).abs()
    frame['precision_gain_half'] = frame['precision_retained_0.5']-frame['precision_emission_0.5']
    summary = {}
    for prefix,part in frame.groupby('prefix'):
        summary[prefix] = {key:interval(part,key) for key in (
            'precision','recall','precision_mae','recall_mae','brier','bce','auroc',
            'precision_retained_0.5','precision_emission_0.5','precision_gain_half') if part[key].notna().any()}
    first = frame[frame.prefix=='1'].set_index('identity')
    full = frame[frame.prefix=='full'].set_index('identity')
    paired = first[['entry_id','group_id','first_contact_label','first_contact_probability']].join(
        full[['first_contact_probability']],rsuffix='_full',how='inner')
    paired['brier_gain_from_context'] = (paired.first_contact_probability-paired.first_contact_label)**2-(
        paired.first_contact_probability_full-paired.first_contact_label)**2
    summary['context'] = interval(paired,'brier_gain_from_context')
    contacts = []
    for path in sorted(local.glob('contact_scores-*.parquet')):
        contacts.extend(pq.read_table(path).to_pylist())
    aggregated = pd.DataFrame(aggregate_contacts(contacts))
    aggregated['aggregation_gain'] = aggregated.weighted_r_precision-aggregated.frequency_r_precision
    aggregated['selection_precision_gain'] = aggregated.selected_rollout_precision-aggregated.random_rollout_precision
    aggregated['selection_f1_gain'] = aggregated.selected_rollout_f1-aggregated.random_rollout_f1
    summary['aggregation'] = {key:interval(aggregated,key) for key in aggregated.columns if key not in {'entry_id','group_id'}}
    calibration = read_pattern(local,'calibration-*.parquet').groupby(['prefix','bin'],as_index=False)[
        ['count','sum_probability','sum_truth']].sum()
    calibration['predicted'] = calibration.sum_probability/calibration['count']
    calibration['observed'] = calibration.sum_truth/calibration['count']
    data = HERE/'data'
    data.mkdir(exist_ok=True)
    (data/f'{args.label}_metrics.json').write_text(json.dumps(summary,indent=2))
    frame.to_csv(data/f'{args.label}_per_rollout.csv.gz',index=False)
    aggregated.to_csv(data/f'{args.label}_aggregation.csv',index=False)
    calibration.to_csv(data/f'{args.label}_calibration.csv',index=False)
    read_pattern(local,'timings-*.parquet').to_csv(data/f'{args.label}_timings.csv',index=False)
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    for prefix in ['1','8','64','full']:
        part = calibration[calibration.prefix==prefix]
        axes[0].plot(part.predicted,part.observed,'o-',label=prefix)
    axes[0].plot([0,1],[0,1],'k--',alpha=.4)
    axes[0].set(xlabel='Predicted correctness',ylabel='Observed correctness',title='Contact calibration')
    axes[0].legend(title='Contacts supplied')
    prefixes = [p for p in ['1','2','4','8','16','32','64','full'] if p in summary]
    for key,label in [('precision_mae','Precision'),('recall_mae','Recall')]:
        axes[1].plot(prefixes,[summary[p][key]['mean'] for p in prefixes],'o-',label=label)
    axes[1].set(xlabel='Contacts supplied',ylabel='Mean absolute error',title='Rollout quality estimates')
    axes[1].legend()
    measures = ['frequency_r_precision','weighted_r_precision']
    axes[2].bar(['Frequency','Proofreader weighted'],[summary['aggregation'][m]['mean'] for m in measures])
    axes[2].set(ylabel='R-precision',title='Aggregation across eight rollouts',ylim=(0,1))
    fig.tight_layout()
    save_plot_with_meta(fig,HERE/'plots'/f'{args.label}_quality.png',
        caption='Held-out protein means. Calibration bins pool contacts; confidence intervals in metrics JSON resample related groups.',dpi=150)
    print(json.dumps({key:value for key,value in summary.items() if key in {'1','full','context','aggregation'}},indent=2))


if __name__ == '__main__':
    main()
