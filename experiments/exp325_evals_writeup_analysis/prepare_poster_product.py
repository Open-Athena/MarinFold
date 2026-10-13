"""Cache Neff contact panels with one three-way structural-contact ranking.

Uses the validated residue-confidence/pair archive from CONTACT_CONFIDENCE.md;
no model or ConFind rerun. Rendering and packaging consume only the resulting
small CSVs. Pair ranks are retained separately for exact point-to-contact lineage.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import sha256, summarize

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
CACHE = HERE / 'scratch/contact_confidence/scored'
OUTPUT = HERE / 'scratch/poster_product_analysis'
CONTACT_FIGURES = {
    '04_contacts_neff': ['r_precision', 'r_precision_long'],
    '04_contacts_pl5_neff': ['p_at_l5', 'p_at_l5_long'],
    '04_contacts_pl2_neff': ['p_at_l2', 'p_at_l2_long'],
}
RANKING = 'plddt_product_degree'


def ranked_pairs(pairs: pd.DataFrame, minimum: int) -> pd.DataFrame:
    """Rank emitted contacts without consulting their correctness labels."""
    eligible = pairs[(pairs.sep >= minimum) & (pairs.degree > 0)].copy()
    eligible['ranking_score'] = eligible.plddt_i * eligible.plddt_j * eligible.degree
    if not np.isfinite(eligible.ranking_score).all():
        raise ValueError('Nonfinite contact-ranking score')
    return eligible.sort_values(['ranking_score', 'i', 'j'], ascending=[False, True, True], kind='stable')


def main() -> None:
    """Score R, L/5 and L/2 from one frozen ordering and cache protein means."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(DATA / 'neff_figure_rows.csv', float_precision='round_trip')
    original['neff_figure_row'] = np.arange(len(original))
    sources = pd.read_csv(DATA / 'contact_confidence_sources.csv')
    previous = pd.read_csv(DATA / 'contact_confidence_metrics.csv', float_precision='round_trip')
    previous = previous[previous.ranking == RANKING].set_index(['stem', 'method', 'range'])
    # The L/5 tables retain truth counts and the full eligible-pair counts that
    # the original R-precision rows omit. Counts must agree across all methods.
    counts = original[original.figure == '04_contacts_pl5_neff'].copy()
    counts['range'] = np.where(counts.metric.str.endswith('_long'), 'long', 'all')
    for column in ['n_true', 'n_candidate', 'L']:
        if (counts.groupby(['stem', 'range'])[column].nunique() != 1).any():
            raise ValueError(f'Inconsistent frozen {column}')
    counts = counts.drop_duplicates(['stem', 'range']).set_index(['stem', 'range'])
    records, inputs, selections = [], [], []
    for index, source in enumerate(sources.itertuples()):
        directory = CACHE / source.method / source.stem
        result = json.loads((directory / 'result.json').read_text())
        path = directory / 'pairs.parquet'
        pin = sha256(path)
        if pin != result['files']['pairs.parquet'] or result['structure_sha256'] != source.structure_sha256:
            raise ValueError(f'Changed validated cache: {directory}')
        pairs = pd.read_parquet(path)
        if pairs.duplicated(['i', 'j']).any() or not pairs.i.lt(pairs.j).all():
            raise ValueError(f'Invalid canonical pairs: {directory}')
        inputs.append(dict(stem=source.stem, method=source.method, pair_file=str(path.relative_to(HERE)),
                           pair_sha256=pin, structure_sha256=source.structure_sha256))
        for region, minimum in [('all', 6), ('long', 24)]:
            count = counts.loc[(source.stem, region)]
            ranked = ranked_pairs(pairs, minimum)
            budgets = {'r_precision': int(count.n_true), 'p_at_l5': max(1, int(count.L)//5),
                       'p_at_l2': max(1, int(count.L)//2)}
            for metric, k in budgets.items():
                if k <= 0:
                    raise ValueError(f'Empty precision denominator: {source.stem}/{region}/{metric}')
                selected = ranked.head(k)
                correct = int(selected.true_contact.sum())
                precision = correct/k
                if metric == 'p_at_l5' and not np.isclose(precision, previous.loc[(source.stem, source.method, region), 'precision'], atol=1e-12, rtol=0):
                    raise ValueError(f'Prior three-way P@L/5 did not reproduce: {directory}/{region}')
                records.append(dict(stem=source.stem, method=source.method, range=region,
                                    metric=metric + ('_long' if region == 'long' else ''),
                                    ranking=RANKING, L=int(count.L), n_true=int(count.n_true),
                                    n_candidate=int(count.n_candidate), requested_n_top=k,
                                    n_selected=len(selected), n_correct=correct, value=precision,
                                    pair_input_row=index))
            selected = ranked.head(max(budgets.values())).copy()
            selected['stem'], selected['method'], selected['range'] = source.stem, source.method, region
            selected['rank'] = np.arange(1, len(selected)+1)
            selected['pair_input_row'] = index
            selections.append(selected)
        if (index+1) % 250 == 0 or index+1 == len(sources):
            print(f'{index+1}/{len(sources)} pair archives scored', flush=True)
    metrics = pd.DataFrame(records)
    if len(metrics) != 333*7*6 or metrics.duplicated(['stem', 'method', 'metric']).any():
        raise ValueError('Incomplete three-way metric table')
    metrics['poster_product_row'] = np.arange(len(metrics))
    metrics.to_csv(DATA / 'poster_product_metrics.csv', index=False)
    pd.DataFrame(inputs).to_csv(DATA / 'poster_product_inputs.csv', index=False)
    pd.concat(selections, ignore_index=True).to_parquet(OUTPUT / 'ranked_pairs.parquet', index=False)
    parts = []
    for figure in CONTACT_FIGURES:
        old = original[original.figure == figure].copy()
        control = old[old.method.isin(['marinfold', 'knn'])].copy()
        control['ranking'] = control.method.map({'marinfold': 'consensus_votes', 'knn': 'transferred_scores'})
        frame = old[~old.method.isin(['marinfold', 'knn'])].rename(columns={
            'value': 'original_value', 'n_top': 'original_n_top',
            'source': 'original_source', 'source_row': 'original_source_row'})
        frame = frame.drop(columns=['requested_n_top', 'n_true', 'n_candidate'])
        frame = frame.merge(metrics[['stem', 'method', 'metric', 'value', 'ranking', 'poster_product_row',
                                     'n_selected', 'n_correct', 'requested_n_top', 'n_true', 'n_candidate']],
                            on=['stem', 'method', 'metric'], validate='one_to_one')
        frame['n_top'] = frame.requested_n_top
        frame['source'] = 'data/poster_product_metrics.csv'
        frame['source_row'] = frame.poster_product_row
        parts.extend([frame, control])
    rows = pd.concat(parts, ignore_index=True)
    if len(rows) != 333*9*6 or rows.value.isna().any():
        raise ValueError('Incomplete matched poster panels')
    rows.to_csv(DATA / 'poster_product_figure_rows.csv', index=False)
    summary = summarize(rows)
    summary.to_csv(DATA / 'poster_product_summary.csv', index=False)
    files = ['poster_product_metrics.csv', 'poster_product_inputs.csv', 'poster_product_figure_rows.csv', 'poster_product_summary.csv']
    manifest = dict(files={name: sha256(DATA/name) for name in files},
                    inputs={name: sha256(DATA/name) for name in ['neff_figure_rows.csv', 'contact_confidence_sources.csv', 'contact_confidence_metrics.csv']},
                    preprocessors={Path(__file__).name: sha256(Path(__file__))},
                    ranked_pairs=dict(path=str((OUTPUT/'ranked_pairs.parquet').relative_to(HERE)), sha256=sha256(OUTPUT/'ranked_pairs.parquet')),
                    ranking='pLDDT_i * pLDDT_j * ConFind degree; positive emitted contacts only; stable input (i,j) ties',
                    confidence='Mean heavy-atom pLDDT per residue, normalized to [0,1], from the same selected archived structure',
                    eligibility='Frozen resolved pair universe; separation >=6 or >=24 before ranking; truth degree>=0.001',
                    denominator='R = true-contact count; L/5 and L/2 use floor(input L / divisor), minimum 1; shortfalls get zero credit',
                    controls='MarinFold 248B-token consensus and Sequence KNN transferred scores unchanged',
                    validation='All 4662 prior three-way P@L/5 values reproduced within 1e-12; 2331 validated pair archives verified by SHA256',
                    raw_pair_archive='https://huggingface.co/buckets/open-athena/MarinFold/resolve/data/exp325-writeup-analysis/contact-confidence-2026-10-13/contact_confidence_inputs_and_pairs.tar.gz')
    (DATA/'poster_product_analysis.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(summary[(summary.cohort=='eval-test') & (summary.tier=='All depths')][['metric','method','mean']].to_string(index=False))


if __name__ == '__main__':
    main()
