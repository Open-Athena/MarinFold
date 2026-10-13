"""Cache L/2 sampling accuracy from the original 100-rollout pools.

The oracle is reselected at P@L/2. Ground truth does not select consensus
contacts. Raw pool hashes and every prior sampling R-precision must reproduce;
all-range and long-range are filtered before selecting the L/2 contacts.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import Sources, annotate, bootstrap, cohort_slices, sha256, summarize, TIERS
from prepare_neff import regroup
from prepare_pl5 import sampling_sources

HERE = Path(__file__).resolve().parent
DATA = HERE/'data'


def main() -> None:
    """Save point-level values, fixed cutoffs, oracle selectors and bootstrap CIs."""
    sources = Sources()
    targets = sources.csv(DATA/'targets.csv', round_trip=True).drop(columns=['source','source_row'])
    _, raw = sampling_sources(targets, divisor=2)
    previous = json.loads((DATA/'pl5_analysis.json').read_text())['raw_sources']
    if {key: row['sha256'] for key, row in raw.items()} != {key: row['sha256'] for key, row in previous.items()}:
        raise ValueError('L/2 sampling inputs differ from the published L/5 pools')
    # Check every individual validity flag and R score as well as the aggregates
    # checked by sampling_sources; this protects the original invalid-map rules.
    individuals = pd.read_csv(DATA/'pl2_sampling_individual.csv.gz', float_precision='round_trip')
    original = pd.read_csv(DATA/'pl5_sampling_individual.csv.gz', float_precision='round_trip')
    keys = ['stem','range','rollout']
    checked = individuals.merge(original, on=keys, suffixes=('_new','_old'), validate='one_to_one')
    if len(checked) != 314*100*2:
        raise ValueError('Incomplete saved sampling pool')
    for column in ['valid','L','n_emitted','raw_source','raw_sha256','r_precision']:
        if not checked[f'{column}_new'].equals(checked[f'{column}_old']):
            raise ValueError(f'Changed sampling identity or scoring: {column}')
    per_protein = sources.csv(DATA/'pl2_sampling_per_protein.csv', round_trip=True)
    neff = sources.csv(DATA/'neff_targets.csv', round_trip=True).drop(columns=['source','source_row'])
    parts=[]
    for method in ['single','consensus','best100']:
        part = per_protein[['stem','range','n_top',method,'source','source_row']].rename(columns={method:'value'})
        part['metric'] = part['range'].map({'all':'p_at_l2','long':'p_at_l2_long'})
        part = annotate(part.assign(method=method,source_column=method,figure='06_sampling_pl2'),targets)
        parts.append(regroup(part,neff))
    rows = pd.concat(parts,ignore_index=True)
    if rows.duplicated(['stem','method','metric']).any() or len(rows)!=314*3*2 or rows.value.isna().any():
        raise ValueError('Incomplete L/2 sampling figure rows')
    if not (rows.n_top == np.maximum(1,rows.L//2)).all():
        raise ValueError('Sampling budget differs from input L/2')
    rows.to_csv(DATA/'pl2_sampling_figure_rows.csv',index=False)
    summary=summarize(rows)
    summary.to_csv(DATA/'pl2_sampling_summary.csv',index=False)
    deltas=[]
    for metric, group in rows.groupby('metric'):
        for cohort, subset in cohort_slices(group).items():
            for tier in ['All depths',*TIERS]:
                part = subset if tier=='All depths' else subset[subset.tier==tier]
                if part.empty:
                    continue
                wide=part.pivot(index='stem',columns='method',values='value').sort_index()
                mean,lo,hi=bootstrap((wide.best100-wide.consensus).to_numpy())
                deltas.append(dict(metric=metric,cohort=cohort,tier=tier,n=len(wide),
                    oracle_minus_consensus=mean,ci_low=lo,ci_high=hi,
                    oracle_better=int((wide.best100>wide.consensus+1e-12).sum()),
                    consensus_better=int((wide.consensus>wide.best100+1e-12).sum()),
                    ties=int(np.isclose(wide.best100,wide.consensus,atol=1e-12,rtol=0).sum()),stems='|'.join(wide.index)))
    pd.DataFrame(deltas).to_csv(DATA/'pl2_sampling_deltas.csv',index=False)
    names=['pl2_sampling_individual.csv.gz','pl2_sampling_per_protein.csv','pl2_sampling_figure_rows.csv',
           'pl2_sampling_summary.csv','pl2_sampling_deltas.csv']
    manifest=dict(files={name:sha256(DATA/name) for name in names},sources=sources.records,raw_sources=raw,
        preprocessors={name:sha256(HERE/name) for name in ['prepare_sampling_pl2.py','prepare_pl5.py','generation/score_contacts.py']},
        checkpoint='contacts-v1-exp277-m2-p06-full-epoch-1.5B-step-266344 (248.584B raw tokens)',
        cohort='Same 314 natural proteins: 97 eval-val + 217 eval-test; first 100 saved iid samples per protein',
        cutoff='k=max(1,floor(input L/2)); all natural rows retain the full denominator',
        ranking='Individual: distinct emitted contacts in emission order, filtered before cutoff; missing ranks get zero credit. Invalid/unfinished individual maps score zero. Consensus: stable vote ranking over all parsed maps, matching the original sampling diagnostic.',
        oracle='Highest individual P@L/2 within each protein and range, reselected from all 100 samples; ties choose earliest rollout index. Uses ground truth.',
        contact_definition='Frozen resolved pair universe; ConFind truth degree>=0.001; separation>=6 (all) or >=24 (long)',
        depth='Neff80 for interactive scatter colors and cached subgroup summaries; overall scores do not depend on depth grouping',
        uncertainty='Equal protein means and 95% intervals from 5000 protein bootstraps, seed325',
        validation='All 314 raw pool hashes and truth hash match L/5; all 62800 individual validity/R-score rows and 1884 aggregated R-score rows reproduce')
    (DATA/'pl2_sampling_analysis.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(summary[(summary.cohort=='natural')&(summary.tier=='All depths')][['metric','method','n','mean','ci_low','ci_high']].to_string(index=False))
    print(pd.DataFrame(deltas).query("cohort=='natural' and tier=='All depths'").to_string(index=False))


if __name__=='__main__':
    main()
