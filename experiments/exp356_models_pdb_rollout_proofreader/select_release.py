"""Freeze a validation-only checkpoint choice before the reserved test is evaluated."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PREFIXES = ['1','2','4','8','16','32','64','full']


def candidate_objectives(frame: pd.DataFrame) -> dict[str, float]:
    """Give proteins and requested prefix lengths equal weight in model selection."""
    measured = frame.assign(objective=frame.bce+(frame.predicted_recall-frame.recall)**2)
    if not np.isfinite(measured.objective).all():
        raise ValueError('Nonfinite validation loss')
    per_prefix = measured.groupby(['prefix','entry_id']).objective.mean().groupby('prefix').mean()
    if set(per_prefix.index)!=set(PREFIXES):
        raise ValueError('Incomplete prefix coverage')
    return per_prefix.loc[PREFIXES].to_dict()


def main() -> None:
    """Compare frozen candidates with the training loss averaged over prefix lengths."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', nargs='+', required=True)
    args = parser.parse_args()
    data = HERE/'data'
    if (data/'release_selection.json').exists():
        raise FileExistsError('The release checkpoint choice is already frozen')
    for path in data.glob('*_evaluation.json'):
        report = json.loads(path.read_text())
        if report['split']=='test' and report['proteins']==1473:
            raise ValueError('Cannot select a checkpoint after examining the reserved test')
    candidates = []
    fingerprints = set()
    for label in args.labels:
        report = json.loads((data/f'{label}_evaluation.json').read_text())
        if report['split']!='validation' or report['proteins']!=1552:
            raise ValueError('Selection requires the complete validation split')
        if any(rank['prefix_end_policy']!='contact_boundary_except_full_rollout' for rank in report['ranks']):
            raise ValueError('Selection requires explicit prefixes without future end markers')
        frame = pd.read_csv(data/f'{label}_per_rollout.csv.gz')
        per_prefix = candidate_objectives(frame)
        candidates.append(dict(label=label,checkpoint=report['checkpoint'],
            objective=float(np.mean(list(per_prefix.values()))),per_prefix_objective=per_prefix))
        fingerprints.add(report['data_fingerprint'])
    if len(fingerprints)!=1:
        raise ValueError('Candidate evaluation corpora differ')
    selected = min(candidates,key=lambda item:(item['objective'],item['checkpoint']))
    result = dict(selected=selected,candidates=candidates,data_fingerprint=next(iter(fingerprints)),
        criterion='Mean over eight prefix lengths of protein-mean (contact BCE + recall squared error)',
        reserved_test_evaluated_at_selection=False,selected_at=datetime.now(UTC).isoformat())
    (data/'release_selection.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
