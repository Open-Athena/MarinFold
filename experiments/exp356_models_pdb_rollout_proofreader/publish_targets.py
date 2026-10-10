"""Audit and freeze the small experimental target manifest in CoreWeave storage."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from storage import ROOT, filesystem, write_json

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Publish immutable inputs only after checking labels and split identities."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared',type=Path,default=HERE/'_cache/prepared-v2')
    parser.add_argument('--out',default=ROOT+'/data/v1')
    args = parser.parse_args()
    target = args.prepared/'targets.parquet'
    rows = pq.read_table(target).to_pylist()
    splits = {s:[r for r in rows if r['split']==s] for s in ['train','validation','test']}
    if len(splits['train']) != 50000 or len({r['entry_id'] for r in rows}) != len(rows):
        raise ValueError('Expected 50,000 unique training structures and distinct target IDs')
    for left,left_rows in splits.items():
        for right,right_rows in splits.items():
            if left >= right:
                continue
            for key in ['pdb_id','group_id','sequence']:
                if {r[key] for r in left_rows}&{r[key] for r in right_rows}:
                    raise ValueError(f'Overlapping {key}: {left}, {right}')
    for row in rows:
        if len(row['sequence']) != row['L'] or row['L'] != row['canonical_end']-row['canonical_start']:
            raise ValueError(f"Sequence mapping mismatch: {row['entry_id']}")
        if len({tuple(p) for p in row['contacts']}) != len(row['contacts']) or not all(
            0 <= a < b < row['L'] and b-a >= 6 for a,b in row['contacts']):
            raise ValueError(f"Invalid reference contacts: {row['entry_id']}")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = dict(target_sha256=digest,target_bytes=target.stat().st_size,
        counts=json.loads((args.prepared/'manifest.json').read_text()),
        eligibility=json.loads((args.prepared/'eligibility_config.json').read_text()),
        reference=dict(source_experiment='exp222_data_pdb_curation_monomers_multimers',
            definition='contacts-v1 pyconfind native-only',contact_distance_angstrom=3.,
            dcut_angstrom=25.,clash_distance_angstrom=2.,minimum_contact_degree=.001,minimum_sequence_separation=6,
            unresolved_residues='not supplied to the predictor; only bounded contiguous terminal cropping allowed'),
        homology=dict(minimum_identity=.3,minimum_shorter_sequence_coverage=.5,
            benchmark_queries_sha256=hashlib.sha256((args.prepared/'benchmark_queries.fasta').read_bytes()).hexdigest()))
    fs,root = filesystem(args.out)
    if fs.exists(root+'/targets.json'):
        if json.loads(fs.cat(root+'/targets.json')) != manifest:
            raise ValueError('Refusing to overwrite a different frozen target dataset')
        print('Target dataset already committed')
        return
    fs.put_file(str(target),root+'/targets.parquet')
    write_json(manifest,args.out+'/targets.json')
    (HERE/'data/targets_provenance.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(out=args.out,**manifest['counts']),indent=2))


if __name__ == '__main__':
    main()
