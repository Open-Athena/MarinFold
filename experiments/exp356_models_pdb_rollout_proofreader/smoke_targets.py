"""Select a small independently auditable experimental-chain GPU smoke corpus."""

import json
from pathlib import Path

import pyarrow.parquet as pq

from prepare import decode_document, read_sequences
from storage import ROOT, write_rows


def main() -> None:
    """Cover short and long chains while keeping smoke families disjoint."""
    source = Path('/data/exp222_pdb_curation/docs/monomers')
    candidates, seen = [], set()
    for shard in sorted(source.glob('*.parquet'))[:8]:
        for row in pq.read_table(shard).to_pylist():
            if not 32 <= row['seq_len'] <= 1000 or row['truncated'] or row['contacts_emitted'] < 5:
                continue
            if row['contacts_emitted'] != row['contacts_passing_min_degree']:
                continue
            group = row['cluster_ids'][0]
            if group < 0 or group in seen:
                continue
            sequence, contacts = decode_document(row)
            _, entities = read_sequences((row['pdb_id'], '/data/tim/af3-db/mmcif_files'))
            if entities[row['entity_ids'][0]] != sequence or 'X' in sequence:
                continue
            seen.add(group)
            candidates.append(dict(entry_id=row['entry_id'], sequence=sequence, L=len(sequence),
                contacts=contacts, group_id=f'smoke:{group}', split='train'))
            if len(candidates) >= 300:
                break
        if len(candidates) >= 300:
            break
    candidates.sort(key=lambda r:r['L'])
    selected = [candidates[round(i*(len(candidates)-1)/47)] for i in range(48)]
    for i, row in enumerate(selected):
        if i % 6 == 0:
            row['split'] = 'validation'
        elif i % 6 == 1:
            row['split'] = 'test'
    write_rows(selected, ROOT + '/data/smoke-v1/targets.parquet')
    print(json.dumps(dict(count=len(selected), min_length=selected[0]['L'], max_length=selected[-1]['L'])))


if __name__ == '__main__':
    main()
