"""Select complete experimental chains and freeze homology-separated targets.

The source contacts are usable as complete references only when the serialized
resolved sequence equals the deposited canonical entity sequence and the source
contact list was not token-budget truncated. This intentionally excludes chains
with unresolved residues instead of treating their missing contacts as negatives.
"""

import argparse
import hashlib
import json
import random
import re
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gemmi
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
AA3 = 'ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split()
AA1 = 'ARNDCQEGHILKMFPSTWYV'
AA = dict(zip(AA3, AA1, strict=True))
AA['UNK'] = 'X'
PAIR = re.compile(r'<contact> <p(\d+)> <p(\d+)>')
RESIDUE = re.compile(r'<p(\d+)> <([A-Z]+)>')
MMSEQS = Path.home() / '.cache/marinfold/mmseqs/mmseqs/bin/mmseqs'


def seed(text: str) -> int:
    """Return a stable seed independent of Python's process hash randomization."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big')


def decode_document(row: dict) -> tuple[str, list[list[int]]]:
    """Recover original sequence indices from one complete monomer document."""
    before, after = row['document'].split('<begin_statements>')
    n = int(row['seq_len'])
    offset = int(row['n_term_index'])
    residues = {(int(p) - offset) % 2000: AA[a] for p, a in RESIDUE.findall(before)}
    if set(residues) != set(range(n)):
        raise ValueError(f"Invalid sequence positions: {row['entry_id']}")
    pairs = set()
    for a, b in PAIR.findall(after):
        i, j = sorted(((int(a) - offset) % 2000, (int(b) - offset) % 2000))
        if not 0 <= i < j < n or j - i < 6:
            raise ValueError(f"Invalid contact: {row['entry_id']} {(i, j)}")
        pairs.add((i, j))
    if len(pairs) != row['contacts_emitted'] or len(pairs) != row['contacts_passing_min_degree']:
        raise ValueError(f"Incomplete contacts: {row['entry_id']}")
    return ''.join(residues[i] for i in range(n)), [list(p) for p in sorted(pairs)]


def read_sequences(task: tuple[str, str]) -> tuple[str, dict[str, str]]:
    """Read experimental entity sequences; parse failures are fatal."""
    pdb, directory = task
    # The PDB mirror separates categories with '#' records. Stop after entity_poly
    # so a sequence lookup does not read megabytes of coordinates per entry.
    # Respect semicolon text fields: '#' inside a quoted sequence/title is data.
    lines = []
    inside = False
    multiline = False
    with (Path(directory) / f'{pdb}.cif').open() as handle:
        for line in handle:
            lines.append(line)
            if line.startswith(';'):
                multiline = not multiline
            if not multiline and line.startswith('_entity_poly.'):
                inside = True
            if inside and not multiline and line.startswith('#'):
                break
    block = gemmi.cif.read_string(''.join(lines)).sole_block()
    category = block.get_mmcif_category('_entity_poly.')
    return pdb, {
        entity: ''.join(sequence.split())
        for entity, sequence in zip(category['entity_id'], category['pdbx_seq_one_letter_code_can'], strict=True)
    }


def write_fasta(rows: list[dict], path: Path) -> None:
    """Write a stable target FASTA."""
    path.write_text(''.join(f">{r['entry_id']}\n{r['sequence']}\n" for r in rows))


def command(args: list[str], log: Path) -> None:
    """Run an audited external program, preserving its complete output."""
    print(' '.join(args), flush=True)
    with log.open('w') as output:
        subprocess.run(args, stdout=output, stderr=subprocess.STDOUT, check=True)


def homology_search(query: Path, target: Path, out: Path, threads: int) -> set[tuple[str, str]]:
    """Find >=30% identity hits covering >=50% of the shorter sequence."""
    if not out.exists():
        command([str(MMSEQS), 'easy-search', str(query), str(target), str(out), str(out) + '.tmp',
                 '--min-seq-id', '0.3', '-c', '0', '-s', '7.5', '--max-seqs', '10000',
                 '--threads', str(threads), '--format-output', 'query,target,fident,alnlen,qlen,tlen',
                 '--alignment-mode', '3'], out.with_suffix('.log'))
    hits = set()
    with out.open() as handle:
        for line in handle:
            q, t, ident, aligned, qlen, tlen = line.rstrip().split('\t')
            if float(ident) >= 0.3 and int(aligned) / min(int(qlen), int(tlen)) >= 0.5:
                hits.add((q, t))
    return hits


class Components:
    """Connect related chains by entry, source cluster and searched homology."""

    def __init__(self, ids: list[str]):
        self.parents = {i: i for i in ids}

    def find(self, value: str) -> str:
        parent = self.parents[value]
        if parent != value:
            self.parents[value] = self.find(parent)
        return self.parents[value]

    def union(self, a: str, b: str) -> None:
        a, b = sorted((self.find(a), self.find(b)))
        self.parents[b] = a


def main() -> None:
    """Build or resume a fully accounted target selection."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('/data/exp222_pdb_curation/docs/monomers'))
    parser.add_argument('--cif-dir', default='/data/tim/af3-db/mmcif_files')
    parser.add_argument('--out', type=Path, default=HERE / '_cache/prepared')
    parser.add_argument('--threads', type=int, default=24)
    parser.add_argument('--train-count', type=int, default=50000)
    parser.add_argument('--validation-count', type=int, default=1500)
    parser.add_argument('--test-count', type=int, default=1500)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    eligible_path = args.out / 'eligible.parquet'
    counts = Counter()
    if eligible_path.exists():
        rows = pq.read_table(eligible_path).to_pylist()
        counts.update(json.loads((args.out / 'eligibility_counts.json').read_text()))
    else:
        columns = ['entry_id', 'pdb_id', 'seq_len', 'n_term_index', 'document', 'entity_ids',
                   'cluster_ids', 'resolution', 'method', 'release_date', 'truncated',
                   'contacts_emitted', 'contacts_passing_min_degree', 'num_chains']
        # Keep multiple genuinely different structures, with at most one equivalent
        # ASU chain per entry/family. Family balancing happens after completeness QC.
        candidates = {}
        for shard in sorted(args.source.glob('*.parquet')):
            for row in pq.read_table(shard, columns=columns).to_pylist():
                counts['source_chains'] += 1
                if not 32 <= row['seq_len'] <= 1000:
                    counts['length_outside_32_1000'] += 1
                    continue
                if row['resolution'] is not None and row['resolution'] > 4.0:
                    counts['resolution_above_4'] += 1
                    continue
                if row['truncated'] or row['contacts_emitted'] != row['contacts_passing_min_degree']:
                    counts['truncated_reference'] += 1
                    continue
                if row['contacts_emitted'] < 5:
                    counts['fewer_than_5_contacts'] += 1
                    continue
                if row['num_chains'] != 1 or len(row['entity_ids']) != 1:
                    raise ValueError(f"Unexpected monomer schema {row['entry_id']}")
                sequence, contacts = decode_document(row)
                if 'X' in sequence:
                    counts['unknown_residues'] += 1
                    continue
                key = (row['pdb_id'], row['entity_ids'][0])
                if key in candidates:
                    counts['equivalent_asu_chain'] += 1
                    continue
                candidates[key] = dict(entry_id=row['entry_id'], pdb_id=row['pdb_id'],
                    entity_id=row['entity_ids'][0], cluster_id=row['cluster_ids'][0],
                    sequence=sequence, contacts=contacts, L=len(sequence),
                    resolution=row['resolution'], method=row['method'], release_date=row['release_date'])
            if counts['source_chains'] % 10000 < 2200:
                print(dict(counts), flush=True)
        pdb_ids = sorted({r['pdb_id'] for r in candidates.values()})
        print(f'Checking canonical completeness in {len(pdb_ids)} entries', flush=True)
        with ProcessPoolExecutor(args.threads) as pool:
            full = dict(pool.map(read_sequences, [(p, args.cif_dir) for p in pdb_ids], chunksize=64))
        rows = []
        for row in candidates.values():
            if full[row['pdb_id']][row['entity_id']] != row['sequence']:
                counts['incomplete_or_noncanonical_chain'] += 1
                continue
            rows.append(row)
        counts['complete_eligible'] = len(rows)
        pq.write_table(pa.Table.from_pylist(rows), eligible_path, compression='zstd')
        (args.out / 'eligibility_counts.json').write_text(json.dumps(counts, indent=2))
    # Round-robin across source families. Cap at eight structures per family,
    # favoring resolution, then a stable random tie-break over PDB identifiers.
    groups = defaultdict(list)
    for row in rows:
        key = f"cluster:{row['cluster_id']}" if row['cluster_id'] >= 0 else 'seq:' + row['sequence']
        groups[key].append(row)
    for members in groups.values():
        members.sort(key=lambda r: (r['resolution'] or 99, seed(r['entry_id'])))
    rows = [members[k] for k in range(8) for _, members in sorted(groups.items(), key=lambda item: seed(item[0])) if len(members) > k]
    counts['family_capped_candidates'] = len(rows)
    if len(rows) < args.train_count + args.validation_count + args.test_count:
        raise ValueError(f'Only {len(rows)} eligible candidates after family cap')
    candidates_fasta = args.out / 'candidates.fasta'
    write_fasta(rows, candidates_fasta)
    references = args.out / 'benchmark_queries.fasta'
    refs = [ROOT / 'experiments/exp225_data_decontaminate_training_corpora/data/reference' / name
            for name in ('eval_queries.fasta', 'foldbench_all_queries.fasta')]
    references.write_text('\n'.join(p.read_text() for p in refs))
    excluded = {target for _, target in homology_search(references, candidates_fasta, args.out / 'benchmark_hits.tsv', args.threads)}
    counts['benchmark_homologs_removed'] = len(excluded)
    rows = [r for r in rows if r['entry_id'] not in excluded]
    write_fasta(rows, args.out / 'clean.fasta')
    clusters = args.out / 'homology_cluster.tsv'
    if not clusters.exists():
        command([str(MMSEQS), 'easy-cluster', str(args.out / 'clean.fasta'), str(args.out / 'homology'),
            str(args.out / 'cluster-tmp'), '--min-seq-id', '0.3', '-c', '0.5', '--cov-mode', '0',
            '--cluster-mode', '1', '--threads', str(args.threads), '-s', '7.5'], args.out / 'clustering.log')
    components = Components([r['entry_id'] for r in rows])
    for line in clusters.read_text().splitlines():
        components.union(*line.split('\t'))
    known = {}
    for row in rows:
        for key in [('pdb', row['pdb_id']), ('sequence', row['sequence'])] + ([('source_cluster', row['cluster_id'])] if row['cluster_id'] >= 0 else []):
            if key in known:
                components.union(known[key], row['entry_id'])
            else:
                known[key] = row['entry_id']
    by_component = defaultdict(list)
    for row in rows:
        row['group_id'] = components.find(row['entry_id'])
        by_component[row['group_id']].append(row)
    held = {'validation': [], 'test': []}
    train_pool = []
    for group, members in sorted(by_component.items(), key=lambda item: seed('split:' + item[0])):
        split = 'train'
        for name, limit in [('validation', args.validation_count), ('test', args.test_count)]:
            if len(held[name]) < limit and len(members) <= 100:
                held[name].extend(members)
                split = name
                break
        if split == 'train':
            train_pool.extend(members)
        for row in members:
            row['split'] = split
    # Search explicitly across the final held-out boundary: clustering alone does
    # not prove that every cross-split pair falls below the homology threshold.
    write_fasta(held['validation'] + held['test'], args.out / 'heldout.fasta')
    write_fasta(train_pool, args.out / 'train_pool.fasta')
    leakage = homology_search(args.out / 'heldout.fasta', args.out / 'train_pool.fasta', args.out / 'cross_split_hits.tsv', args.threads)
    bad_groups = {r['group_id'] for r in train_pool if r['entry_id'] in {t for _, t in leakage}}
    train_pool = [r for r in train_pool if r['group_id'] not in bad_groups]
    counts['cross_split_groups_removed'] = len(bad_groups)
    # Validate validation/test separation too; move every connected conflicting
    # test family out of the held-out test, without moving it into training.
    write_fasta(held['validation'], args.out / 'validation.fasta')
    write_fasta(held['test'], args.out / 'test.fasta')
    val_test = homology_search(args.out / 'validation.fasta', args.out / 'test.fasta', args.out / 'validation_test_hits.tsv', args.threads)
    bad_test = {r['group_id'] for r in held['test'] if r['entry_id'] in {t for _, t in val_test}}
    held['test'] = [r for r in held['test'] if r['group_id'] not in bad_test]
    # Restore family round-robin ordering rather than sampling proportional to family size.
    train_ids = {r['entry_id'] for r in train_pool}
    train = [r for r in rows if r['entry_id'] in train_ids][:args.train_count]
    if len(train) != args.train_count:
        raise ValueError(f'Only {len(train)} clean training structures remain')
    selected = train + held['validation'] + held['test']
    pq.write_table(pa.Table.from_pylist(selected), args.out / 'targets.parquet', compression='zstd')
    for split in ('train', 'validation', 'test'):
        subset = [r for r in selected if r['split'] == split]
        counts[f'{split}_structures'] = len(subset)
        counts[f'{split}_groups'] = len({r['group_id'] for r in subset})
    counts['mmseqs_version'] = subprocess.check_output([str(MMSEQS), 'version'], text=True).strip()
    (args.out / 'manifest.json').write_text(json.dumps(counts, indent=2))
    data = HERE / 'data'
    data.mkdir(exist_ok=True)
    (data / 'dataset_manifest.json').write_text(json.dumps(counts, indent=2))
    pd.DataFrame([{k: v for k, v in r.items() if k not in {'sequence', 'contacts'}} for r in selected]).to_csv(data / 'structure_manifest.csv.gz', index=False)
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == '__main__':
    main()
