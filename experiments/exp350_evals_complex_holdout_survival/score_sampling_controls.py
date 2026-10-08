"""Compare saved interface maps with amino-acid and residue-degree matched nulls.

The amino-acid null is an exact stratified hypergeometric simulation. Each
chain-A/chain-B residue-type stratum retains its predicted contact count.
The degree null uses a symmetric bipartite edge-switch chain with a uniform
stationary distribution over simple graphs with the same per-residue degrees.
We report it as approximate and separately run longer-chain diagnostics.
Neither control fits parameters to experimental contacts: truth is used only
to score randomized maps and apply the same oracle selection as MarinFold.
"""

import argparse
import csv
import ctypes
import hashlib
import json
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from score_rollout_sampling import BUDGETS, group_interval, maximum_weights, write_csv

HERE = Path(__file__).resolve().parent
PUBLIC = ('hf://buckets/open-athena/MarinFold/data/evals/exp350_foldbench_pair_holdout/'
          'contact_eval_v1/sampling_controls_v1')


def compile_switch_kernel() -> ctypes.CDLL:
    """Compile the small CPU kernel into a source-hashed temporary library."""
    source = HERE / 'degree_switch.cpp'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    library = Path(tempfile.gettempdir()) / f'exp350-degree-switch-{digest}.so'
    if not library.exists():
        subprocess.run(['c++', '-O3', '-std=c++17', '-shared', '-fPIC', '-fopenmp',
                        str(source), '-o', str(library)], check=True)
    kernel = ctypes.CDLL(str(library))
    array = np.ctypeslib.ndpointer
    kernel.sample_degree_null.argtypes = [
        array(dtype=np.int32, flags='C_CONTIGUOUS'),
        array(dtype=np.int32, flags='C_CONTIGUOUS'),
        array(dtype=np.int64, flags='C_CONTIGUOUS'),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        array(dtype=np.uint8, flags='C_CONTIGUOUS'),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_int,
        array(dtype=np.int32, flags='C_CONTIGUOUS'),
        array(dtype=np.float64, flags='C_CONTIGUOUS'),
        array(dtype=np.float64, flags='C_CONTIGUOUS'),
        array(dtype=np.int32, flags='C_CONTIGUOUS'),
    ]
    kernel.sample_degree_null.restype = None
    return kernel


def degree_draws(maps: list[np.ndarray], truth: np.ndarray, draws: int,
                 burn: int, thin: int, seed: int, threads: int) -> tuple[np.ndarray, list[dict]]:
    """Draw fixed-degree maps and validate the final graph of every chain."""
    kernel = compile_switch_kernel()
    sizes = np.array([len(edges) for edges in maps])
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    edges = np.concatenate(maps, axis=0).astype(np.int32)
    left = np.ascontiguousarray(edges[:, 0])
    right = np.ascontiguousarray(edges[:, 1])
    tp = np.zeros((draws, len(maps)), dtype=np.int32)
    acceptance = np.zeros(len(maps))
    overlap = np.zeros(len(maps))
    final_right = np.empty_like(right)
    kernel.sample_degree_null(left, right, offsets, len(maps), *truth.shape,
                              np.ascontiguousarray(truth, dtype=np.uint8), draws,
                              burn, thin, seed, threads, tp, acceptance, overlap, final_right)
    diagnostics = []
    for i, initial in enumerate(maps):
        start, stop = offsets[i:i + 2]
        final = np.column_stack([left[start:stop], final_right[start:stop]])
        if len(np.unique(final, axis=0)) != len(initial):
            raise ValueError(f'Degree shuffle created duplicate edges in map {i}')
        for side in (0, 1):
            if Counter(initial[:, side]) != Counter(final[:, side]):
                raise ValueError(f'Degree shuffle changed residue degrees in map {i}')
        if tp[-1, i] != sum(truth[a, b] for a, b in final):
            raise ValueError(f'Degree shuffle true-positive count mismatch in map {i}')
        if len(initial) and acceptance[i] == 0:
            pairs = {tuple(pair) for pair in initial}
            for u, x in initial:
                for v, z in initial:
                    if u != v and x != z and (u, z) not in pairs and (v, x) not in pairs:
                        raise ValueError(f'Degree chain made no moves despite a valid switch in map {i}')
        values = tp[:, i].astype(float)
        variance = np.var(values)
        lag_one = (float(np.mean((values[:-1] - values.mean()) * (values[1:] - values.mean()))
                   / variance) if variance else 0.0)
        diagnostics.append({'rollout': i, 'n_predicted': len(initial),
                            'acceptance_rate': acceptance[i], 'mean_original_edge_fraction': overlap[i],
                            'no_valid_switch': bool(len(initial) and acceptance[i] == 0),
                            'tp_lag1_autocorrelation': lag_one,
                            'mean_tp': float(values.mean())})
    return tp, diagnostics


def amino_acid_draws(maps: list[np.ndarray], truth: np.ndarray,
                     left_types: np.ndarray, right_types: np.ndarray,
                     draws: int, seed: int) -> np.ndarray:
    """Simulate independent exact AA-pair-stratified contact counts."""
    n_types = max(int(left_types.max()), int(right_types.max())) + 1
    strata = left_types[:, None] * n_types + right_types[None, :]
    population = np.bincount(strata.ravel(), minlength=n_types**2)
    positives = np.bincount(strata[truth.astype(bool)], minlength=n_types**2)
    counts = np.zeros((len(maps), n_types**2), dtype=np.int64)
    for i, edges in enumerate(maps):
        counts[i] = np.bincount(strata[edges[:, 0], edges[:, 1]], minlength=n_types**2)
    if np.any(counts > population):
        raise ValueError('Predicted stratum count exceeds eligible pair population')
    rng = np.random.default_rng(seed)
    tp = np.zeros((draws, len(maps)), dtype=np.int32)
    for stratum in np.flatnonzero(positives):
        tp += rng.hypergeometric(positives[stratum], population[stratum] - positives[stratum],
                                 counts[:, stratum], size=tp.shape).astype(np.int32)
    return tp


def metric_samples(tp: np.ndarray, sizes: np.ndarray, n_true: int,
                   universe: int, complete: np.ndarray) -> dict[str, np.ndarray]:
    """Score null samples with the same F1 and binary-tie R-precision rules."""
    f1 = 2 * tp / (n_true + sizes)
    rp = np.zeros_like(f1)
    dense = sizes >= n_true
    rp[:, dense] = tp[:, dense] / sizes[dense]
    sparse = ~dense
    rp[:, sparse] = (tp[:, sparse] + (n_true - sizes[sparse])
                    * (n_true - tp[:, sparse]) / (universe - sizes[sparse])) / n_true
    f1[:, ~complete] = 0
    rp[:, ~complete] = 0
    return {'f1': f1, 'r_precision': rp}


def summarize_draws(target: dict, arm: str, scores: dict[str, np.ndarray],
                    observed: dict[str, np.ndarray]) -> list[dict]:
    """Average oracle order statistics, retaining replicate-level estimates."""
    rows = []
    for metric, values in scores.items():
        ranked = np.sort(values, axis=1)
        for k in BUDGETS:
            weights = maximum_weights(values.shape[1], k)
            maxima = ranked @ weights
            actual = float(np.sort(observed[metric]) @ weights)
            # Consecutive blocks allow serial correlation between MCMC draws;
            # the block SE is a simulation diagnostic, not biological uncertainty.
            blocks = np.array_split(maxima, 20)
            block_means = np.array([block.mean() for block in blocks])
            estimate = float(maxima.mean())
            rows.append({'target_id': target['target_id'], 'split': target['split'],
                         'group_id': target['group_id'], 'arm': arm, 'metric': metric, 'k': k,
                         'model_expected_best': actual, 'null_expected_best': estimate,
                         'model_minus_null': actual - estimate,
                         'null_mc_block_se': float(block_means.std(ddof=1) / np.sqrt(len(blocks))),
                         'n_draws': values.shape[0]})
    return rows


def load_maps(target: dict, samples: list[dict]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert resolved canonical contacts to a compact bipartite universe."""
    if [row['rollout'] for row in samples] != list(range(1000)):
        raise ValueError(f"{target['stem']}: expected 1000 ordered attempts")
    left, right = target['resolved_positions_by_chain']
    left_index = {p: i for i, p in enumerate(left)}
    right_index = {p: i for i, p in enumerate(right)}
    sequence = ''.join(target['chain_sequences'])
    alphabet = {aa: i for i, aa in enumerate(sorted(set(sequence)))}
    left_types = np.array([alphabet[sequence[p]] for p in left])
    right_types = np.array([alphabet[sequence[p]] for p in right])
    truth = np.zeros((len(left), len(right)), dtype=np.uint8)
    for a, b in target['gt_contacts']:
        truth[left_index[a], right_index[b]] = 1
    complete = np.array([row['finish_reason'] == 'stop' for row in samples])
    maps = []
    for row, stopped in zip(samples, complete, strict=True):
        edges = sorted({(left_index[a], right_index[b]) for a, b in row['contacts']
                        if stopped and a in left_index and b in right_index})
        maps.append(np.array(edges, dtype=np.int32).reshape(-1, 2))
    return maps, truth, left_types, right_types, complete


def run_target(target: dict, samples: list[dict], args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    """Score both stronger nulls and a longer-chain sensitivity run."""
    start = time.monotonic()
    maps, truth, left_types, right_types, complete = load_maps(target, samples)
    sizes = np.array([len(edges) for edges in maps])
    actual_tp = np.array([[sum(int(truth[a, b]) for a, b in edges) for edges in maps]])
    n_true, universe = int(truth.sum()), truth.size
    observed = {key: values[0] for key, values in metric_samples(
        actual_tp, sizes, n_true, universe, complete).items()}
    target_seed = int.from_bytes(hashlib.sha256(target['stem'].encode()).digest()[:4], 'little') + args.seed
    results, diagnostics = [], []
    aa_tp = amino_acid_draws(maps, truth, left_types, right_types, args.draws, target_seed)
    results.extend(summarize_draws(target, 'amino_acid_pair', metric_samples(
        aa_tp, sizes, n_true, universe, complete), observed))
    raw = {'amino_acid_tp': aa_tp, 'sizes': sizes, 'complete': complete, 'observed_tp': actual_tp}
    for arm, draws, burn, thin, seed in (
        ('residue_degree', args.draws, 100, 20, target_seed + 1),
        ('residue_degree_long', args.long_draws, 500, 100, target_seed + 2),
    ):
        tp, records = degree_draws(maps, truth, draws, burn, thin, seed, args.threads)
        raw[arm + '_tp'] = tp
        diagnostics.extend({'target_id': target['target_id'], 'split': target['split'],
                            'arm': arm, **row} for row in records)
        results.extend(summarize_draws(target, arm, metric_samples(
            tp, sizes, n_true, universe, complete), observed))
    args.raw_out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.raw_out / (target['stem'] + '.npz'), **raw)
    print(target['stem'], target['split'],
          {r['arm']: round(r['null_expected_best'], 4) for r in results
           if r['metric'] == 'f1' and r['k'] == 1000},
          f"model={max(observed['f1']):.4f} seconds={time.monotonic() - start:.1f}", flush=True)
    return results, diagnostics


def main() -> None:
    """Run controls on the frozen 23-target pool and save reproducible tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=HERE / 'data/sampling_controls_v1')
    parser.add_argument('--raw-out', type=Path, default=Path('/tmp/exp350-sampling-controls-v1'))
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--long-draws', type=int, default=250)
    parser.add_argument('--threads', type=int, default=24)
    parser.add_argument('--seed', type=int, default=3501)
    parser.add_argument('--target', help='Optional target stem for a small execution check')
    args = parser.parse_args()
    if min(args.draws, args.long_draws) < 20:
        raise ValueError('At least 20 draws are required for block simulation error estimates')
    source_manifest = HERE / 'data/sampling_v1/manifest.json'
    provenance = json.loads(source_manifest.read_text())
    for name, digest in provenance['raw_files'].items():
        if hashlib.sha256((args.root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Raw rollout artifact hash mismatch: {name}')
    samples: dict[str, list[dict]] = {}
    for path in sorted((args.root / 'rollouts').glob('*.parquet')):
        for row in pq.read_table(path).to_pylist():
            samples.setdefault(row['stem'], []).append(row)
    target_path = HERE / 'data/foldbench_complex_contact_eval_targets.parquet'
    targets = pq.read_table(target_path).to_pylist()
    if args.target:
        targets = [target for target in targets if target['stem'] == args.target]
        if len(targets) != 1:
            raise ValueError(f'Unknown target: {args.target}')
    elif set(samples) != {target['stem'] for target in targets}:
        raise ValueError('Raw target set does not match frozen target set')
    results, diagnostics = [], []
    for target in targets:
        rows, records = run_target(target, sorted(samples[target['stem']], key=lambda row: row['rollout']), args)
        results.extend(rows)
        diagnostics.extend(records)
        write_csv(args.out / 'per_target.csv', results)
        write_csv(args.raw_out / 'mixing_per_rollout.csv', diagnostics)
    with (HERE / 'data/sampling_v1/per_target.csv').open() as handle:
        reference = {(r['target_id'], int(r['k'])): r for r in csv.DictReader(handle)}
    for row in results:
        prior = float(reference[row['target_id'], row['k']]['expected_best_' + row['metric']])
        if not np.isclose(prior, row['model_expected_best'], rtol=0, atol=1e-12):
            raise ValueError(f"Model score changed in conditional analysis: {row['target_id']}")
    summary = []
    for split in ('dev', 'test', 'all'):
        for arm in ('amino_acid_pair', 'residue_degree', 'residue_degree_long'):
            for metric in ('f1', 'r_precision'):
                for k in BUDGETS:
                    rows = [r for r in results if r['arm'] == arm and r['metric'] == metric and r['k'] == k
                            and (split == 'all' or r['split'] == split)]
                    if not rows:
                        continue
                    low, high = group_interval(rows, 'model_minus_null')
                    summary.append({'split': split, 'arm': arm, 'metric': metric, 'k': k,
                                    'n_targets': len(rows), 'n_groups': len({r['group_id'] for r in rows}),
                                    'mean_model': np.mean([r['model_expected_best'] for r in rows]),
                                    'mean_null': np.mean([r['null_expected_best'] for r in rows]),
                                    'mean_delta': np.mean([r['model_minus_null'] for r in rows]),
                                    'delta_95_low': low, 'delta_95_high': high,
                                    'null_mc_se': np.sqrt(sum(r['null_mc_block_se']**2 for r in rows)) / len(rows),
                                    'n_model_above_null': sum(r['model_minus_null'] > 0 for r in rows)})
    write_csv(args.out / 'summary.csv', summary)
    mixing = []
    for target in targets:
        for metric in ('f1', 'r_precision'):
            for k in (100, 1000):
                pair = {row['arm']: row for row in results if row['target_id'] == target['target_id']
                        and row['metric'] == metric and row['k'] == k}
                short, long = pair['residue_degree'], pair['residue_degree_long']
                mc_se = np.hypot(short['null_mc_block_se'], long['null_mc_block_se'])
                difference = long['null_expected_best'] - short['null_expected_best']
                mixing.append({'target_id': target['target_id'], 'split': target['split'],
                               'metric': metric, 'k': k, 'primary_null': short['null_expected_best'],
                               'long_chain_null': long['null_expected_best'], 'long_minus_primary': difference,
                               'combined_mc_se': mc_se, 'difference_in_mc_se': difference / mc_se if mc_se else 0.0})
    write_csv(args.out / 'mixing_comparison.csv', mixing)
    diagnostic_summary = []
    for target in targets:
        for arm in ('residue_degree', 'residue_degree_long'):
            records = [row for row in diagnostics if row['target_id'] == target['target_id'] and row['arm'] == arm]
            nonempty = [row for row in records if row['n_predicted']]
            diagnostic_summary.append({'target_id': target['target_id'], 'split': target['split'], 'arm': arm,
                                       'n_nonempty_maps': len(nonempty),
                                       'n_no_valid_switch': sum(row['no_valid_switch'] for row in records),
                                       'mean_acceptance': np.mean([row['acceptance_rate'] for row in nonempty]),
                                       'mean_original_edge_fraction': np.mean([row['mean_original_edge_fraction'] for row in nonempty]),
                                       'mean_tp_lag1': np.mean([row['tp_lag1_autocorrelation'] for row in nonempty])})
    write_csv(args.out / 'mixing_summary.csv', diagnostic_summary)
    manifest = {'public_prefix': PUBLIC, 'n_targets': len(targets),
                'draws': args.draws, 'long_draws': args.long_draws, 'seed': args.seed,
                'threads': args.threads, 'budgets': list(BUDGETS),
                'amino_acid_null': 'exact counts in ordered chain-A/chain-B residue-type strata; no replacement within each map',
                'validation': 'raw input hashes, all model scores, final graph simplicity and both degree sequences, no-move graphs have no valid switch',
                'degree_chain': {'burn_per_edge': 100, 'thin_per_edge': 20},
                'degree_chain_long': {'burn_per_edge': 500, 'thin_per_edge': 100},
                'source_raw_manifest_sha256': hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
                'targets_sha256': hashlib.sha256(target_path.read_bytes()).hexdigest(),
                'code_sha256': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                for name in ('score_sampling_controls.py', 'degree_switch.cpp')},
                'derived_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in args.out.iterdir() if p.is_file() and p.name != 'manifest.json'},
                'raw_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in args.raw_out.iterdir() if p.is_file()},
                'degree_method_reference': 'https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.APPROX-RANDOM.2018.36',
                'limitations': 'degree sampling is finite-chain approximate, not an exact independent sampler'}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for row in summary:
        if row['split'] == 'test' and row['k'] in (100, 1000):
            print(row, flush=True)


if __name__ == '__main__':
    main()
