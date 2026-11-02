"""Ensure reserved-test analysis cannot silently expand the frozen comparison."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def analysis_command(root: Path) -> list[str]:
    """Build a standalone analysis invocation against synthetic test inputs."""
    return [sys.executable, str(Path(__file__).with_name('analyze.py')),
            '--plan', str(root / 'plan.json'), '--targets', str(root / 'targets.parquet'),
            '--pool', str(root / 'pool'), '--pool-generation', str(root / 'pool'),
            '--out', str(root / 'results')]


def test_reserved_analysis_only_scores_the_frozen_pair(tmp_path: Path) -> None:
    plan = dict(split='test', entries=['synthetic'], pool_mode='independent', pool_sizes=[3],
                evaluation_configs=['ordinary-direct:3:frequency', 'ordinary-direct:3:sumq'])
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    pq.write_table(pa.Table.from_pylist([dict(entry_id='synthetic', group_id='g', split='test',
        L=32, contacts=[[0, 24]])]), tmp_path / 'targets.parquet')
    pool = tmp_path / 'pool' / 'synthetic'
    pool.mkdir(parents=True)
    # Two low-confidence wrong votes outweigh one correct vote under frequency;
    # confidence reverses that ranking. No recall field is supplied, so an
    # accidental evaluation of the unselected F1 method also fails this test.
    rows = [dict(sample=i, L=32, pairs=[[1, 25] if i < 2 else [0, 24]],
                 probabilities=[.1 if i < 2 else .9], finished=True) for i in range(3)]
    pq.write_table(pa.Table.from_pylist(rows), pool / 'pool-3.parquet')
    (pool / 'pool-3.done.json').write_text('{}')
    pd.DataFrame([dict(elapsed_seconds=1., gpu_name='synthetic')]).to_csv(
        pool / 'pool-3.timings.csv', index=False)
    subprocess.run(analysis_command(tmp_path), check=True, capture_output=True, text=True)
    result = pd.read_csv(tmp_path / 'results' / 'per_protein.csv.gz').set_index('method')
    assert set(result.index) == {'frequency', 'sumq'}
    assert result.loc['frequency', 'r_precision'] == 0.
    assert result.loc['sumq', 'r_precision'] == 1.


def test_incomplete_reserved_cohort_fails_before_reading_targets(tmp_path: Path) -> None:
    plan = dict(split='test', entries=['uncommitted'], pool_mode='independent', pool_sizes=[128],
                evaluation_configs=['ordinary-direct:128:frequency', 'ordinary-direct:128:sumq'])
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    result = subprocess.run(analysis_command(tmp_path), capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Reserved-test cohort is incomplete' in result.stderr
    assert not (tmp_path / 'results').exists()
