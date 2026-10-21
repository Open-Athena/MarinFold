"""Sum successful warm inference work without double-counting nested prefixes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    """Produce a phase ledger from complete contemporaneous timing tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', action='append', required=True, help='phase_name=analysis_directory')
    parser.add_argument('--nested-length-phase', action='append', default=[],
                        help='Count only the final 2L cumulative timing in this fixed-length phase')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    phases = [spec.split('=', 1) for spec in args.phase]
    if len({name for name, _ in phases}) != len(phases):
        raise ValueError('Repeated phase would double-count compute')
    if not set(args.nested_length_phase) <= {name for name, _ in phases}:
        raise ValueError('Unknown nested-length phase')
    records, sources = [], []
    for name, directory in phases:
        path = Path(directory)
        complete = pd.read_csv(path / 'completeness.csv')
        if complete.empty or not complete.complete.all():
            raise ValueError('Phase is incomplete: ' + name)
        timing_path = path / 'timings.csv'
        timing = pd.read_csv(timing_path)
        if name in args.nested_length_phase:
            if not timing['mode'].str.endswith(('fixed1L', 'fixed2L')).all():
                raise ValueError('Nested-length filtering is only valid for L/2L timings')
            timing = timing[timing['mode'].str.endswith('fixed2L')]
        if (timing.empty or timing.gpu_name.isna().any()
                or not np.isfinite(timing.elapsed_seconds).all() or (timing.elapsed_seconds < 0).any()):
            raise ValueError('Missing hardware or invalid timing: ' + name)
        keys = [column for column in ['stem', 'mode', 'sample', 'sample_start', 'sample_end'] if column in timing]
        if timing.duplicated(keys).any():
            raise ValueError('Duplicate stage timing identities: ' + name)
        for hardware, rows in timing.groupby('gpu_name'):
            seconds = rows.elapsed_seconds.sum()
            records.append(dict(phase=name, hardware=hardware, proteins=rows.stem.nunique(),
                stage_records=len(rows), successful_warm_gpu_seconds=seconds,
                successful_warm_gpu_hours=seconds / 3600))
        sources.append(dict(phase=name, timings=str(timing_path),
            sha256=hashlib.sha256(timing_path.read_bytes()).hexdigest(),
            retained='only final cumulative 2L records' if name in args.nested_length_phase else 'all stage records'))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    result.to_csv(args.out, index=False)
    args.out.with_suffix('.provenance.json').write_text(json.dumps(dict(sources=sources,
        scope='Successful warm single-GPU inference stages only. Not billed GPU-hours. Excludes staging, loading, '
              'queueing, CPU aggregation, diagnostics, smoke tests and work lost to failed/preempted attempts. '
              'Nested L/2L continuations count only the cumulative 2L timing.'), indent=2) + '\n')
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
