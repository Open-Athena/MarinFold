"""Freeze inputs and analysis choices before any new prediction is inspected."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def digest(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Select only authorized proteins and pin checkpoint identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-repo', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_repo
    origin = source / 'experiments/exp245_evals_foldbench_held_out_monomers/data'
    output = HERE / 'inputs'
    output.mkdir(exist_ok=True)
    sets = pd.read_csv(origin / 'eval_sets.csv')
    sets = sets[sets.eval_set.isin(['eval-val', 'eval-denovo']) & sets.scorable.eq(1)].copy()
    assert sets.eval_set.value_counts().to_dict() == {'eval-val': 97, 'eval-denovo': 19}
    targets = pd.read_parquet(origin / 'eval_targets_foldbench_monomers.parquet')
    targets = targets.merge(sets, on='stem', validate='one_to_one')
    assert len(targets) == 116 and (targets.input_seq == targets.sequence).all()
    assert (targets.L == targets.seq_len).all()
    depths = pd.read_csv(REPO / 'experiments/exp260_evals_msa_depth_stratified/data/msa_depth.csv')
    depths = depths[depths.msa_volume.eq('foldbench')].copy()
    targets = targets.merge(depths, on='stem', how='left', validate='one_to_one')
    # An absent or mismatched alignment cannot silently enter a depth tier.
    targets['msa_identity_valid'] = targets.found.eq(True) & targets.query_len.eq(targets.L)
    targets['msa_neff'] = targets['n_eff_0.8'].where(targets.msa_identity_valid)
    targets['msa_depth'] = targets.n_seqs.where(targets.msa_identity_valid)
    targets.to_csv(HERE / 'data/targets.csv', index=False)
    targets[['dataset', 'stem', 'L', 'input_seq', 'eval_set']].to_json(output / 'targets.json', orient='records', indent=2)
    truth_path = origin / 'gt_universe_scored.jsonl'
    assert digest(truth_path) == 'f30c23e3d2fbab245755fc01548388b41730ddfa45da87325539698cadb153e5'
    selected = set(targets.stem)
    truth = [json.loads(line) for line in truth_path.read_text().splitlines() if json.loads(line)['stem'] in selected]
    assert len(truth) == 116 and {r['stem'] for r in truth} == selected
    (output / 'ground_truth.json').write_text(json.dumps(truth))
    mpnn = json.loads((source / 'experiments/exp364_models_mpnn_only_lr_sweep/data/publication_source.json').read_text())
    spec_path = REPO / 'experiments/exp277_models_single_mpnn_pilot/evals/2026-09-13_rollout_v2/checkpoint_specs.py'
    spec = importlib.util.spec_from_file_location('checkpoint_specs', spec_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    default = module.EXP277_CHECKPOINT
    models = {
        'mpnn': {'nickname': mpnn['checkpoint']['run_id'], 'step': 186884,
                 'uri': mpnn['checkpoint']['hf_export'], 'files': mpnn['files']},
        'default': {'nickname': default.run_name, 'step': default.step, 'uri': default.coreweave_uri,
                    'files': [{'name': f.name, 'size': f.size, 'etag': f.digest} for f in default.checkpoint_files]},
    }
    natural = targets[targets.eval_set.eq('eval-val')].sort_values(['msa_neff', 'stem'])
    panel = natural.iloc[[0, 13, 27, 41, 55, 69, 83, 96]].stem.tolist()
    plan = {
        'issue': 373, 'cluster': 'cw-us-east-02a', 'gpu_limit': 12,
        'models': models, 'eval_counts': {'eval-val': 97, 'eval-denovo': 19},
        'temperatures': [1.0, 0.7, 1.3], 'fixed_blocks': [100, 100, 100],
        'randomized_rollouts': 100, 'top_p': 0.95, 'top_k': -1,
        'dtype': 'bfloat16', 'max_context': 8192, 'initial_budget': 'min(6L+128,8192-prompt_tokens)',
        'continuation': 'Preserve prompt and generated prefix; continue to remaining context with independent derived seed; retain capped samples.',
        'sample_seed': 'SHA256(exp373:stem:mode:block:rollout) mod (2**63-1)',
        'fixed_prompt_id': '{stem}:r0', 'randomized_prompt_id': '{stem}:r{rollout}',
        'representative_panel': panel, 'bootstrap_seed': 373, 'bootstrap_draws': 10000,
        'neff_tiers': ['<10', '10–<100', '100–<1000', '≥1000'],
        'marinfold_revision': 'd1bea417a64cc042ad931422200c3edeb873f2e0',
        'artifact_root': 'hf://buckets/open-athena/MarinFold/data/exp373-mpnn-diversity-neff/v1',
        'source_inputs': {str(p.relative_to(source)): digest(p) for p in [origin/'eval_sets.csv', origin/'eval_targets_foldbench_monomers.parquet', truth_path]},
        'metadata_sha256': digest(REPO/'experiments/exp260_evals_msa_depth_stratified/data/msa_depth.csv'),
        'metric_source_sha256': digest(REPO/'experiments/exp89_evals_contacts_v1_model_on_eval_set/compute_metrics.py'),
    }
    (HERE / 'data/plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(targets.groupby('eval_set').agg(n=('stem','size'), valid_neff=('msa_neff','count')))
    print(pd.cut(natural.msa_neff, [0,10,100,1000,float('inf')], right=False).value_counts(sort=False))
    print('Panel:', panel)


if __name__ == '__main__':
    main()
