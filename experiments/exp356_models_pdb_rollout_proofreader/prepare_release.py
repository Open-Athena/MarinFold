"""Assemble the evaluated production release and its human-readable model card."""

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from storage import ROOT, filesystem, upload_directory, write_json

HERE = Path(__file__).resolve().parent
PUBLIC = 'hf://buckets/open-athena/MarinFold'


def remote_json(uri: str) -> dict:
    """Read one small durable run record."""
    fs, key = filesystem(uri)
    return json.loads(fs.cat(key))


def model_card(release: dict, metadata: dict, metrics: dict) -> str:
    """Describe the actual selected checkpoint, scope, evidence and inference API."""
    full, aggregation = metrics['full'], metrics['aggregation']
    rows = '\n'.join(f"| {prefix} | {metrics[prefix]['brier']['mean']:.4f} | "
        f"{metrics[prefix]['precision_mae']['mean']:.4f} | {metrics[prefix]['recall_mae']['mean']:.4f} |"
        for prefix in ['1','2','4','8','16','32','64','full'])
    context = metrics['context']
    return f'''# MarinFold experimental-PDB rollout proofreader

This model predicts a probability of correctness for every proposed contact in a
MarinFold rollout or supplied prefix, plus current precision and recall. All supplied
contacts attend to one another in the readout, so later contacts can revise earlier
judgments. A single contact is a supported input; no unseen continuation is supplied.

## Model and training

- Checkpoint: `{release['model']}`.
- Architecture: `{metadata['architecture']}`. Frozen exp277 step-266344 causal
  Qwen3 features feed a four-layer, 512-wide bidirectional contact encoder.
- 1,487,731,202 total parameters; 22,181,378 trainable parameters.
- Training: 50,000 experimental PDB chain structures (46,933 PDB entries;
  34,894 distinct supplied sequences; 9,074 connected related groups).
- 399,213 retained training rollouts; eight generator samples requested per structure.
  Three completed epochs, global batch 64 on eight CoreWeave H100 GPUs.
- Selected step: {metadata['step']}; training completed at step {release['training_completed_step']}.
- [W&B]({release['wandb_url']}); [implementation and reports](https://github.com/Open-Athena/MarinFold/tree/{release['git_commit']}/experiments/exp356_models_pdb_rollout_proofreader).

The input is physically cropped before encoding. Training draws 25% full rollouts,
25% short prefixes of 1/2/4/8 contacts, and 50% uniformly sampled prefix lengths.
Contact probabilities use binary cross-entropy; the recall head uses squared error.
Estimated precision is the mean probability over distinct proposed pairs. Duplicate
contacts do not increase precision or recall counts. A true end marker is retained
only when the generator has actually emitted it.

## Reserved test results

The final test covers 1,473 structures in 295 related groups. The 1,552-structure
validation split selected the architecture and checkpoint; the reserved test was
evaluated after selection. Values below are means over proteins after averaging
their rollouts. Full-rollout contact AUROC is {full['auroc']['mean']:.4f}.

| Contacts supplied | Contact Brier | Precision MAE | Recall MAE |
| --- | ---: | ---: | ---: |
{rows}

Later contacts improve first-contact Brier by {context['mean']:.4f}
(95% group-bootstrap interval {context['low']:.4f}–{context['high']:.4f}).
Keeping the highest-scored half yields precision
{full['precision_retained_0.5']['mean']:.4f}, compared with
{full['precision_emission_0.5']['mean']:.4f} for the first half in emission order.
Selecting a rollout by estimated precision gives
{aggregation['selected_rollout_precision']['mean']:.4f}, versus
{aggregation['random_rollout_precision']['mean']:.4f} for a random rollout.
Probability-weighted aggregation R-precision is
{aggregation['weighted_r_precision']['mean']:.4f}, versus
{aggregation['frequency_r_precision']['mean']:.4f} for frequency alone.
The report JSONs include 1,000-replicate confidence intervals resampling related groups.
No post-hoc calibration is applied to this release.

## Inference

The checkpoint contains the frozen backbone, trained heads, tokenizer, source and
pinned dependencies. Use Linux and Python 3.12; the GPU path uses PyTorch bfloat16
autocast. Download with the Hugging Face CLI:

```bash
hf buckets sync {release['model']} ./proofreader
uv run --project ./proofreader --extra train python ./proofreader/infer.py \\
  --checkpoint ./proofreader --rollout rollout.json --out predictions.json
```

`rollout.json` is `{{"prompt_ids": [...], "completion_ids": [...]}}` using this
checkpoint's tokenizer and the actual generator prompt. End a prefix at a complete
contact triple and omit all subsequent tokens. Outputs include the ordered contact
pairs, `probability_correct`, `estimated_precision`, and `estimated_recall`.
The CLI also saves a timing CSV. Loading the existing model once and calling
`score_rollout` avoids repeated model loading for multiple requests.

Alternatively, supply `--sequence SEQUENCE --contacts pairs.json`, where the file
contains zero-based pairs such as `[[0, 8], [5, 20]]`. Add `--finished` only for an
actually completed rollout. This constructs a new sequence prompt; preserving the
original tokenized rollout matches the evaluated input format most closely.

## Scope and reference definition

The measured distribution is actual exp277 step-266344 rollouts at temperature 1
and top-p 0.95, for single protein chains of 32–1,000 supplied residues. The context
limit is 8,192 tokens including the assessment token. Other generators, sampling
settings, proteins outside this range, complexes and edited rollouts are unvalidated.

References are the complete native-only exp222 pyconfind contact sets: 3 Å side-chain
threshold, minimum contact degree 0.001, sequence separation at least six. Recall
uses that full reference contact count, not the number of emitted contacts. Only
observed residues are supplied: a canonical sequence or one contiguous segment
covering at least 90%, with at most 20 omitted terminal residues per end. Internal
gaps and truncated reference contact sets are rejected.

Splits keep PDB entries, exact supplied sequences and connected homology groups
together, with explicit cross-split searches and benchmark-homolog exclusion.
These are held-out structures for proofreader training. Absence from the pretrained
generator's training corpus is not established. Scores are probabilistic estimates
against one experimental reference, not a guarantee that a contact is physically
impossible in another conformation.

## Data and reproducibility

Public raw rollouts and target labels: `{release['data']}`.
Public metrics, provenance, per-rollout tables, timing ledgers and plots:
`{release['reports']}`. Every model file has a SHA256 entry in `manifest.json`.
The corpus `_SUCCESS.json` identifies all audited parquet files and their hashes.
Training optimizer states are retained on working storage, outside this inference export.
'''


def main() -> None:
    """Publish a report bundle only after both complete held-out evaluations exist."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation-label', required=True)
    parser.add_argument('--test-label', required=True)
    args = parser.parse_args()
    validation = json.loads((HERE/'data'/f'{args.validation_label}_evaluation.json').read_text())
    test = json.loads((HERE/'data'/f'{args.test_label}_evaluation.json').read_text())
    checkpoint = validation['checkpoint']
    if test['checkpoint'] != checkpoint or validation['data_fingerprint'] != test['data_fingerprint']:
        raise ValueError('Release evaluations must share a checkpoint and audited corpus')
    if (validation['split'], validation['proteins'], test['split'], test['proteins']) != ('validation',1552,'test',1473):
        raise ValueError('Release requires complete production validation and test coverage')
    metadata = remote_json(checkpoint+'/proofreader.json')
    if metadata['architecture'] != 'exp277-frozen-causal-contact-encoder-v2':
        raise ValueError('The release card describes the selected frozen-backbone architecture')
    if metadata['data_fingerprint'] != validation['data_fingerprint']:
        raise ValueError('The checkpoint and release evaluation use different corpus manifests')
    run = metadata['config']['run_name']
    completion = remote_json(f'{ROOT}/runs/{run}/complete.json')
    if completion['step'] != completion['max_steps'] or completion['step'] != 18714:
        raise ValueError('The production three-epoch schedule has not completed')
    git_commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()
    release = dict(checkpoint=checkpoint, validation_complete=True, test_complete=True,
        validation_label=args.validation_label, test_label=args.test_label,
        model=f'{PUBLIC}/checkpoints/{run}/step-{metadata["step"]}',
        data=f'{PUBLIC}/data/exp356/rollouts-v1', reports=f'{PUBLIC}/data/exp356/reports',
        training_completed_step=completion['step'], data_fingerprint=validation['data_fingerprint'],
        wandb_url=f'https://wandb.ai/open-athena/MarinFold/runs/{run}', git_commit=git_commit,
        prepared_at=datetime.now(UTC).isoformat())
    local = HERE/'_cache/release'/run/f'step-{metadata["step"]}'
    local.mkdir(parents=True,exist_ok=True)
    for directory in [HERE/'data',HERE/'plots']:
        for path in directory.iterdir():
            if path.is_file() and path.suffix in {'.json','.jsonl','.csv','.gz','.png','.pdf'}:
                shutil.copyfile(path,local/path.name)
    (local/'release.json').write_text(json.dumps(release,indent=2))
    metrics = json.loads((HERE/'data'/f'{args.test_label}_metrics.json').read_text())
    (local/'model_card.md').write_text(model_card(release,metadata,metrics))
    destination = f'{ROOT}/reports/release/{run}/step-{metadata["step"]}'
    files = upload_directory(local,destination)
    write_json(dict(release,files=files),destination+'/_SUCCESS.json')
    print(json.dumps(dict(release,reports_source=destination),indent=2))


if __name__ == '__main__':
    main()
