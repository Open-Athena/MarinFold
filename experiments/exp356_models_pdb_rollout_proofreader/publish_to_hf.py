"""Publish the evaluated model and reproducible artifacts to the public HF bucket.

Run on a CPU worker in the same CoreWeave region as the working artifacts. The
export omits optimizer states and checks the entire outbound copy against the
repository's 10 GB transfer limit before moving any large object off-region.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfFileSystem

from storage import ROOT, filesystem, stage_directory, write_json
from training_data import stage_rollouts

HERE=Path(__file__).resolve().parent
PUBLIC='hf://buckets/open-athena/MarinFold'


def file_manifest(directory: Path) -> dict:
    """Hash the actual export, including its inference code and co-located tokenizer."""
    result={}
    for path in sorted(directory.iterdir()):
        if path.is_file() and not path.name.startswith('.'):
            with path.open('rb') as handle:
                digest=hashlib.file_digest(handle,'sha256').hexdigest()
            result[path.name]=dict(size=path.stat().st_size,sha256=digest)
    return result


def sync(directory: Path, destination: str) -> None:
    """Upload through the supported HF CLI without exposing credentials in argv."""
    subprocess.run(['hf','buckets','sync',str(directory),destination,'--exclude','.*','--quiet'],check=True)


def main() -> None:
    """Export only a checkpoint named by the completed release evaluation report."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data',default=ROOT+'/data/v1/rollouts')
    parser.add_argument('--reports',default=ROOT+'/reports/release')
    parser.add_argument('--targets',default=ROOT+'/data/v1/targets.parquet')
    parser.add_argument('--out',default=ROOT+'/release.json')
    args=parser.parse_args()
    local=Path('/tmp/exp356-publication')
    reports=local/'reports'
    stage_directory(args.reports,reports)
    release=json.loads((reports/'release.json').read_text())
    if release['checkpoint']!=args.checkpoint or not release['validation_complete'] or not release['test_complete']:
        raise ValueError('Release report does not identify a fully evaluated checkpoint')
    if not (reports/'model_card.md').exists():
        raise FileNotFoundError('Release reports must include model_card.md')
    model=local/'model'
    stage_directory(args.checkpoint,model,include_training_state=False)
    metadata=json.loads((model/'proofreader.json').read_text())
    run_name=metadata['config']['run_name']
    step=metadata['step']
    if run_name.startswith('debug-'):
        raise ValueError('Smoke checkpoints cannot be published as production models')
    # The training manifest lists optimizer files; replace it with one describing
    # precisely the public, inference-only export.
    training_manifest=json.loads((model/'manifest.json').read_text())
    (model/'manifest.json').unlink()
    shutil.copyfile(reports/'model_card.md',model/'README.md')
    for name in ['infer.py','model.py','records.py','storage.py','pyproject.toml','uv.lock']:
        shutil.copyfile(HERE/name,model/name)
    export=file_manifest(model)
    for name,item in training_manifest['files'].items():
        if name!='training_state.pt' and export.get(name)!=item:
            raise ValueError(f'Export differs from the committed checkpoint: {name}')
    (model/'manifest.json').write_text(json.dumps(dict(source_checkpoint=args.checkpoint,files=export),indent=2))
    data=local/'data'
    manifest=stage_rollouts(args.data,data)
    for item in manifest['files']:
        with (data/item['name']).open('rb') as handle:
            digest=hashlib.file_digest(handle,'sha256').hexdigest()
        if digest!=item['sha256']:
            raise ValueError(f'Staged rollout differs from the audited corpus: {item["name"]}')
    shutil.copyfile(data/'manifest.json',data/'_SUCCESS.json')
    fs,key=filesystem(args.targets)
    fs.get_file(key,str(data/'targets.parquet'))
    dataset_bytes=sum(p.stat().st_size for p in data.iterdir() if p.is_file())
    total_bytes=sum(p.stat().st_size for d in [model,data,reports] for p in d.iterdir() if p.is_file())
    if total_bytes>10_000_000_000:
        raise ValueError(f'Export is {total_bytes} bytes, exceeding the explicit-approval transfer limit')
    model_uri=f'{PUBLIC}/checkpoints/{run_name}/step-{step}'
    data_uri=f'{PUBLIC}/data/exp356/rollouts-v1'
    report_uri=f'{PUBLIC}/data/exp356/reports'
    print(json.dumps(dict(model=model_uri,data=data_uri,reports=report_uri,bytes=total_bytes),indent=2),flush=True)
    sync(model,model_uri)
    sync(data,data_uri)
    sync(reports,report_uri)
    anonymous=HfFileSystem(token=False)
    public_model=model_uri.removeprefix('hf://')
    with anonymous.open(public_model+'/manifest.json') as handle:
        if json.load(handle)['files']!=export:
            raise ValueError('Anonymous public manifest differs from the export')
    for name,item in export.items():
        if anonymous.info(public_model+'/'+name)['size']!=item['size']:
            raise ValueError(f'Public model file has wrong size: {name}')
    with anonymous.open(data_uri.removeprefix('hf://')+'/_SUCCESS.json') as handle:
        if json.load(handle)!=manifest:
            raise ValueError('Anonymous dataset manifest differs from the audited corpus')
    public_files={Path(item['name']).name:item['size']
        for item in anonymous.ls(data_uri.removeprefix('hf://'),detail=True) if item['type']=='file'}
    for path in data.iterdir():
        if path.is_file() and not path.name.startswith('.') and public_files.get(path.name)!=path.stat().st_size:
            raise ValueError(f'Public dataset file is missing or has wrong size: {path.name}')
    with anonymous.open(report_uri.removeprefix('hf://')+'/release.json') as handle:
        if json.load(handle)!=release:
            raise ValueError('Anonymous release report differs from the evaluated release')
    result=dict(checkpoint=args.checkpoint,model=model_uri,data=data_uri,reports=report_uri,
        total_export_bytes=total_bytes,dataset_bytes=dataset_bytes,
        dataset_counts=manifest['counts'],published_at=datetime.now(UTC).isoformat())
    write_json(result,args.out)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
