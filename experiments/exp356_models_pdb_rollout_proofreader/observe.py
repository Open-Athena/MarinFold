"""Take one compact monitoring snapshot; the agent owns the recovery loop."""

import argparse
import json
import subprocess
from pathlib import Path

from storage import ROOT, filesystem

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Read job state and durable progress without creating competing monitor loops."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--job')
    parser.add_argument('--run')
    parser.add_argument('--rollouts')
    args = parser.parse_args()
    if args.job:
        result = subprocess.run(['uv','run','--project','/home/bizon/git/marin','--package','marin-iris',
            'iris','--cluster','marin','rpc','controller','get-job-status','--job-id',args.job],
            capture_output=True,text=True,check=True)
        status=json.loads(result.stdout)
        path=HERE/'_cache/observations'
        path.mkdir(parents=True,exist_ok=True)
        (path/(args.job.replace('/','_')+'.json')).write_text(json.dumps(status,indent=2))
        # Preserve compact top-level state; full task details remain on disk.
        print(json.dumps({k:v for k,v in status.items() if k not in {'tasks','request','environment'}},default=str)[:6000])
    if args.run:
        for name in ['started.json','progress.json','resume.json','best.json','complete.json']:
            fs,key=filesystem(f'{ROOT}/runs/{args.run}/{name}')
            if fs.exists(key):
                print(name,fs.cat(key).decode())
    if args.rollouts:
        fs,root=filesystem(args.rollouts)
        objects=fs.ls(root,detail=False) if fs.exists(root) else []
        print(json.dumps(dict(committed_batches=sum(p.endswith('.done.json') for p in objects),
            completed_shards=[Path(p).name for p in objects if p.endswith('.complete.json')],
            dataset_committed=any(p.endswith('_SUCCESS.json') for p in objects))))


if __name__=='__main__':
    main()
