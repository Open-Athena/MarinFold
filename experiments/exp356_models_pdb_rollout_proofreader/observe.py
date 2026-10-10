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
    parser.add_argument('--brief',action='store_true')
    args = parser.parse_args()
    if args.job:
        result = subprocess.run(['uv','run','--project','/home/bizon/git/marin','--package','marin-iris',
            'iris','--cluster','marin','rpc','controller','get-job-status','--job-id',args.job],
            capture_output=True,text=True,check=True)
        status=json.loads(result.stdout)
        status={'job':status['job']}
        path=HERE/'_cache/observations'
        path.mkdir(parents=True,exist_ok=True)
        (path/(args.job.replace('/','_')+'.json')).write_text(json.dumps(status,indent=2))
        # Job state is sufficient for monitoring; do not persist request env vars.
        print(json.dumps(status,default=str))
    if args.run:
        records={}
        for name in ['started.json','progress.json','resume.json','best.json','paused.json','complete.json']:
            fs,key=filesystem(f'{ROOT}/runs/{args.run}/{name}')
            if fs.exists(key):
                records[name]=json.loads(fs.cat(key))
                if not args.brief:
                    print(name,json.dumps(records[name]))
        if args.brief:
            progress=records.get('progress.json',{})
            result={key:progress[key] for key in ['step','epoch','train/loss','elapsed_seconds'] if key in progress}
            result['max_steps']=records.get('started.json',{}).get('max_steps')
            result['best']=records.get('best.json')
            fs,root=filesystem(f'{ROOT}/runs/{args.run}')
            validations=fs.glob(root+'/validation-step-*.json')
            if validations:
                latest=max(validations,key=lambda p:int(Path(p).stem.split('-')[-1]))
                result['validation']=json.loads(fs.cat(latest))
            result['complete']='complete.json' in records
            if 'paused.json' in records:
                result['last_pause_step']=records['paused.json']['step']
            print(json.dumps(result))
    if args.rollouts:
        fs,root=filesystem(args.rollouts)
        objects=fs.ls(root,detail=False) if fs.exists(root) else []
        print(json.dumps(dict(committed_batches=sum(p.endswith('.done.json') for p in objects),
            completed_shards=[Path(p).name for p in objects if p.endswith('.complete.json')],
            dataset_committed=any(p.endswith('_SUCCESS.json') for p in objects))))


if __name__=='__main__':
    main()
