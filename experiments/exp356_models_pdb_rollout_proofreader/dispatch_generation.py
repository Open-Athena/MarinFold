"""Submit a bounded set of independent production inference shards."""

import argparse
import subprocess
import sys
from pathlib import Path

from storage import ROOT

HERE=Path(__file__).resolve().parent


def main() -> None:
    """Record every root job; allow explicit continuation after a submission error."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--shards',type=int,default=32)
    parser.add_argument('--start-shard',type=int,default=0)
    parser.add_argument('--stop-shard',type=int)
    parser.add_argument('--attempt',type=int,default=1)
    parser.add_argument('--cluster',default='cw-us-east-02a')
    args=parser.parse_args()
    stop=args.stop_shard if args.stop_shard is not None else args.shards
    if not 0<=args.start_shard<stop<=args.shards:
        parser.error('Require 0 <= start < stop <= shard count')
    logs=HERE/'_cache/submissions'
    logs.mkdir(parents=True,exist_ok=True)
    for shard in range(args.start_shard,stop):
        name=f'exp356-generate-prod-s{shard:02d}-a{args.attempt:02d}'
        command=[sys.executable,str(HERE/'launch.py'),'generate','--name',name,'--gpus','1',
            '--cluster',args.cluster,'--','--targets',ROOT+'/data/v1/targets.parquet',
            '--out',ROOT+'/data/v1/rollouts','--shards',str(args.shards),'--shard',str(shard)]
        log=logs/f'{name}.log'
        with log.open('w') as output:
            result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Submission failed at shard {shard}; inspect {log}')
        print(f'Submitted /bizon/{name}',flush=True)


if __name__=='__main__':
    main()
