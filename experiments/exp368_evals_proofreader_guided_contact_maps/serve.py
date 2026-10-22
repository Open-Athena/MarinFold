"""Own eight CoreWeave GPUs and one bounded, shared model transfer.

The server remains alive between experiment phases. Child failures are recorded
and reported; they never terminate the model cache or hide remaining failures.
Task descriptions and source bundles are written only by this experiment's
submission script. Every source bundle is hash-checked before extraction.
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from runtime import GENERATOR, HERE, READER, ROOT, TARGETS, digest_file, load_release, stage_reader


def stage_shared(api) -> dict:
    """Transfer one reader backbone and a small head, then reconstruct locally."""
    reader = Path('/tmp/exp368-reader')
    generator = Path('/tmp/exp368-generator')
    head_dir = Path('/tmp/exp368-head')
    generator.mkdir(parents=True, exist_ok=True)
    # The upper bound is checked before any cross-region weight transfer.
    fs, source = api.storage.filesystem(READER)
    items = [item for item in fs.ls(source, detail=True)
             if item['type'] == 'file' and Path(item['name']).name != 'training_state.pt']
    shared_fs, shared_root = api.storage.filesystem(ROOT + '/shared-generator-v1')
    shared_items = shared_fs.ls(shared_root, detail=True)
    target_fs, target_path = api.storage.filesystem(TARGETS)
    transfer_bytes = sum(item['size'] for item in items) + sum(item['size'] for item in shared_items)
    transfer_bytes += target_fs.size(target_path)
    if transfer_bytes > 7_000_000_000:
        raise ValueError(f'Shared input staging exceeds its 7 GB ceiling: {transfer_bytes}')
    stage_reader(api, reader)
    api.storage.stage_directory(ROOT + '/shared-generator-v1', head_dir)
    expected = json.loads((HERE / 'shared_generator_manifest.json').read_text())
    for name, metadata in expected.items():
        if digest_file(head_dir / name) != metadata['sha256']:
            raise ValueError('Shared generator artifact changed: ' + name)
    weights = load_file(head_dir / 'lm_head.safetensors')
    with safe_open(reader / 'model.safetensors', framework='pt', device='cpu') as tensors:
        for name in tensors.keys():
            value = tensors.get_tensor(name)
            if name == 'embed_tokens.weight':
                value = value[:2845].contiguous()
            weights['model.' + name] = value
        save_file(weights, generator / 'model.safetensors', metadata={'format': 'pt'})
    del weights
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
        shutil.copyfile(head_dir / name, generator / name)
    target_local = Path('/tmp/exp368-targets.parquet')
    target_fs.get_file(target_path, str(target_local))
    result = dict(input_transfer_bytes=transfer_bytes, reader=READER, generator=GENERATOR,
        targets_sha256=digest_file(target_local), target_source=TARGETS,
        generator_weights_sha256=digest_file(generator / 'model.safetensors'))
    Path('/tmp/exp368-shared-ready.json').write_text(json.dumps(result, indent=2))
    return result


def unpack_bundle(fs, uri: str, expected_sha: str) -> Path:
    """Extract a verified source snapshot with no absolute or traversal paths."""
    destination = Path('/tmp/exp368-sources') / expected_sha
    if destination.exists():
        return destination
    content = fs.cat(uri.removeprefix('s3://'))
    if hashlib.sha256(content).hexdigest() != expected_sha:
        raise ValueError('Source bundle hash mismatch')
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(content), mode='r:gz') as archive:
        archive.extractall(destination, filter='data')
    return destination


def mirror_committed(api, local: Path, remote: str, uploaded: dict[str, int], remaining: int) -> int:
    """Publish data before its commit markers, with a cumulative byte ceiling."""
    files = set()
    for marker in local.rglob('*.done.json'):
        base = str(marker).removesuffix('.done.json')
        for suffix in ['.parquet', '.timings.csv', '.done.json']:
            path = Path(base + suffix)
            if path.exists():
                files.add(path)
    files.update(local.rglob('*.complete.json'))
    files.update(local.rglob('*.pin.json'))
    fs, root = api.storage.filesystem(remote)
    count = 0
    for path in sorted(files, key=lambda p: (p.name.endswith(('.done.json', '.complete.json')), str(p))):
        relative = str(path.relative_to(local))
        stamp = path.stat().st_mtime_ns
        if uploaded.get(relative) == stamp:
            continue
        size = path.stat().st_size
        if count + size > remaining:
            raise ValueError('Cross-region artifact transfer would exceed the 9.5 GB campaign ceiling')
        fs.put_file(str(path), root + '/' + relative, timeout=120)
        uploaded[relative] = stamp
        count += size
    return count


def main() -> None:
    """Run queued experiment tasks on free GPUs until an explicit stop marker."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', default=ROOT + '/server-v1')
    parser.add_argument('--hours', type=float, default=48)
    args = parser.parse_args()
    torch.set_num_threads(4)
    api = load_release()
    if torch.cuda.device_count() != 8:
        raise ValueError('The shared server requires exactly eight GPUs')
    fs, control = api.storage.filesystem(args.control)
    fs.makedirs(control + '/queue', exist_ok=True)
    print('[exp368-server] staging one shared copy', flush=True)
    shared = stage_shared(api)
    api.storage.write_json(shared, args.control + '/shared_models.json')
    subprocess.run(['bash', str(HERE / 'pool_bootstrap.sh'), '--prepare-only'], check=True)
    local = Path('/tmp/exp368-results')
    local.mkdir(parents=True, exist_ok=True)
    active, finished, uploaded = {}, {}, {}
    transferred = shared['input_transfer_bytes']
    started = time.monotonic()
    while time.monotonic() - started < args.hours * 3600:
        for task_id, (process, handle, device, task, log) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            fs.put_file(str(log), control + '/logs/' + task_id + '.log', timeout=120)
            finished[task_id] = dict(state='succeeded' if code == 0 else 'failed', exit_code=code, task=task)
            api.storage.write_json(finished[task_id], args.control + '/tasks/' + task_id + '.json')
            del active[task_id]
            print(f'[exp368-server] {task_id} exit={code}', flush=True)
        transferred += mirror_committed(api, local, args.control + '/results', uploaded, 9_500_000_000 - transferred)
        stop = fs.exists(control + '/STOP.json')
        if stop and not active:
            break
        free = sorted(set(range(8)) - {item[2] for item in active.values()})
        for path in ([] if stop else sorted(fs.glob(control + '/queue/*.json'))):
            if not free:
                break
            task_id = Path(path).stem
            if task_id in active or task_id in finished:
                continue
            task = json.loads(fs.cat(path))
            source = unpack_bundle(fs, task['bundle'], task['bundle_sha256'])
            kind = task['kind']
            if kind not in {'guided', 'pool', 'score'}:
                raise ValueError('Unsupported task kind')
            for key in ['output', 'plan']:
                if '/' in task[key] or '..' in task[key]:
                    raise ValueError('Task plan/output must be a simple filename')
            output = local / task['output']
            output.mkdir(parents=True, exist_ok=True)
            plan = json.loads((source / task['plan']).read_text())
            for entry in plan['entries']:
                (output / entry).mkdir(parents=True, exist_ok=True)
            command = ['bash', 'pool_bootstrap.sh'] if kind == 'pool' else [sys.executable,
                'score_pool.py' if kind == 'score' else 'worker.py']
            command += ['--plan', task['plan'], '--out', output.as_uri(), '--shard', str(task['shard']),
                        '--shards', str(task['shards'])]
            if kind == 'score':
                command += ['--input', (local / task['input']).as_uri()]
            if task.get('smoke'):
                command += ['--smoke']
            device = free.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device), EXP368_SHARED_MODELS='1',
                       EXP368_TARGETS='/tmp/exp368-targets.parquet', EXP368_VLLM_READY='1')
            log = Path('/tmp') / (task_id + '.log')
            handle = log.open('w')
            process = subprocess.Popen(command, cwd=source, env=env, stdout=handle, stderr=subprocess.STDOUT)
            active[task_id] = (process, handle, device, task, log)
            print(f'[exp368-server] {task_id} started gpu={device}', flush=True)
        active_status = {}
        for key, item in active.items():
            with item[4].open('rb') as stream:
                stream.seek(max(0, item[4].stat().st_size - 3000))
                tail = stream.read().decode(errors='replace')
            active_status[key] = dict(gpu=item[2], task=item[3], log_tail=tail)
        api.storage.write_json(dict(active=active_status,
            finished=finished, artifact_transfer_bytes=transferred, elapsed_seconds=time.monotonic() - started),
            args.control + '/status.json')
        time.sleep(10)
    if active:
        raise TimeoutError('Server deadline reached with active tasks; inspect durable outputs')
    api.storage.write_json(dict(finished=finished, artifact_transfer_bytes=transferred), args.control + '/complete.json')


if __name__ == '__main__':
    main()
