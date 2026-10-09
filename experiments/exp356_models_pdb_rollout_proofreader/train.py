"""Resumable single-node, eight-H100 production training of the proofreader."""

import argparse
import contextlib
import csv
import io
import json
import math
import os
import platform
import random
import re
import shutil
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel

from model import ASSESSMENT, load_model, loss_and_metrics
from records import collate
from storage import GENERATOR, ROOT, filesystem, stage_directory, upload_directory, write_json
from training_data import epoch_order, example, fingerprint, read_table, stage_rollouts


def device_batch(example_record, pad_id: int, device: torch.device) -> dict:
    """Move a physically cropped example to this training rank."""
    return {k: torch.from_numpy(v).to(device) for k, v in collate([example_record], pad_id).items()}


def predict(model, batch: dict):
    """Run only the input tensors through the backbone, keeping targets separate."""
    return model(batch['input_ids'], batch['token_mask'], batch['contact_positions'])


def validation(model, table, tokenizer, device, rank: int, world: int, limit: int,
               timing_uri: str, model_load_seconds: float, model_nickname: str) -> dict:
    """Use deterministic held-out proteins and a fixed range of prefix lengths."""
    model.eval()
    counts = torch.zeros(7, device=device, dtype=torch.float64)
    assess = tokenizer.convert_tokens_to_ids(ASSESSMENT)
    # A fixed evenly spread sample of rows, independent of training order/crop RNG.
    indices = np.linspace(0, len(table)-1, min(limit, len(table)), dtype=int)
    gpu = torch.cuda.get_device_properties(device)
    worker = dict(model_nickname=model_nickname, runner_tag='iris', gpu_name=gpu.name,
        gpu_total_memory_gb=gpu.total_memory/1e9, gpu_compute_capability=f'{gpu.major}.{gpu.minor}',
        hostname=socket.gethostname(), platform=platform.platform(), torch_version=torch.__version__)
    timings = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for serial in range(rank, len(indices), world):
            k = (1, 4, 16, 10**6)[serial % 4]
            begun = time.perf_counter()
            row = table.slice(int(indices[serial]), 1).to_pylist()[0]
            record = example(table, int(indices[serial]), assess, 0, k)
            batch = device_batch(record, tokenizer.pad_token_id, device)
            torch.cuda.synchronize()
            infer_started = time.perf_counter()
            logits, recall = predict(model, batch)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - infer_started
            loss, metrics = loss_and_metrics(logits, recall, batch)
            values = [loss, metrics['contact_loss'], metrics['recall_mse'], metrics['precision_mae'], metrics['recall_mae'], metrics['brier']]
            counts[:6] += torch.stack(values).double()
            counts[6] += 1
            timings.append(dict(stem=row['entry_id'], identity=row['identity'], n_residues=row['L'],
                n_pairs=len(record.labels), mode=f'validation-prefix-{k}', elapsed_seconds=elapsed,
                model_load_seconds=model_load_seconds, total_seconds=time.perf_counter()-begun+model_load_seconds,
                timestamp_utc=datetime.now(UTC).isoformat(), **worker))
    if timings:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(timings[0]))
        writer.writeheader()
        writer.writerows(timings)
        fs, key = filesystem(timing_uri+f'/rank-{rank}.csv')
        fs.pipe(key, output.getvalue().encode())
    if world > 1:
        dist.all_reduce(counts)
    model.train()
    names = ('loss', 'contact_loss', 'recall_mse', 'precision_mae', 'recall_mae', 'brier')
    return {name: float(counts[i]/counts[6]) for i, name in enumerate(names)}


def schedule_config(args) -> dict:
    """Identify settings that must stay fixed when replaying a saved optimizer cursor."""
    return {key:getattr(args,key) for key in ['epochs','accumulation','learning_rate',
        'head_learning_rate','warmup_steps','max_steps']}


def prune_checkpoints(root: str, keep: int, best: str | None) -> list[str]:
    """Keep recent recovery states and the validation winner within this run only."""
    fs,path = filesystem(root)
    checkpoints = []
    for item in fs.ls(path,detail=False):
        match = re.fullmatch(r'step-(\d+)',Path(item.rstrip('/')).name)
        if match:
            checkpoints.append((int(match[1]),item))
    retained = {p for _,p in sorted(checkpoints)[-keep:]}
    if best:
        _,best_path = filesystem(best)
        retained.add(best_path)
    removed = []
    for _,item in checkpoints:
        if item not in retained:
            fs.rm(item,recursive=True)
            removed.append(item)
    return removed


def checkpoint(model, tokenizer, optimizer, args, step: int, epoch: int, next_batch: int,
               best_loss: float, improved: bool, data_hash: str, local_root: Path, rank: int, world: int) -> None:
    """Commit model and optimizer first, then atomically publish the resume pointer."""
    if world > 1:
        dist.barrier()
    if rank == 0:
        path = local_root / f'step-{step}'
        metadata = dict(architecture='exp277-bidirectional-contact-recall-v1', step=step,
            epoch=epoch, next_batch=next_batch, data_fingerprint=data_hash,
            generator=GENERATOR, assessment_token=ASSESSMENT, attention='bidirectional',
            precision='mean unique-contact probabilities', config=vars(args), best_validation_loss=best_loss,
            code=json.loads(Path('code_manifest.json').read_text()))
        model.save(path, tokenizer, metadata)
        torch.save(dict(optimizer=optimizer.state_dict(), step=step, epoch=epoch,
            next_batch=next_batch, best_loss=best_loss, data_fingerprint=data_hash,
            accumulation=args.accumulation, world_size=world,schedule=schedule_config(args)), path / 'training_state.pt')
        remote = f'{args.out}/checkpoints/{args.run_name}/step-{step}'
        files = upload_directory(path, remote)
        write_json(dict(files=files, **metadata), remote + '/manifest.json')
        best_pointer = f'{args.out}/runs/{args.run_name}/best.json'
        if improved:
            write_json(dict(step=step,path=remote,validation_loss=best_loss),best_pointer)
        write_json(dict(path=remote, step=step), f'{args.out}/runs/{args.run_name}/resume.json')
        print(f'[exp356] CHECKPOINT {remote}', flush=True)
        shutil.rmtree(path)
        fs,key = filesystem(best_pointer)
        best = json.loads(fs.cat(key))['path'] if fs.exists(key) else None
        removed = prune_checkpoints(f'{args.out}/checkpoints/{args.run_name}',args.keep_checkpoints,best)
        if removed:
            print(f'[exp356] retired {len(removed)} older recovery checkpoints',flush=True)
    if world > 1:
        dist.barrier()


def main() -> None:
    """Train with explicit exact-step recovery and validation-selected checkpoints."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=ROOT + '/data/v1/rollouts')
    parser.add_argument('--out', default=ROOT)
    parser.add_argument('--generator', default=GENERATOR)
    parser.add_argument('--run-name', default='exp356-exp277-bidir-pdb50k-v1')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--head-learning-rate', type=float, default=2e-4)
    parser.add_argument('--warmup-steps', type=int, default=200)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--keep-checkpoints',type=int,default=3)
    parser.add_argument('--validate-every', type=int, default=500)
    parser.add_argument('--validation-examples', type=int, default=256)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--stop-after-step', type=int, help='Save and pause for an intentional recovery test')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.keep_checkpoints < 2:
        parser.error('Keep at least two recovery checkpoints')
    setup_started = time.perf_counter()
    rank = int(os.getenv('RANK', '0'))
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world = int(os.getenv('WORLD_SIZE', '1'))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(356)
    np.random.seed(356)
    torch.manual_seed(356)
    local = Path('/tmp/exp356-training')
    model_path, data_path = local / 'model', local / 'data'
    resume_path = None
    if rank == 0:
        if args.resume:
            fs, pointer = filesystem(f'{args.out}/runs/{args.run_name}/resume.json')
            if fs.exists(pointer):
                resume_path = json.loads(fs.cat(pointer))['path']
        stage_directory(resume_path or args.generator, model_path)
        stage_rollouts(args.data, data_path)
        (local / 'resume-source.json').write_text(json.dumps(resume_path))
    if world > 1:
        dist.barrier()
    resume_path = json.loads((local / 'resume-source.json').read_text())
    manifest = json.loads((data_path / 'manifest.json').read_text())
    data_hash = fingerprint(manifest)
    train_table = read_table(data_path, 'train')
    validation_table = read_table(data_path, 'validation')
    model, tokenizer = load_model(model_path, initialize=resume_path is None)
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.to(device)
    optimizer = torch.optim.AdamW([
        dict(params=model.backbone.parameters(), lr=args.learning_rate, initial_lr=args.learning_rate),
        dict(params=list(model.contact_head.parameters())+list(model.recall_head.parameters()),
             lr=args.head_learning_rate, initial_lr=args.head_learning_rate)], weight_decay=0.01, fused=True)
    step, start_epoch, next_batch, best_loss = 0, 0, 0, float('inf')
    if resume_path:
        state = torch.load(model_path / 'training_state.pt', map_location=device, weights_only=False)
        if state['data_fingerprint'] != data_hash or state['world_size'] != world or state['accumulation'] != args.accumulation:
            raise ValueError('Resume data or global batch changed')
        if state['schedule'] != schedule_config(args):
            raise ValueError('Optimizer or learning-rate schedule changed on resume')
        optimizer.load_state_dict(state['optimizer'])
        step, start_epoch, next_batch, best_loss = state['step'], state['epoch'], state['next_batch'], state['best_loss']
        del state
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else model
    steps_per_epoch = math.ceil(len(train_table)/(world*args.accumulation))
    max_steps = args.max_steps or steps_per_epoch*args.epochs
    run = None
    if rank == 0:
        run = wandb.init(entity='open-athena', project='MarinFold', id=args.run_name,
            name=args.run_name, resume='allow', config={**vars(args), 'data_fingerprint':data_hash,
                'code':json.loads(Path('code_manifest.json').read_text()),
                'world_size':world, 'train_rollouts':len(train_table), 'steps_per_epoch':steps_per_epoch,
                'max_steps':max_steps}, tags=['exp356', 'proofreader', 'experimental-pdb', 'bidirectional'])
        run.define_metric('optimizer_step')
        run.define_metric('train/*',step_metric='optimizer_step')
        run.define_metric('validation/*',step_metric='optimizer_step')
        write_json(dict(wandb_url=run.url, wandb_name=args.run_name, job_id=os.getenv('IRIS_JOB_ID'),
            max_steps=max_steps, started_at=datetime.now(UTC).isoformat()), f'{args.out}/runs/{args.run_name}/started.json')
        print(f'[exp356] WANDB {run.url} max_steps={max_steps}', flush=True)
    assess = tokenizer.convert_tokens_to_ids(ASSESSMENT)
    load_seconds = time.perf_counter()-setup_started
    initial = validation(model, validation_table, tokenizer, device, rank, world, args.validation_examples,
        f'{args.out}/runs/{args.run_name}/validation-timings/step-{step}', load_seconds, args.run_name)
    if rank == 0:
        run.log({'optimizer_step':step,**{f'validation/{k}':v for k,v in initial.items()}})
        print(f'[exp356] validation step={step} {json.dumps(initial)}', flush=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    training_started = time.perf_counter()
    stop = False
    for epoch in range(start_epoch, args.epochs):
        order = epoch_order(len(train_table), epoch, world, args.accumulation)
        begin = next_batch if epoch == start_epoch else 0
        for batch_index in range(begin, len(order)):
            if step >= max_steps:
                stop = True
                break
            factor = min(1.0, (step+1)/max(1,args.warmup_steps))
            progress = max(0.0, (step-args.warmup_steps)/max(1,max_steps-args.warmup_steps))
            factor *= 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi*min(1.0, progress)))
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] * factor
            metrics_sum = torch.zeros(6, device=device)
            tokens = torch.zeros((), device=device)
            for micro in range(args.accumulation):
                index = int(order[batch_index, micro, rank])
                record = example(train_table, index, assess, epoch)
                batch = device_batch(record, tokenizer.pad_token_id, device)
                sync = wrapped.no_sync() if world > 1 and micro < args.accumulation-1 else contextlib.nullcontext()
                with sync, torch.autocast('cuda', dtype=torch.bfloat16):
                    logits, recall = predict(wrapped, batch)
                    loss, metrics = loss_and_metrics(logits, recall, batch)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f'Nonfinite loss: step {step}, row {index}')
                    scaled = loss / args.accumulation
                scaled.backward()
                metrics_sum += torch.stack([loss.detach(), *metrics.values()]) / args.accumulation
                tokens += batch['token_mask'].sum()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            next_epoch, next_index = (epoch+1, 0) if batch_index+1 == len(order) else (epoch, batch_index+1)
            if step % 10 == 0 or step == 1:
                if world > 1:
                    dist.all_reduce(metrics_sum)
                    dist.all_reduce(tokens)
                if rank == 0:
                    names = ['loss', 'contact_loss', 'recall_mse', 'precision_mae', 'recall_mae', 'brier']
                    logged = {f'train/{k}':float(v/world) for k,v in zip(names,metrics_sum,strict=True)}
                    logged.update(step=step, epoch=epoch+batch_index/len(order), learning_rate=optimizer.param_groups[0]['lr'],
                        grad_norm=float(grad_norm), tokens_this_step=float(tokens), elapsed_seconds=time.perf_counter()-training_started)
                    run.log(dict(optimizer_step=step,**logged))
                    print(f'[exp356] TRAIN {step}/{max_steps} {json.dumps(logged)}', flush=True)
                    write_json(logged, f'{args.out}/runs/{args.run_name}/progress.json')
            improved = False
            if step % args.validate_every == 0 or step == max_steps:
                measured = validation(model, validation_table, tokenizer, device, rank, world, args.validation_examples,
                    f'{args.out}/runs/{args.run_name}/validation-timings/step-{step}', load_seconds, args.run_name)
                improved = measured['loss'] < best_loss
                if improved:
                    best_loss = measured['loss']
                if rank == 0:
                    run.log({'optimizer_step':step,**{f'validation/{k}':v for k,v in measured.items()}})
                    write_json(dict(step=step, **measured), f'{args.out}/runs/{args.run_name}/validation-step-{step}.json')
                    print(f'[exp356] validation step={step} {json.dumps(measured)}', flush=True)
            if improved or step % args.checkpoint_every == 0 or step == max_steps or step == args.stop_after_step:
                checkpoint(model, tokenizer, optimizer, args, step, next_epoch, next_index, best_loss, improved, data_hash, local/'checkpoints', rank, world)
            if step == max_steps or step == args.stop_after_step:
                stop = True
                break
        if stop:
            break
    if rank == 0:
        status_file = 'complete.json' if step == max_steps else 'paused.json'
        write_json(dict(step=step, max_steps=max_steps, best_validation_loss=best_loss,
            completed_at=datetime.now(UTC).isoformat()), f'{args.out}/runs/{args.run_name}/{status_file}')
        run.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
