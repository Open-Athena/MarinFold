"""Replay saved H100 predictions on the released model in the GB200 runtime."""

import argparse
import hashlib
import io
import json
import os
import platform
import socket
import time
from datetime import UTC, datetime

import numpy as np
import pyarrow.parquet as pq
import torch

from regional_inputs import stage_regional
from runtime import HERE, load_release, verify_source
from worker import load_engine, verify_cache


def main() -> None:
    """Check frozen replay thresholds before using the alternative GPU backend."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--attention-backend', choices=['auto', 'math', 'efficient'], default='auto')
    parser.add_argument('--full-precision-reductions', action='store_true')
    args = parser.parse_args()
    plan = json.loads((HERE / 'backend_check_plan.json').read_text())
    source = verify_source()
    if sum(row['bytes'] for row in plan['references']) > plan['maximum_reference_bytes']:
        raise ValueError('Reference replay exceeds its small-artifact transfer bound')
    api = load_release()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    if args.attention_backend != 'auto':
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(args.attention_backend == 'math')
        torch.backends.cuda.enable_mem_efficient_sdp(args.attention_backend == 'efficient')
    if args.full_precision_reductions:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    setup_start = time.perf_counter()
    os.environ.update(stage_regional(api, 'US-EAST-08A'))
    engine = load_engine(api)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - setup_start
    targets = {row['entry_id']: row for row in api.storage.read_rows(os.environ['EXP368_TARGETS'])}
    gpu = torch.cuda.get_device_properties(0)
    worker = dict(model_nickname='exp356-step-18000', runner_tag='iris', gpu_name=gpu.name,
        gpu_total_memory_gb=gpu.total_memory / 1e9, gpu_compute_capability=f'{gpu.major}.{gpu.minor}',
        hostname=socket.gethostname(), platform=platform.platform(), torch_version=torch.__version__,
        cuda_version=torch.version.cuda, attention_backend=args.attention_backend,
        full_precision_reductions=args.full_precision_reductions)
    results, timings = [], []
    assessment = engine.tokenizer.convert_tokens_to_ids('<assess>')
    for reference in plan['references']:
        fs, path = api.storage.filesystem(reference['uri'])
        payload = fs.cat(path)
        if len(payload) != reference['bytes'] or hashlib.sha256(payload).hexdigest() != reference['sha256']:
            raise ValueError('H100 reference artifact changed: ' + reference['entry_id'])
        rows = pq.read_table(io.BytesIO(payload)).to_pylist()
        target = targets[reference['entry_id']]
        if target['split'] != 'validation' or {row['sample'] for row in rows} != set(range(8)) or len(rows) != 8:
            raise ValueError('Replay must use the frozen eight-rollout validation references')
        prompt, offset = api.records.make_prompt(target['sequence'], f"exp368:{target['entry_id']}:r0")
        prompt_ids = engine.tokenizer.encode(prompt, add_special_tokens=False)
        if prompt_ids != next(row['prompt_ids'] for row in rows if row['sample'] == 0):
            raise ValueError('Reference prompt differs from the released tokenizer and recipe')
        started = time.perf_counter()
        cache = verify_cache(engine, prompt_ids, target['L'], offset)
        eligible = [row for row in rows if row['contact_ends']]
        examples = [api.records.make_example(dict(row, identity=f"{row['entry_id']}:r{row['sample']}"),
                    len(row['contact_ends']), assessment) for row in eligible]
        arrays = api.records.collate(examples, engine.tokenizer.pad_token_id)
        ids = torch.as_tensor(arrays['input_ids'], device='cuda')
        mask = torch.as_tensor(arrays['token_mask'], device='cuda')
        positions = torch.as_tensor(arrays['contact_positions'], device='cuda')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            logits, recall = engine.reader(ids, mask, positions)
        actual = logits.sigmoid().tolist()
        recalls = recall.tolist()
        errors = np.concatenate([np.abs(np.asarray(actual[i][:len(row['probabilities'])])
                                       - row['probabilities']) for i, row in enumerate(eligible)])
        recall_errors = np.abs(np.asarray(recalls) - [row['predicted_recall'] for row in eligible])
        if not np.isfinite(errors).all() or not np.isfinite(recall_errors).all():
            raise ValueError('Nonfinite cross-backend prediction error')
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        result = dict(entry_id=target['entry_id'], L=target['L'], contacts=len(errors),
            max_bf16_probability_error=float(errors.max()), mean_bf16_probability_error=float(errors.mean()),
            max_bf16_recall_error=float(recall_errors.max()), **cache)
        results.append(result)
        timings.append(dict(stem=target['entry_id'], n_residues=target['L'], n_pairs=len(errors),
            mode='cross-backend-replay-and-cache-verification', elapsed_seconds=elapsed,
            model_load_seconds=load_seconds, total_seconds=load_seconds + elapsed,
            timestamp_utc=datetime.now(UTC).isoformat(), **worker))
        print('[exp368-backend-check] ' + json.dumps(result), flush=True)
    passed = all(row['max_bf16_probability_error'] <= plan['max_bf16_probability_error']
                 and row['max_bf16_recall_error'] <= plan['max_bf16_recall_error'] for row in results)
    result = dict(status='passed' if passed else 'failed', plan=plan, source_sha256=source['sha256'],
                  results=results, worker=worker, limitation='Numerical agreement, not bitwise sampler equivalence.')
    api.storage.write_json(result, args.out + '/result.json')
    api.storage.write_csv(results, args.out + '/errors.csv')
    api.storage.write_csv(timings, args.out + '/timings.csv')
    if not passed:
        raise ValueError('Cross-backend replay exceeded its predeclared thresholds')


if __name__ == '__main__':
    main()
