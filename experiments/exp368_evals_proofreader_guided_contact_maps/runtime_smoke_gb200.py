"""Check an ARM/Blackwell fallback without transferring experiment weights."""

import argparse
import json
import platform
import socket
import time
from datetime import UTC, datetime

import torch
from torch import nn
from transformers import Qwen3Config, Qwen3Model

from engine import Engine, Policy
from runtime import load_release, verify_source
from test_engine import Tokenizer
from worker import verify_cache


def main() -> None:
    """Verify CUDA, cached readout and lookahead using a tiny random model."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    source = verify_source()
    if platform.machine() != 'aarch64' or not torch.cuda.is_available():
        raise RuntimeError('The GB200 smoke requires ARM and functioning CUDA')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(368)
    api = load_release()
    started = time.perf_counter()
    config = Qwen3Config(vocab_size=2010, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=512, layer_types=['full_attention'] * 2)
    reader = api.model.CausalContactProofreader(Qwen3Model(config), 2002,
        readout_width=32, readout_layers=2, readout_heads=4).to('cuda')
    engine = Engine(reader, nn.Linear(32, 2003, bias=False).to('cuda'), Tokenizer())
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    inference_start = time.perf_counter()
    prompt = [14, 26, 32, 2002]
    verification = verify_cache(engine, prompt, 24, 0)
    rng = torch.Generator(device='cuda').manual_seed(368)
    state = engine.start(prompt)
    for policy in [Policy('top10', candidates=16, fraction=.1),
                   Policy('lookahead', candidates=16, fraction=.1, lookahead=8, repeats=2)]:
        state = engine.choose(state, policy, 24, 0, rng, max_contacts=32)
    if state.count != 2 or state.tokens.shape[1] != len(prompt) + 6:
        raise ValueError('Fallback lookahead committed hypothetical future contacts')
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - inference_start
    gpu = torch.cuda.get_device_properties(0)
    worker = dict(gpu_name=gpu.name, gpu_total_memory_gb=gpu.total_memory / 1e9,
        gpu_compute_capability=f'{gpu.major}.{gpu.minor}', hostname=socket.gethostname(),
        platform=platform.platform(), torch_version=torch.__version__, cuda_version=torch.version.cuda,
        timestamp_utc=datetime.now(UTC).isoformat())
    timing = dict(stem='synthetic-runtime-smoke', n_residues=24, n_pairs=state.count,
        mode='cache-and-lookahead-runtime-check', elapsed_seconds=elapsed,
        model_load_seconds=load_seconds, total_seconds=time.perf_counter() - started,
        model_nickname='tiny-random-qwen3-proofreader', runner_tag='iris', **worker)
    result = dict(status='passed', source_sha256=source['sha256'], verification=verification,
        timing=timing, checkpoint_bytes_transferred=0,
        limitation='Synthetic runtime check only; production weights and outputs are not validated.')
    api.storage.write_csv([timing], args.out + '/timings.csv')
    api.storage.write_json(result, args.out + '/result.json')
    print('[exp368-runtime-smoke] ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
