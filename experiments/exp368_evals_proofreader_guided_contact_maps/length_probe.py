"""Test whether generator stopping limits guided contact-map coverage.

Two samplers emit exactly L and 2L distinct contacts with EOS disabled. The L
map is a prefix of the same 2L continuation, not another independent sample.
No synthetic EOS is appended: the proofreader evaluates truncated prefixes,
which were part of its training contract. Every prefix is durably committed.
"""

import argparse
import json
import platform
import socket
import time
from datetime import UTC, datetime

import torch

from engine import Engine, Policy, State
from runtime import HERE, TARGETS, load_release, pin_run
from worker import load_engine, released_score, verify_cache


def advance_to_contacts(engine: Engine, state: State, policy: Policy, length: int,
                        offset: int, rng: torch.Generator, count: int) -> State:
    """Continue a nonterminal prefix to an exact budget without sampling EOS."""
    if state.finished or count < state.count or policy.lookahead or policy.repair or policy.particles > 1:
        raise ValueError('Fixed-length probe requires an unfinished ordinary or immediate-guidance prefix')
    while state.count < count:
        state = engine.choose(state, policy, length, offset, rng, count)
    return state


def main() -> None:
    """Run a frozen length-control shard with cumulative inference timings."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    api = load_release()
    plan = json.loads((HERE / args.plan).read_text())
    pin = pin_run(api, plan, args.out, args.shard, kind='fixed-length-probe', shards=args.shards)
    targets = {row['entry_id']: row for row in api.storage.read_rows(TARGETS)}
    panel = sorted([targets[entry] for entry in plan['entries']], key=lambda row: (row['L'], row['entry_id']))
    if any(row['split'] != plan['split'] for row in panel):
        raise ValueError('Length-control panel has the wrong split')
    panel = panel[args.shard::args.shards]
    multipliers = plan['contact_budgets_L']
    if multipliers != [1, 2]:
        raise ValueError('This probe fixes exactly L and 2L contacts')
    samplers = [Policy(**row) for row in plan['samplers']]
    expected = {f'{sampler.name}-fixed{multiple}L' for sampler in samplers for multiple in multipliers}
    if {row['name'] for row in plan['policies']} != expected:
        raise ValueError('Analysis policy names differ from the length-control samplers')
    fs, root = api.storage.filesystem(args.out)
    started = time.perf_counter()
    engine = load_engine(api)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    gpu = torch.cuda.get_device_properties(0)
    worker = dict(model_nickname='exp277-step-266344+exp356-step-18000', runner_tag='iris',
        gpu_name=gpu.name, gpu_total_memory_gb=gpu.total_memory / 1e9,
        gpu_compute_capability=f'{gpu.major}.{gpu.minor}', hostname=socket.gethostname(),
        platform=platform.platform(), torch_version=torch.__version__)
    print(f'[exp368-length] loaded; proteins={len(panel)} samplers={len(samplers)}', flush=True)
    for target in panel:
        for policy in samplers:
            for sample in range(plan['rollouts']):
                keys = {multiple: f"{target['entry_id']}/{policy.name}-fixed{multiple}L-r{sample}"
                        for multiple in multipliers}
                if all(fs.exists(root + '/' + key + '.done.json') for key in keys.values()):
                    continue
                start = time.perf_counter()
                identity = f"exp368:{target['entry_id']}:r{sample}"
                prompt_text, offset = api.records.make_prompt(target['sequence'], identity)
                prompt = engine.tokenizer.encode(prompt_text, add_special_tokens=False)
                largest = max(multipliers) * target['L']
                maximum = min((8192 - len(prompt) - 1) // 3, (target['L'] - 6) * (target['L'] - 5) // 2)
                if largest > maximum:
                    raise ValueError('Requested exact contact budget exceeds context or contact universe')
                if args.smoke and sample == 0:
                    verification = verify_cache(engine, prompt, target['L'], offset)
                    api.storage.write_json(verification, args.out + '/' + keys[1] + '.verification.json')
                engine.generated_tokens = engine.scored_contacts = 0
                engine.candidate_diagnostics = []
                rng = torch.Generator(device=engine.device).manual_seed(api.records.stable_seed('sample:' + identity))
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                segment_start = time.perf_counter()
                state = engine.start(prompt)
                elapsed = 0.
                for multiple in multipliers:
                    count = multiple * target['L']
                    state = advance_to_contacts(engine, state, policy, target['L'], offset, rng, count)
                    q, recall = released_score(engine, state)
                    torch.cuda.synchronize()
                    elapsed += time.perf_counter() - segment_start
                    key = keys[multiple]
                    if not fs.exists(root + '/' + key + '.done.json'):
                        completion = state.tokens[0, len(prompt):].tolist()
                        # Labels annotate a finished prefix; neither sampler nor
                        # subsequent continuation receives these target contacts.
                        parsed = api.records.parse_contacts(engine.tokenizer.convert_ids_to_tokens(completion),
                            offset, target['L'], target['contacts'])
                        if parsed['pairs'] != state.pairs[0].tolist() or len(parsed['pairs']) != count:
                            raise ValueError('Fixed-length cache and released parser disagree')
                        name = f'{policy.name}-fixed{multiple}L'
                        row = dict(entry_id=target['entry_id'], group_id=target['group_id'], split=target['split'],
                            L=target['L'], policy=name, sample=sample, prompt_ids=prompt, completion_ids=completion,
                            finished=False, probabilities=q[0].tolist(), predicted_recall=recall[0].item(),
                            gt_count=len(target['contacts']), **parsed)
                        api.storage.write_rows([row], args.out + '/' + key + '.parquet')
                        if engine.candidate_diagnostics:
                            native = set(map(tuple, target['contacts']))
                            diagnostics = [dict(item, label=int(tuple(item['pair']) in native))
                                           for item in engine.candidate_diagnostics]
                            api.storage.write_rows(diagnostics, args.out + '/' + key + '.candidates.parquet')
                            api.storage.write_json(dict(rows=len(diagnostics), **pin),
                                                   args.out + '/' + key + '.candidates.done.json')
                        timing = dict(stem=target['entry_id'], n_residues=target['L'], n_pairs=count,
                            mode=name, sample=sample, elapsed_seconds=elapsed, model_load_seconds=load_seconds,
                            total_seconds=time.perf_counter() - start + load_seconds,
                            timing_scope='cumulative fixed-length prefix generation and intermediate/final scoring; excludes dumps',
                            generated_tokens=engine.generated_tokens, scored_contacts=engine.scored_contacts,
                            peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
                            timestamp_utc=datetime.now(UTC).isoformat(), **worker)
                        api.storage.write_csv([timing], args.out + '/' + key + '.timings.csv')
                        api.storage.write_json(pin, args.out + '/' + key + '.done.json')
                        print(f'[exp368-length] {key} contacts={count} seconds={elapsed:.2f}', flush=True)
                    del q, recall
                    torch.cuda.synchronize()
                    segment_start = time.perf_counter()
                del state
            for multiple in multipliers:
                name = f'{policy.name}-fixed{multiple}L'
                key = f"{target['entry_id']}/{name}"
                samples = [key + f'-r{sample}' for sample in range(plan['rollouts'])]
                if not all(fs.exists(root + '/' + sample + '.done.json') for sample in samples):
                    raise ValueError('Length-control group is incomplete after its sampling loop')
                rows = [row for sample in samples for row in api.storage.read_rows(args.out + '/' + sample + '.parquet')]
                api.storage.write_rows(rows, args.out + '/' + key + '.parquet')
                api.storage.write_json(dict(rollouts=len(rows), **pin), args.out + '/' + key + '.done.json')
    api.storage.write_json(dict(proteins=len(panel), **pin), args.out + f'/shard-{args.shard:03d}.complete.json')


if __name__ == '__main__':
    main()
