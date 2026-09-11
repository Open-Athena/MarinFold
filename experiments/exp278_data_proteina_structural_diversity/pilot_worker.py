"""Run the first length-by-fold sampling screen with persistent Proteina models."""

import argparse
import json
import time

import fsspec
import torch

from sample_worker import put_bytes, run_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=["short", "long"], required=True)
    parser.add_argument("--samples-per-case", type=int, default=64)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--gpu-hours", type=float, default=20)
    parser.add_argument("--only-length", type=int, action="append")
    args = parser.parse_args()
    began = time.perf_counter()
    model = None
    previous_length = None
    lengths = [60, 100, 200] if args.model == "short" else [300, 400, 500]
    batch_sizes = {60: 32, 100: 32, 200: 32, 300: 16, 400: 8, 500: 8}
    cases = []
    for length in lengths:
        for condition in ["unconditional", "1.x.x.x", "2.x.x.x", "3.x.x.x"]:
            batch_size = batch_sizes[length]
            if args.samples_per_case % batch_size:
                parser.error(
                    "Samples per case must be divisible by its fixed batch size"
                )
            case_id = f"{args.model}-l{length}-{condition}"
            case = dict(
                model=args.model,
                length=length,
                batch_size=batch_size,
                batches=args.samples_per_case // batch_size,
                seed=278000 + length * 10 + len(cases) * 10000,
                cath=condition,
                noise=0.45 if args.model == "short" else 0.35,
                compile=args.compile,
                output=f"{args.output}/{case_id}",
                control="",
            )
            cases.append(case)
    put_bytes(args.output + "/cases.json", json.dumps(cases, indent=2).encode())
    for case in cases:
        if args.only_length and case["length"] not in args.only_length:
            continue
        if time.perf_counter() - began > args.gpu_hours * 3600:
            raise RuntimeError("Pilot worker reached its GPU-hour limit")
        fs, path = fsspec.core.url_to_fs(case["output"] + "/complete.json")
        if fs.exists(path):
            print(f"Already complete: {case['output']}", flush=True)
            continue
        if args.compile and previous_length != case["length"]:
            # Static graphs specialize on length and conditioning. Release the
            # previous bucket's graphs before PyTorch's cache limit falls back
            # to eager execution. The checkpoint remains resident on the GPU.
            torch.compiler.reset()
        model = run_case(argparse.Namespace(**case), model)
        previous_length = case["length"]
    completed = []
    for case in cases:
        fs, path = fsspec.core.url_to_fs(case["output"] + "/complete.json")
        completed.append(fs.exists(path))
    if all(completed):
        put_bytes(
            args.output + "/complete.json",
            json.dumps(
                {"cases": len(cases), "candidates": len(cases) * args.samples_per_case}
            ).encode(),
        )


if __name__ == "__main__":
    main()
