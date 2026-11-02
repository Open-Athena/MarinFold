# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""B1 layer ablation: the contact readout with one attention or MLP output mean-ablated.

Plan section 6.1. For each of the model's decoder layers and each sublayer
(attention, MLP), replace that sublayer's output with its mean and rerun the
production pairwise readout on eval-val and eval-denovo. 2 x n_layers
conditions plus the clean model (48 + 1 for the 24-layer 1.5B models).

The mean of each sublayer is taken over every position the clean readout
processes (prefix passes and tail chunks) across all proteins and seeds of
this run, accumulated during the clean pass. It is saved, and ``--means``
points a later run at saved means, so a run on another protein set ablates
with the same vectors (B1's extensions use the eval-set means, committed as
``data/b1_sublayer_means.npz``). With ``--means`` the conditions are
independent, so ``--shard``/``--n-shards`` split them (clean included)
across instances.

The script records raw data and applies no threshold. Per condition,
protein and seed it keeps the full P(contact) matrix and R-precision in four
separation bands, so any metric, split or threshold can be computed later.
Every condition sees the same proteins and document seeds as the clean
model, so CIs come from a paired bootstrap over proteins on one run;
``--seed-offset`` draws document realizations the default readout never
uses, for repeat runs that add independent samples.

Outputs under ``--output-dir`` (one condition at a time, so a rerun resumes):

- ``sublayer_means.npz``: ``means`` ``[n_layers, 2, hidden]`` float32
  (site 0 attention, 1 MLP) and ``counts`` (positions averaged).
- ``pcontact/<condition>.npz``: per stem, ``[seeds, n_pairs]`` float32, the
  upper triangle (``np.triu_indices(L, 1)``) of each seed's P(contact).
- ``scores/<condition>.csv``: one row per protein and seed (plus seed
  ``mean``, the seed-averaged matrix): R-precision for all pairs
  (separation >= 6), short (6-11), medium (12-23) and long (>= 24).
- ``timings/<condition>.csv``: the AGENTS.md predictor-timing rows.

Conditions are ``clean`` and ``L<layer>_<site>``, e.g. ``L07_mlp``.

    uv run python b1_layer_ablation.py --device cuda --output-dir out/b1
    uv run python b1_layer_ablation.py --device cuda --protein-set legacy-extra --means data/b1_sublayer_means.npz \
        --layers 14 15 16 17 18 19 20 21 22 23 --output-dir out/b1_extra
    uv run python b1_layer_ablation.py --device cpu --limit 2 --seeds 2 --layers 0 --output-dir out/b1_smoke
"""

import argparse
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from marinfold.registry import resolve_model_entry

import harness

SITES: tuple[harness.Site, ...] = ("attn", "mlp")
BANDS = {"all": (6, None), "short": (6, 11), "medium": (12, 23), "long": (24, None)}


def condition_name(layer: int, site: harness.Site) -> str:
    return f"L{layer:02d}_{site}"


def score_rows(condition: str, protein: harness.EvalProtein, per_seed: np.ndarray, seed_offset: int) -> list[dict]:
    mean = per_seed.sum(axis=0) / len(per_seed)
    rows = []
    for seed, matrix in [*((str(seed_offset + k), m) for k, m in enumerate(per_seed)), ("mean", mean)]:
        row = {"condition": condition, "stem": protein.stem, "eval_set": protein.eval_set,
               "n_residues": protein.length, "seed": seed}
        for band, (low, high) in BANDS.items():
            row[f"rp_{band}"] = harness.r_precision(matrix, protein, min_separation=low, max_separation=high)
        rows.append(row)
    return rows


def run_condition(condition: str, proteins, backend, args, static: dict) -> pd.DataFrame:
    """Score every protein under the hooks already attached; write the condition's outputs."""
    scores, timings, matrices = [], [], {}
    for protein in proteins:
        start = time.perf_counter()
        per_seed = harness.pcontact_per_seed(backend, protein, seeds=args.seeds, seed_offset=args.seed_offset)
        elapsed = time.perf_counter() - start
        upper = np.triu_indices(protein.length, k=1)
        matrices[protein.stem] = per_seed[:, upper[0], upper[1]].astype(np.float32)
        scores.extend(score_rows(condition, protein, per_seed, args.seed_offset))
        timings.append({
            "stem": protein.stem, "n_residues": protein.length,
            "n_pairs": protein.length * (protein.length - 1) // 2, "mode": f"b1-{condition}-{args.seeds}seed",
            "elapsed_seconds": round(elapsed, 3), "total_seconds": round(time.perf_counter() - start, 3),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **static,
        })
    for sub in ("pcontact", "scores", "timings"):
        (args.output_dir / sub).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_csv(args.output_dir / "scores" / f"{condition}.csv", index=False, float_format="%.6g")
    pd.DataFrame(timings).to_csv(args.output_dir / "timings" / f"{condition}.csv", index=False)
    # Written last: its presence marks the condition done for resume.
    np.savez_compressed(args.output_dir / "pcontact" / f"{condition}.npz", **matrices)
    return pd.DataFrame(scores)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="MODELS.yaml nickname (default: registry default, exp277)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seeds", type=int, default=harness.DEFAULT_SEEDS)
    parser.add_argument("--seed-offset", type=int, default=0, help="first document seed; nonzero for repeat runs")
    parser.add_argument("--tail-batch-size", type=int, default=64, help="tails per cached forward; 64 fits an L4 at L <= 1000")
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="ablate only these layers (default all)")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N proteins (smoke test)")
    parser.add_argument("--protein-set", choices=harness.PROTEIN_SETS, default="eval")
    parser.add_argument("--max-length", type=int, default=None, help="drop proteins longer than this")
    parser.add_argument("--sample", type=int, default=None, help="seeded random subset of this many proteins")
    parser.add_argument("--means", type=Path, default=None, help="saved sublayer_means.npz to ablate with")
    parser.add_argument("--shard", type=int, default=0, help="with --means: run conditions shard::n-shards")
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--runner-tag", default="local")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds < 2 and args.seed_offset == 0:
        raise ValueError("--seeds 1 at offset 0 would differ from the production single-seed salt; use >= 2")
    if args.n_shards > 1 and args.means is None:
        raise ValueError("--n-shards needs --means: without it the clean pass computes the means every shard uses")

    entry = resolve_model_entry(args.model)
    proteins = harness.load_protein_set(args.protein_set, max_length=args.max_length, sample=args.sample)[: args.limit]
    load_start = time.perf_counter()
    backend = harness.load_model(args.model, device=args.device, dtype=args.dtype, tail_batch_size=args.tail_batch_size)
    model_load_seconds = time.perf_counter() - load_start
    n_layers = len(harness.decoder_layers(backend.model))
    layers = range(n_layers) if args.layers is None else args.layers
    static = {
        "model_nickname": entry.nickname, "model_load_seconds": round(model_load_seconds, 2),
        "runner_tag": args.runner_tag, "n_seeds": args.seeds, "seed_offset": args.seed_offset,
        "tail_batch_size": args.tail_batch_size, "dtype": args.dtype,
        "hostname": socket.gethostname(), "platform": platform.platform(), "torch_version": torch.__version__,
    }
    if args.device.startswith("cuda"):
        props = torch.cuda.get_device_properties(0)
        static |= {"gpu_name": props.name, "gpu_total_memory_gb": round(props.total_memory / 2**30, 1),
                   "gpu_compute_capability": f"{props.major}.{props.minor}"}
    print(f"model {entry.nickname}, {n_layers} layers, {len(proteins)} {args.protein_set} proteins, seeds "
          f"{args.seed_offset}..{args.seed_offset + args.seeds - 1}", flush=True)

    # The per-seed readout must be the production readout: at offset 0 its
    # seed average has to equal pairwise_score_matrix bit for bit.
    if args.seed_offset == 0:
        reference = harness.pcontact(backend, proteins[0], seeds=args.seeds)
        per_seed = harness.pcontact_per_seed(backend, proteins[0], seeds=args.seeds)
        if not np.array_equal(per_seed.sum(axis=0) / args.seeds, reference):
            raise RuntimeError("per-seed readout does not reproduce pairwise_score_matrix")
        print("per-seed readout reproduces the production readout", flush=True)
    else:
        harness.pcontact_per_seed(backend, proteins[0], seeds=1, seed_offset=args.seed_offset)  # warm-up
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()

    means_path = args.output_dir / "sublayer_means.npz"

    def done(name: str) -> bool:
        return (args.output_dir / "pcontact" / f"{name}.npz").exists()

    conditions = ["clean", *(condition_name(layer, site) for layer in layers for site in SITES)]
    conditions = conditions[args.shard :: args.n_shards]
    if args.means is not None or (done("clean") and means_path.exists()):
        saved = np.load(args.means if args.means is not None else means_path)
        means = {(k, site): torch.from_numpy(saved["means"][k, s])
                 for k in range(n_layers) for s, site in enumerate(SITES)}
        print(f"loaded sublayer means; conditions {conditions}", flush=True)
        if "clean" in conditions and not done("clean"):
            start = time.perf_counter()
            clean = run_condition("clean", proteins, backend, args, static)
            print(f"clean {time.perf_counter() - start:.0f}s rp_all {clean[clean.seed == 'mean'].rp_all.mean():.4f}",
                  flush=True)
    else:
        start = time.perf_counter()
        with harness.sublayer_means(backend.model) as (means, counts):
            clean = run_condition("clean", proteins, backend, args, static)
        counts = np.array([[counts[(k, site)] for site in SITES] for k in range(n_layers)])
        stacked = torch.stack([torch.stack([means[(k, site)] for site in SITES]) for k in range(n_layers)])
        np.savez(means_path, means=stacked.cpu().numpy(), counts=counts)
        print(f"clean {time.perf_counter() - start:.0f}s rp_all {clean[clean.seed == 'mean'].rp_all.mean():.4f}", flush=True)

    for layer in layers:
        for site in SITES:
            name = condition_name(layer, site)
            if name not in conditions or done(name):
                continue
            start = time.perf_counter()
            with harness.mean_ablation(backend.model, {(layer, site): means[(layer, site)]}):
                scores = run_condition(name, proteins, backend, args, static)
            print(f"{name} {time.perf_counter() - start:.0f}s rp_all {scores[scores.seed == 'mean'].rp_all.mean():.4f}",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
