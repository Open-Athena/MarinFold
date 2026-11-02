# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""B1 tuned lens: the contact readout decoded from every layer boundary.

Plan section 6.1. The production readout takes two next-token distributions
over the protein's position tokens: lp1 at the ``<contact>`` query and lp2
at each ``<contact> <p_i>`` query. Here those query residuals are taken at
every layer boundary (0 = embeddings, k = output of decoder layer k - 1, 24
= the final residual) and decoded two ways:

- **Logit lens**: the model's own final norm and unembedding applied to the
  boundary-k residual as is.
- **Tuned lens**: a per-boundary affine map first, fitted by ridge
  regression from the boundary-k residual to the final residual, both
  RMS-normalized (the final norm is scale-invariant, so only the direction
  of the final residual reaches the logits). At boundary 24 the tuned lens
  is the identity, so both lenses are the model.

Each lens gives an lp1 and lp2 at every boundary, hence a P(contact) matrix
and an R-precision per boundary, scored exactly as the production readout.

The maps are fitted on the legacy-554 proteins in
``data/a2_legacy_gt.jsonl.gz``, 4 document seeds each, minus any that are
the same PDB chain as an exp245 eval protein (6 ``denovo_pdb`` chains are
eval-denovo proteins), and evaluated on ``--protein-sets`` (see
``harness.load_protein_set``). 20% of the fit proteins are held out to pick
each boundary's ridge strength from ``RIDGE_GRID`` (times the mean diagonal
of XᵀX) by held-out MSE; the maps are then refitted on all of them. A run
finding ``lens.npz`` in ``--output-dir`` uses it instead of fitting, so
shards over a large protein set share one fit.

The script records raw data and applies no threshold. Outputs under
``--output-dir``:

- ``lens.npz``: ``weights`` ``[24, hidden + 1, hidden]`` float32 (boundary
  0..23; the last input row is the bias), ``ridge``.
- ``lens_fit.csv``: per boundary and ridge strength, held-out MSE and R² of
  the tuned map, with the logit lens (identity map) as a reference row.
- ``pcontact/<stem>.npz``: ``pcontact`` ``[2, 25, seeds, n_pairs]`` float32
  (lens 0 logit, 1 tuned; upper triangle ``np.triu_indices(L, 1)``) and
  ``production`` ``[seeds, n_pairs]``, the model's own readout. Written
  last; it marks the protein done for resume.
- ``scores/<stem>.csv``: R-precision per lens, boundary and seed (plus seed
  ``mean``, the seed-averaged matrix) in the four B1 bands.
- ``timings/<stem>.csv``: the AGENTS.md predictor-timing rows, plus the
  boundary-24 check: ``final_max_abs_dp``, the largest difference between
  the boundary-24 decode and the production readout, and both seed-averaged
  all-pairs R-precisions (``final_rp``, ``production_rp``). The run stops
  if those differ by more than ``FINAL_RP_TOLERANCE``: the production
  readout's bf16 logits round differently from the float32 decode, so the
  matrices are not equal entrywise and a small protein's R-precision can
  move by one contact (1/17 for a 30-residue chain). The tolerance catches
  decoding bugs (wrong rows or token ids give near-random R-precision);
  agreement over all proteins is checked in the analysis.

    uv run python b1_tuned_lens.py --device cuda --protein-sets eval legacy-extra --output-dir out/lens
    uv run python b1_tuned_lens.py --device cuda --protein-sets panel-long --max-length 1000 \
        --shard 0 --n-shards 3 --output-dir out/lens_panel   # with lens.npz copied in
    uv run python b1_tuned_lens.py --device cpu --limit 2 --fit-limit 5 --seeds 2 --output-dir out/lens_smoke
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
from marinfold.document_structures.contacts_v1.inference import (
    _PROB_FLOOR,
    fwd_matrix,
    pcontact_from_fwd,
    prefix_and_positions,
)
from marinfold.document_structures.contacts_v1.vocab import position_token
from marinfold.registry import resolve_model_entry

import harness
from b1_layer_ablation import BANDS

DATA = Path(__file__).resolve().parent / "data"
FIT_SET = DATA / "a2_legacy_gt.jsonl.gz"
RIDGE_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
HOLDOUT = 0.2
SEED = 341
LENSES = ("logit", "tuned")
FINAL_RP_TOLERANCE = 0.2


def rms_normalize(hidden: torch.Tensor, eps: float) -> torch.Tensor:
    """``hidden / rms(hidden)`` in float32: the final norm without its weight."""
    hidden = hidden.float()
    return hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + eps)


def features(hidden: torch.Tensor, eps: float) -> torch.Tensor:
    """Tuned-lens input: the RMS-normalized residual with a constant 1 appended."""
    normed = rms_normalize(hidden, eps)
    return torch.cat([normed, torch.ones_like(normed[:, :1])], dim=1)


def query_pass(backend, protein: harness.EvalProtein, seed: int) -> tuple[list[torch.Tensor], np.ndarray, list[int]]:
    """One document realization: per-boundary query residuals ``[L + 1, hidden]``, production fwd, position ids."""
    built = prefix_and_positions(protein.structure, entry_id=f"{protein.stem}#cv1ens{seed}")
    if built is None:
        raise ValueError(f"{protein.stem}: contacts-v1 could not serialize this chain")
    prefix, seq_positions, length = built
    with harness.query_residuals(backend.model) as captured:
        fwd = fwd_matrix(backend, prefix, seq_positions)
    residuals = [torch.cat(calls) for calls in captured]
    if residuals[0].shape[0] != length + 1:
        raise RuntimeError(f"{protein.stem}: captured {residuals[0].shape[0]} queries, expected {length + 1}")
    pos_ids = backend.tokenizer.convert_tokens_to_ids([position_token(p) for p in seq_positions])
    return residuals, fwd, pos_ids


def decode(model, residual: torch.Tensor, pos_ids: torch.Tensor) -> np.ndarray:
    """Final norm, unembedding and softmax over the full vocabulary, in float32; probabilities of ``pos_ids``.

    The model's own readout rounds its logits to the model dtype; in bf16,
    logits near 20 are spaced 0.125 apart, which moves probabilities by up
    to ~12%. Decoding in float32 keeps that rounding out of the lens.
    """
    with torch.inference_mode():
        normed = model.model.norm(residual.float())
        logits = normed @ model.lm_head.weight.float().T
        probs = torch.softmax(logits, dim=-1).index_select(-1, pos_ids)
    return probs.cpu().numpy().astype(np.float64)


def pcontact_from_probs(probs: np.ndarray) -> np.ndarray:
    """P(contact) from ``[L + 1, L]`` query probabilities (row 0 lp1, rows 1.. lp2), as ``fwd_matrix`` builds it."""
    logp = np.log(np.clip(probs, _PROB_FLOOR, None))
    return pcontact_from_fwd(logp[0][:, None] + logp[1:])


def lens_pcontact(model, residuals: list[torch.Tensor], weights: torch.Tensor, pos_ids: list[int]) -> np.ndarray:
    """``[2, n_boundaries, L, L]`` P(contact) under the logit and tuned lens at every boundary."""
    eps = model.model.norm.variance_epsilon
    ids = torch.tensor(pos_ids, device=residuals[0].device)
    final = len(residuals) - 1
    out = []
    for lens in LENSES:
        per_boundary = []
        for k, residual in enumerate(residuals):
            if lens == "tuned" and k < final:
                residual = features(residual, eps) @ weights[k]
            per_boundary.append(pcontact_from_probs(decode(model, residual, ids)))
        out.append(np.stack(per_boundary))
    return np.stack(out)


def accumulate(stats: dict, residuals: list[torch.Tensor], eps: float) -> None:
    """Add one realization's queries to the normal equations of every boundary below the last."""
    target = rms_normalize(residuals[-1], eps)
    for k, residual in enumerate(residuals[:-1]):
        x = features(residual, eps)
        stats["xtx"][k] += (x.T @ x).double()
        stats["xty"][k] += (x.T @ target).double()
    stats["yty"] += float((target.double() ** 2).sum())
    stats["n"] += target.shape[0]


def new_stats(n_boundaries: int, hidden: int, device: str) -> dict:
    return {
        "xtx": torch.zeros(n_boundaries, hidden + 1, hidden + 1, dtype=torch.float64, device=device),
        "xty": torch.zeros(n_boundaries, hidden + 1, hidden, dtype=torch.float64, device=device),
        "yty": 0.0, "n": 0,
    }


def ridge_solve(xtx: torch.Tensor, xty: torch.Tensor, ratio: float) -> torch.Tensor:
    """Ridge weights; the penalty is ``ratio`` times the mean feature diagonal and spares the bias."""
    hidden = xtx.shape[0] - 1
    penalty = torch.full((hidden + 1,), ratio * float(xtx.diagonal()[:hidden].mean()), dtype=xtx.dtype,
                         device=xtx.device)
    penalty[-1] = 0.0
    return torch.linalg.solve(xtx + torch.diag(penalty), xty)


def mse(weights: torch.Tensor, xtx: torch.Tensor, xty: torch.Tensor, yty: float, n: int) -> float:
    """Mean squared error of ``x @ weights`` against the target, from the normal-equation sums."""
    sse = yty - 2 * float((weights * xty).sum()) + float((weights * (xtx @ weights)).sum())
    return sse / (n * weights.shape[1])


def fit_lens(backend, proteins, args, hidden: int, n_boundaries: int) -> tuple[torch.Tensor, pd.DataFrame]:
    """Ridge-fit one affine map per boundary below the last; pick each strength on held-out proteins."""
    eps = backend.model.model.norm.variance_epsilon
    order = np.random.default_rng(SEED).permutation(len(proteins))
    n_val = max(1, int(round(HOLDOUT * len(proteins))))
    held_out = set(order[:n_val].tolist())
    stats = {"train": new_stats(n_boundaries, hidden, args.device), "val": new_stats(n_boundaries, hidden, args.device)}
    start = time.perf_counter()
    for index, protein in enumerate(proteins):
        part = "val" if index in held_out else "train"
        for seed in range(args.seeds):
            residuals, _, _ = query_pass(backend, protein, seed)
            accumulate(stats[part], residuals, eps)
    print(f"fit pass {time.perf_counter() - start:.0f}s: {stats['train']['n']} train, {stats['val']['n']} held-out queries",
          flush=True)

    train, val = stats["train"], stats["val"]
    identity = torch.cat([torch.eye(hidden, dtype=torch.float64), torch.zeros(1, hidden, dtype=torch.float64)])
    identity = identity.to(args.device)
    rows, chosen = [], []
    for k in range(n_boundaries):
        mean_target = val["xty"][k][-1] / val["n"]
        total = val["yty"] / (val["n"] * hidden) - float((mean_target ** 2).mean())
        logit = mse(identity, val["xtx"][k], val["xty"][k], val["yty"], val["n"])
        rows.append({"boundary": k, "lens": "logit", "ridge": np.nan, "heldout_mse": logit, "heldout_r2": 1 - logit / total})
        best = None
        for ratio in RIDGE_GRID:
            weights = ridge_solve(train["xtx"][k], train["xty"][k], ratio)
            error = mse(weights, val["xtx"][k], val["xty"][k], val["yty"], val["n"])
            rows.append({"boundary": k, "lens": "tuned", "ridge": ratio, "heldout_mse": error, "heldout_r2": 1 - error / total})
            if best is None or error < best[1]:
                best = (ratio, error)
        chosen.append(best[0])
    weights = torch.stack([
        ridge_solve(train["xtx"][k] + val["xtx"][k], train["xty"][k] + val["xty"][k], chosen[k]).float()
        for k in range(n_boundaries)
    ])
    fit = pd.DataFrame(rows)
    fit["chosen"] = [row["lens"] == "tuned" and row["ridge"] == chosen[row["boundary"]] for row in rows]
    return weights, fit


def score_rows(protein: harness.EvalProtein, pcontact: np.ndarray) -> list[dict]:
    """R-precision rows for ``pcontact`` ``[2, boundaries, seeds, L, L]``."""
    rows = []
    seeds = pcontact.shape[2]
    for li, lens in enumerate(LENSES):
        for k in range(pcontact.shape[1]):
            matrices = [*((str(s), pcontact[li, k, s]) for s in range(seeds)),
                        ("mean", pcontact[li, k].sum(axis=0) / seeds)]
            for seed, matrix in matrices:
                row = {"lens": lens, "boundary": k, "stem": protein.stem, "eval_set": protein.eval_set,
                       "n_residues": protein.length, "seed": seed}
                for band, (low, high) in BANDS.items():
                    row[f"rp_{band}"] = harness.r_precision(matrix, protein, min_separation=low, max_separation=high)
                rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="MODELS.yaml nickname (default: registry default, exp277)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seeds", type=int, default=harness.DEFAULT_SEEDS)
    parser.add_argument("--tail-batch-size", type=int, default=64)
    parser.add_argument("--protein-sets", nargs="+", choices=harness.PROTEIN_SETS, default=["eval"])
    parser.add_argument("--max-length", type=int, default=None, help="drop proteins longer than this")
    parser.add_argument("--shard", type=int, default=0, help="evaluate proteins shard::n-shards")
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N eval proteins (smoke test)")
    parser.add_argument("--fit-limit", type=int, default=None, help="fit on only the first N fit proteins (smoke test)")
    parser.add_argument("--runner-tag", default="local")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds < 2:
        raise ValueError("--seeds 1 would differ from the production single-seed salt; use >= 2")

    entry = resolve_model_entry(args.model)
    proteins = [p for name in args.protein_sets
                for p in harness.load_protein_set(name, max_length=args.max_length)]
    proteins = proteins[args.shard :: args.n_shards][: args.limit]
    eval_chains = {harness.base_stem(p.stem) for p in harness.load_eval_proteins()}
    fit_proteins = [p for p in harness.load_legacy_proteins(FIT_SET) if harness.base_stem(p.stem) not in eval_chains]
    fit_proteins = fit_proteins[: args.fit_limit]
    shared = {harness.base_stem(p.stem) for p in proteins} & {harness.base_stem(p.stem) for p in fit_proteins}
    if shared:
        raise ValueError(f"fit and evaluated proteins share chains: {sorted(shared)[:5]}")
    load_start = time.perf_counter()
    backend = harness.load_model(args.model, device=args.device, dtype=args.dtype, tail_batch_size=args.tail_batch_size)
    model_load_seconds = time.perf_counter() - load_start
    hidden = backend.model.config.hidden_size
    n_boundaries = len(harness.decoder_layers(backend.model))
    static = {
        "model_nickname": entry.nickname, "model_load_seconds": round(model_load_seconds, 2),
        "runner_tag": args.runner_tag, "n_seeds": args.seeds, "tail_batch_size": args.tail_batch_size,
        "dtype": args.dtype, "hostname": socket.gethostname(), "platform": platform.platform(),
        "torch_version": torch.__version__,
    }
    if args.device.startswith("cuda"):
        props = torch.cuda.get_device_properties(0)
        static |= {"gpu_name": props.name, "gpu_total_memory_gb": round(props.total_memory / 2**30, 1),
                   "gpu_compute_capability": f"{props.major}.{props.minor}"}
    print(f"model {entry.nickname}, {n_boundaries} layers, {len(fit_proteins)} fit and {len(proteins)} evaluated "
          f"proteins ({' + '.join(args.protein_sets)}, shard {args.shard}/{args.n_shards}), "
          f"{args.seeds} seeds", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lens_path = args.output_dir / "lens.npz"
    if lens_path.exists():
        saved = np.load(lens_path)
        weights = torch.from_numpy(saved["weights"]).to(args.device)
        print("loaded lens", flush=True)
    else:
        weights, fit = fit_lens(backend, fit_proteins, args, hidden, n_boundaries)
        fit.to_csv(args.output_dir / "lens_fit.csv", index=False, float_format="%.6g")
        np.savez(lens_path, weights=weights.cpu().numpy(), ridge=fit[fit.chosen].ridge.to_numpy())
        print(fit[fit.chosen | (fit.lens == "logit")].pivot(index="boundary", columns="lens", values="heldout_r2")
              .round(3).T.to_string(), flush=True)

    for sub in ("pcontact", "scores", "timings"):
        (args.output_dir / sub).mkdir(exist_ok=True)
    for protein in proteins:
        done = args.output_dir / "pcontact" / f"{protein.stem}.npz"
        if done.exists():
            continue
        start = time.perf_counter()
        lens, production = [], []
        for seed in range(args.seeds):
            residuals, fwd, pos_ids = query_pass(backend, protein, seed)
            lens.append(lens_pcontact(backend.model, residuals, weights, pos_ids))
            production.append(pcontact_from_fwd(fwd))
        pcontact = np.stack(lens, axis=2)  # [2, boundaries, seeds, L, L]
        production = np.stack(production)
        final_max_abs_dp = float(np.abs(pcontact[0, -1] - production).max())
        final_rp = harness.r_precision(pcontact[0, -1].sum(axis=0) / args.seeds, protein)
        production_rp = harness.r_precision(production.sum(axis=0) / args.seeds, protein)
        if abs(final_rp - production_rp) > FINAL_RP_TOLERANCE:
            raise RuntimeError(f"{protein.stem}: boundary-{n_boundaries} decode R-precision {final_rp:.3f}, "
                               f"production {production_rp:.3f}")
        elapsed = time.perf_counter() - start
        upper = np.triu_indices(protein.length, k=1)
        pd.DataFrame(score_rows(protein, pcontact)).to_csv(
            args.output_dir / "scores" / f"{protein.stem}.csv", index=False, float_format="%.6g")
        pd.DataFrame([{
            "stem": protein.stem, "n_residues": protein.length,
            "n_pairs": protein.length * (protein.length - 1) // 2, "mode": f"b1-lens-{args.seeds}seed",
            "elapsed_seconds": round(elapsed, 3), "total_seconds": round(time.perf_counter() - start, 3),
            "final_max_abs_dp": final_max_abs_dp, "final_rp": final_rp, "production_rp": production_rp,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **static,
        }]).to_csv(args.output_dir / "timings" / f"{protein.stem}.csv", index=False)
        np.savez_compressed(done, pcontact=pcontact[..., upper[0], upper[1]].astype(np.float32),
                            production=production[:, upper[0], upper[1]].astype(np.float32))
        print(f"{protein.stem} L={protein.length} {elapsed:.0f}s final rp {final_rp:.3f} production {production_rp:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
