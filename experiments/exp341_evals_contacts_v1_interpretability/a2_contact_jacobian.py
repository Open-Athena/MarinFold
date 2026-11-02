# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""A2: contact Jacobian by substitution, and its locality score (plan section 5.2).

For each protein in ``data/a2_proteins.csv``, mutate every residue k to four
representative amino acids (Ala, Gly, Pro, and a charge flip: K/R/H -> E,
D/E -> K, anything else -> D), rerun the production pairwise readout, and
record the change in P(contact). A substitution identical to the wild type
is skipped.

A mutant keeps its parent's ``entry_id``, so ``build_document`` draws the
same statement order and position offset for it: the two prefixes differ in
the one ``<AA>`` token at residue k, and ΔP carries no shuffle noise. The
script asserts this for every protein's first mutant.

Storing all of ΔP would be 4L x L x L values per protein, so the instance
reduces it. The predicted contacts are the top-L pairs by clean P(contact)
at separation >= 6. For each, the script keeps ΔP under every mutation
(``dp_top``, ``[L, 4, L]`` float16, NaN for skipped substitutions), and the
locality score

    locality(i, j) = sum over k in N(i) ∪ N(j) of sum_s |ΔP_ks(i, j)|
                     / sum over all k of sum_s |ΔP_ks(i, j)|

with N(x) the residues within 3 of x. ``null_locality`` is |N(i) ∪ N(j)| / L,
the share a mutation effect spread evenly over the chain would give.
Coevolution-like coupling predicts locality well above that; recall of a
motif or fold predicts dependence spread over the motif or chain.

Outputs under ``--output-dir`` (one protein at a time, so a rerun resumes):

- ``proteins/<key>.npz``: ``top_pairs``, ``p_clean_top``, ``dp_top``,
  ``mean_abs_dp`` (``[L, 4]``, mean |ΔP| over all separation >= 6 pairs).
- ``contacts/<key>.csv``: one row per predicted contact: ``i``, ``j``,
  separation, clean P, whether it is a true contact, ``locality``,
  ``null_locality``.
- ``timings/<key>.csv``: the AGENTS.md predictor-timing row.

Shards take every protein whose greedy, cost-balanced assignment
(``est_l40s_seconds``, longest first) matches ``--shard`` of ``--n-shards``.

    uv run python a2_contact_jacobian.py --device cuda --shard 0 --n-shards 6 --output-dir out/a2
"""

import argparse
import dataclasses
import heapq
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from marinfold.document_structures.contacts_v1.inference import gt_contact_matrix, prefix_and_positions
from marinfold.document_structures.contacts_v1.parse import ONE_LETTER_TO_THREE
from marinfold.registry import resolve_model_entry

import harness
from score_panel import BatchSizedBackends
from time_readout import gpu_columns

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
THREE_TO_ONE = {three: one for one, three in ONE_LETTER_TO_THREE.items()}
NEIGHBOURHOOD = 3
SUBSTITUTIONS = ("A", "G", "P", "flip")


def substitutes(wild_type: str) -> list[str | None]:
    """The four substitutions for a residue, None where one equals the wild type."""
    flip = "E" if wild_type in "KRH" else "K" if wild_type in "DE" else "D"
    return [aa if aa != wild_type else None for aa in ("A", "G", "P", flip)]


def mutant(protein: harness.EvalProtein, k: int, aa: str) -> harness.EvalProtein:
    residues = list(protein.structure.residues)
    residues[k] = dataclasses.replace(residues[k], resname=ONE_LETTER_TO_THREE[aa])
    structure = dataclasses.replace(protein.structure, residues=tuple(residues))
    return dataclasses.replace(protein, structure=structure)


def locality_scores(dp_top: np.ndarray, top_i: np.ndarray, top_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per contact: share of total |ΔP| from residues within NEIGHBOURHOOD of i or j, and its null.

    ``dp_top`` is ``[L residues, substitutions, contacts]`` with NaN for
    skipped substitutions. Locality is NaN for a contact no mutation moved.
    """
    length = dp_top.shape[0]
    effect = np.nansum(np.abs(dp_top), axis=1)  # [L residues, contacts]
    positions = np.arange(length)[:, None]
    near = (np.abs(positions - top_i[None, :]) <= NEIGHBOURHOOD) | (np.abs(positions - top_j[None, :]) <= NEIGHBOURHOOD)
    total = effect.sum(axis=0)
    local = (effect * near).sum(axis=0)
    locality = np.divide(local, total, out=np.full_like(total, np.nan), where=total > 0)
    return locality, near.sum(axis=0) / length


def assign_shards(proteins: pd.DataFrame, n_shards: int) -> pd.Series:
    """Longest-first greedy assignment to the least-loaded shard."""
    loads = [(0.0, shard) for shard in range(n_shards)]
    assignment = {}
    for key, cost in proteins.sort_values("est_l40s_seconds", ascending=False)[["key", "est_l40s_seconds"]].itertuples(
        index=False
    ):
        load, shard = heapq.heappop(loads)
        assignment[key] = shard
        heapq.heappush(loads, (load + cost, shard))
    return proteins["key"].map(assignment)


def check_one_token_change(backend, protein: harness.EvalProtein, changed: harness.EvalProtein, seeds: int) -> None:
    for k in range(seeds):
        entry_id = protein.stem if seeds == 1 else f"{protein.stem}#cv1ens{k}"
        a = backend.tokenizer.encode(prefix_and_positions(protein.structure, entry_id=entry_id)[0], add_special_tokens=False)
        b = backend.tokenizer.encode(prefix_and_positions(changed.structure, entry_id=entry_id)[0], add_special_tokens=False)
        differing = sum(x != y for x, y in zip(a, b))
        if len(a) != len(b) or differing != 1:
            raise AssertionError(f"{protein.stem}: mutant prefix differs in {differing} tokens (lengths {len(a)}, {len(b)})")


def run_protein(protein: harness.EvalProtein, backend_for, args) -> tuple[dict, pd.DataFrame, dict]:
    length = protein.length
    scorer = backend_for(protein)
    clean = harness.pcontact(scorer, protein, seeds=args.seeds)

    rows, cols = np.triu_indices(length, k=harness.MIN_SEQ_SEPARATION)
    order = np.argsort(-clean[rows, cols], kind="mergesort")[:length]
    top_i, top_j = rows[order], cols[order]
    gt = gt_contact_matrix(protein.structure.gt_contacts, length, harness.MIN_SEQ_SEPARATION)

    dp_top = np.full((length, len(SUBSTITUTIONS), len(top_i)), np.nan, dtype=np.float32)
    mean_abs_dp = np.full((length, len(SUBSTITUTIONS)), np.nan, dtype=np.float32)
    checked = False
    for k, residue in enumerate(protein.structure.residues):
        for s, aa in enumerate(substitutes(THREE_TO_ONE.get(residue.resname, "X"))):
            if aa is None:
                continue
            changed = mutant(protein, k, aa)
            if not checked:
                check_one_token_change(scorer, protein, changed, args.seeds)
                checked = True
            delta = harness.pcontact(scorer, changed, seeds=args.seeds) - clean
            dp_top[k, s] = delta[top_i, top_j]
            mean_abs_dp[k, s] = np.abs(delta[rows, cols]).mean()

    locality, null_locality = locality_scores(dp_top, top_i, top_j)
    contacts = pd.DataFrame({
        "key": protein.stem,
        "set": protein.eval_set,
        "rank": np.arange(len(top_i)),
        "i": top_i,
        "j": top_j,
        "separation": top_j - top_i,
        "p_clean": clean[top_i, top_j],
        "is_true": gt[top_i, top_j],
        "locality": locality,
        "null_locality": null_locality,
    })
    arrays = {
        "top_pairs": np.stack([top_i, top_j], axis=1),
        "p_clean_top": clean[top_i, top_j].astype(np.float32),
        "dp_top": dp_top.astype(np.float16),
        "mean_abs_dp": mean_abs_dp,
    }
    n_mutants = int(np.isfinite(mean_abs_dp).sum())
    return arrays, contacts, {"n_mutants": n_mutants}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="MODELS.yaml nickname (default: registry default, exp277)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", help="model dtype; float32 measures bf16 rounding in ΔP")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--tail-batch-size", type=int, default=256, help="upper bound on tails per forward")
    parser.add_argument("--kv-budget-gib", type=float, default=16.0, help="cap on the replicated prefix KV cache")
    parser.add_argument("--proteins", choices=("eval", "panel"), default="eval",
                        help="eval: data/a2_proteins.csv; panel: the A4 subset in data/a2_panel_proteins.csv")
    parser.add_argument("--keys", nargs="*", default=None, help="run only these protein keys (smoke test)")
    parser.add_argument("--runner-tag", default="local")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.proteins == "eval":
        table = pd.read_csv(DATA / "a2_proteins.csv")
        loaded = harness.load_eval_proteins() + harness.load_legacy_proteins(DATA / "a2_legacy_gt.jsonl.gz")
    else:
        table = pd.read_csv(DATA / "a2_panel_proteins.csv")
        loaded = harness.load_panel_proteins(DATA / "panel.parquet")
    table["shard"] = assign_shards(table, args.n_shards)
    wanted = table[table["shard"] == args.shard] if args.keys is None else table[table["key"].isin(args.keys)]
    proteins = {p.stem: p for p in loaded if p.stem in set(wanted["key"])}
    todo = [proteins[key] for key in wanted.sort_values("est_l40s_seconds", ascending=False)["key"]]

    entry = resolve_model_entry(args.model)
    load_start = time.perf_counter()
    backend = harness.load_model(
        args.model, device=args.device, dtype=args.dtype, tail_batch_size=args.tail_batch_size
    )
    model_load_seconds = time.perf_counter() - load_start
    sized = BatchSizedBackends(backend, args)

    def backend_for(protein: harness.EvalProtein):
        return sized.for_protein(protein)[0]

    static = {
        "mode": f"a2-{len(SUBSTITUTIONS)}sub-{args.seeds}seed-{args.dtype}",
        "model_load_seconds": round(model_load_seconds, 2),
        "model_nickname": entry.nickname,
        "runner_tag": args.runner_tag,
        **gpu_columns(args.device),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "n_seeds": args.seeds,
        "dtype": args.dtype,
        "tail_batch_size": args.tail_batch_size,
    }
    for directory in ("proteins", "contacts", "timings"):
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)

    print(f"shard {args.shard}/{args.n_shards}: {len(todo)} proteins, "
          f"est {wanted['est_l40s_seconds'].sum() / 3600:.1f} L40S-hours", flush=True)
    for done, protein in enumerate(todo, start=1):
        if (args.output_dir / "timings" / f"{protein.stem}.csv").exists():
            continue
        start = time.perf_counter()
        arrays, contacts, extra = run_protein(protein, backend_for, args)
        elapsed = time.perf_counter() - start
        np.savez_compressed(args.output_dir / "proteins" / f"{protein.stem}.npz", **arrays)
        contacts.to_csv(args.output_dir / "contacts" / f"{protein.stem}.csv", index=False, float_format="%.6g")
        pd.DataFrame([{
            "stem": protein.stem,
            "set": protein.eval_set,
            "n_residues": protein.length,
            "n_pairs": protein.length * (protein.length - 1) // 2,
            "elapsed_seconds": round(elapsed, 3),
            "total_seconds": round(time.perf_counter() - start, 3),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **extra,
            **static,
        }]).to_csv(args.output_dir / "timings" / f"{protein.stem}.csv", index=False)
        true = contacts["is_true"]
        print(f"[{done}/{len(todo)}] {protein.stem} L={protein.length} {elapsed / 60:.1f} min; "
              f"locality {contacts['locality'].mean():.3f} (null {contacts['null_locality'].mean():.3f}), "
              f"true {contacts.loc[true, 'locality'].mean():.3f}, false {contacts.loc[~true, 'locality'].mean():.3f}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
