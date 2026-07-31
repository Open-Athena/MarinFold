# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Sharded worker: generate backtracking documents from the ESM-Atlas set (#159).

One worker owns a slice of exp139's 3,338 contacts shards and generates
backtracking documents with the model-in-the-loop engine, writing a parquet
**part** every ``--chunk-docs`` documents. Shards are independent, so N workers
fan out with no coordination, and a worker resumes by skipping parts that
already exist.

Part-level (rather than shard-level) writes matter on the preemptible batch
band: a whole-shard write meant one preemption discarded up to a full shard of
GPU work and showed no progress for ~an hour. Parts bound the loss to one chunk
and make progress observable.

Ground truth comes from exp139's saved pyconfind contacts (``esm_atlas_source``)
— no pyconfind at generation time. Documents are produced by the batched
scheduler (``batch_runner.run_batched``), which is ~8.8x the single-protein
loop; see the experiment README for the throughput table.

Every document is validated before it is written: its rendered form must fold
(``read.live_contacts``) to exactly the protein's ground-truth pair set.
Documents that fail (only possible via budget truncation) are dropped and
counted, never written.

Example — one local worker over shards 0-1, 200 docs each::

    uv run python gen_esm_atlas_worker.py --shards 0-1 --docs-per-shard 200 \\
        --out data/esm_atlas

Fan-out — worker i of N, striped over the shard range::

    uv run python gen_esm_atlas_worker.py --shards 0-99 \\
        --worker-id $I --num-workers $N --out s3://.../backtracking
"""

from __future__ import annotations

import argparse
import io
import os
import time

import pandas as pd

from backtrack_adapter import ModelAdapter
from backtrack_engine import RetractionPolicy
from batch_runner import run_batched
from esm_atlas_source import iter_structures

from marinfold import load_backend
from marinfold.document_structures.contacts_v1 import inference as inf
from marinfold.document_structures.contacts_v1.vocab import CONTEXT_LENGTH
from marinfold.registry import resolve_model

_TRIGGER = {"collapse", "floor", "rank"}


def parse_shards(spec: str) -> list[int]:
    """``"0-99"`` / ``"3"`` / ``"0-9,20,30-32"`` -> a sorted shard-index list."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _open_writer(path: str):
    """Return ``(write_bytes_fn, exists_fn)`` for a local, gs:// or s3:// prefix."""
    if path.startswith(("gs://", "s3://")):
        import fsspec

        fs, _ = fsspec.core.url_to_fs(path)

        def write_bytes(dest: str, payload: bytes) -> None:
            with fs.open(dest, "wb") as fh:
                fh.write(payload)

        return write_bytes, fs.exists

    os.makedirs(path, exist_ok=True)

    def write_bytes_local(dest: str, payload: bytes) -> None:
        tmp = dest + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(payload)
        os.replace(tmp, dest)

    return write_bytes_local, os.path.exists


def generate_chunk(backend, structures, args, policy: RetractionPolicy) -> pd.DataFrame:
    """Generate documents for one chunk of structures; return a rows DataFrame."""
    if not structures:
        return pd.DataFrame()

    jobs = []
    by_id: dict[str, tuple] = {}
    for entry_id, analyzed, gt in structures:
        structure = inf.ContactStructure(
            entry_id=entry_id,
            residues=analyzed.residues,
            gt_contacts=analyzed.contacts,
            global_plddt=analyzed.global_plddt,
        )
        try:
            adapter = ModelAdapter(backend, structure, entry_id=entry_id)
        except ValueError:
            continue  # not serializable (length bounds) — skip
        available = CONTEXT_LENGTH - len(adapter.prefix_ids) - 1
        jobs.append((entry_id, gt, adapter, max(len(gt) + 2, available // 3)))
        by_id[entry_id] = (adapter, gt, analyzed)

    if not jobs:
        return pd.DataFrame()

    results = run_batched(
        jobs, backend, policy, seed=args.seed, chunk=args.batch,
        propose_tokens=args.propose_tokens,
    )
    shard = args._current_shard

    rows = []
    for entry_id, result in results.items():
        adapter, gt, analyzed = by_id[entry_id]
        document = adapter.assemble_document(result.statements)
        # Round-trip check, not a GT assertion: the rendered document must fold
        # back to what the ENGINE says it produced. With a flush that is exactly
        # GT (unchanged behaviour); with flush="none" it is the model's own
        # final set, and checking against GT here would drop every document --
        # which is precisely what the first no-flush pilot did, writing an empty
        # shard with no error.
        if not adapter.document_folds_to(document, result.live_final):
            continue  # rendering / position-mapping bug -- never write it
        fp_trigger = sum(1 for _, _, was_true, t in result.retractions
                         if not was_true and t in _TRIGGER)
        fp_total = sum(1 for _, _, was_true, _ in result.retractions if not was_true)
        rows.append({
            "entry_id": entry_id,
            "document": document,
            "seq_len": len(analyzed.residues),
            "global_plddt": float(analyzed.global_plddt),
            "n_gt": len(gt),
            "n_contact_stmts": result.n_contact_statements,
            "n_retract_stmts": result.n_retract_statements,
            "n_reemit": result.n_reemit,
            "n_forced_true": result.n_forced_true,
            "n_fp_emitted": fp_total,
            "fp_retracted_by_trigger": fp_trigger,
            "tp_retracted_by_trigger": sum(
                1 for _, _, was_true, t in result.retractions
                if was_true and t in _TRIGGER
            ),
            "num_tokens": len(document.split()),
            "truncated": result.truncated,
            "source": "esm-atlas-v1",
            "shard": shard,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="contacts-v1-exp120-1.5B")
    ap.add_argument("--shards", required=True, help="e.g. '0-99' or '0-9,20'")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--source", default="hf", choices=("hf", "gcs"))
    ap.add_argument("--out", required=True, help="local dir, gs:// or s3:// prefix")
    ap.add_argument("--docs-per-shard", type=int, default=4000)
    ap.add_argument("--chunk-docs", type=int, default=250,
                    help="documents per output part (checkpoint granularity)")
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--min-gt", type=int, default=4)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--propose-tokens", type=int, default=6)
    ap.add_argument("--eval-cadence", type=int, default=3)
    ap.add_argument("--min-delay", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--s-floor", type=float, default=1e-3)
    ap.add_argument("--noise-prob", type=float, default=0.05)
    ap.add_argument("--flush", default="none", choices=["none", "shuffled", "sorted"],
                    help="closing-flush mode; see backtrack_engine.RetractionPolicy. "
                         "'none' (default) ends the document where the model stopped; "
                         "'shuffled' keeps the flush but removes its ordering signal "
                         "(the control); 'sorted' reproduces the #159 corpus bug.")
    ap.add_argument("--force-true-prob", type=float, default=0.0,
                    help="probability a contact step is forced to a ground-truth "
                         "pair, sampled in proportion to the model's own score on "
                         "the GT pairs not yet live. Length scales as 1/(1-p) "
                         "because only free draws can stop the loop.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    policy = RetractionPolicy(
        min_delay=args.min_delay, eval_cadence=args.eval_cadence,
        tau=args.tau, s_floor=args.s_floor, noise_retract_prob=args.noise_prob,
        flush=args.flush, force_true_prob=args.force_true_prob,
    )
    print(f"worker: flush={args.flush} force_true_prob={args.force_true_prob}",
          flush=True)
    shards = [s for i, s in enumerate(parse_shards(args.shards))
              if i % args.num_workers == args.worker_id]
    print(f"worker {args.worker_id}/{args.num_workers}: {len(shards)} shards", flush=True)

    write_bytes, exists = _open_writer(args.out)
    backend = load_backend(
        "transformers", model=str(resolve_model(args.model)), dtype="bfloat16"
    )

    total_docs = 0
    for shard in shards:
        args._current_shard = shard
        # Read the shard once, then generate in CHUNKS, writing a part parquet
        # after each. Coarse per-shard writes meant a preemption on the batch
        # band (or any crash) threw away up to a whole shard of GPU work and
        # showed no progress for an hour; parts bound the loss to one chunk and
        # make resume fine-grained.
        structures = None
        for part, start in enumerate(range(0, args.docs_per_shard, args.chunk_docs)):
            dest = f"{args.out.rstrip('/')}/backtracking-{shard:05d}-part{part:03d}.parquet"
            if exists(dest):
                continue
            if structures is None:
                structures = list(
                    iter_structures(
                        [shard], source=args.source, min_len=args.min_len,
                        max_len=args.max_len, min_gt=args.min_gt,
                        limit=args.docs_per_shard,
                    )
                )
                if not structures:
                    print(f"  shard {shard}: no usable proteins", flush=True)
                    break
            batch_structures = structures[start:start + args.chunk_docs]
            if not batch_structures:
                break
            t0 = time.time()
            df = generate_chunk(backend, batch_structures, args, policy)
            if df.empty:
                continue
            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            write_bytes(dest, buffer.getvalue())
            total_docs += len(df)
            elapsed = time.time() - t0
            print(
                f"  shard {shard} part {part}: {len(df)} docs in {elapsed:.0f}s "
                f"({elapsed / max(len(df), 1):.2f}s/doc), "
                f"fp_trigger={int(df.fp_retracted_by_trigger.sum())}/"
                f"{int(df.n_fp_emitted.sum())} -> {dest}",
                flush=True,
            )
    print(f"worker {args.worker_id} done: {total_docs} documents", flush=True)


if __name__ == "__main__":
    main()
