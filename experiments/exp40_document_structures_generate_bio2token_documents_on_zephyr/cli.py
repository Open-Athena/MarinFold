# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Driver: generate bio2token-v1 documents at scale on Iris/Zephyr TPUs.

Consumes an input manifest (one row per structure: ``entry_id`` + a structure
URI in ``gcs_uri`` + optional provenance columns) and, per input shard, fetches
the structures and emits bio2token-v1 documents by calling into
:func:`generate_rows.generate_shard`. The heavy neural forward pass runs on a
**TPU worker** (``--device xla``); the launcher (``iris job run``) stays a tiny
CPU coordinator.

The per-row cost is neural inference rather than network I/O, so each worker
fetches its shard concurrently and then tokenizes it in **bucketed batches** on
the accelerator (see ``generate_rows`` / ``tokenizer``).

Examples in the README. Local smoke (in-process, CPU):

    python cli.py generate --input 'file://…/shard_*.parquet' \\
        --device cpu --num-docs 8 --out /tmp/exp40_smoke.parquet

Iris TPU smoke:

    uv run iris --cluster=marin job run --cpu 1 --memory 2GB --extra tpu -- \\
        python cli.py generate \\
            --input 'gs://…/manifest/val/shard_*.parquet' \\
            --out 'gs://marin-us-east5/…/exp40/smoke/corpus.parquet' \\
            --device xla --tpu-type v6e-4 --zone us-east5-b \\
            --num-docs 100 --max-workers 1

The output bucket's region must match the TPU zone (in-region writes); the
launcher's ``--extra tpu`` makes each worker install the ``torch_xla`` wheel.
"""

import argparse
import os
import sys
import typing
from functools import partial

import fsspec

import generate_rows
from tokenizer import DEFAULT_MAX_BATCH, DEFAULT_MAX_BATCH_TOKENS
from vocab import NAME

# Columns projected from the manifest: required (entry_id + cif source) plus
# optional provenance passthrough (carried onto each output row when present).
_DESIRED_COLUMNS: tuple[str, ...] = ("entry_id", *generate_rows.PASSTHROUGH_COLUMNS)


def _resolve_columns(input_path: str, cif_column: str) -> list[str]:
    """Peek the first input parquet; keep the desired columns it actually has.

    Done once on the coordinator so every shard's ``load_parquet(columns=...)``
    uses the same stable list (stable output schema across shards). ``entry_id``
    and the cif source column are required; the rest is provenance passthrough.
    """
    import pyarrow.parquet as pq

    fs, _ = fsspec.core.url_to_fs(input_path)
    matches = sorted(fs.glob(input_path))
    if not matches:
        raise FileNotFoundError(f"No parquet files match {input_path!r}")
    with fsspec.open(fs.unstrip_protocol(matches[0]), "rb") as f:
        present = set(pq.ParquetFile(f).schema_arrow.names)
    if "entry_id" not in present:
        raise ValueError(f"{matches[0]}: input missing required 'entry_id' "
                         f"(found {sorted(present)})")
    if cif_column not in present:
        raise ValueError(f"{matches[0]}: input missing cif column {cif_column!r} "
                         f"(found {sorted(present)})")
    columns = [c for c in _DESIRED_COLUMNS if c in present]
    if cif_column not in columns:
        columns.append(cif_column)
    return columns


def _worker_resources(args: argparse.Namespace):
    """Build the per-worker ResourceConfig: TPU for xla, CPU otherwise."""
    from fray import ResourceConfig

    if args.device == "xla":
        # with_tpu derives cpu/ram from the TPU host VM; pin the zone so a large
        # pool can't spill, and co-locate the output bucket with this zone.
        # Also pin `regions` consistently with the zone: an UNSET regions field
        # (None) makes the worker *inherit the launcher job's region*, which
        # collides with an explicit zone (e.g. launcher region us-central2 +
        # zone us-central1-a -> "unschedulable: no groups in zone").
        zone = args.zone or None
        region = zone.rsplit("-", 1)[0] if zone else (args.region or None)
        return ResourceConfig.with_tpu(
            args.tpu_type, zone=zone,
            regions=[region] if region else None,
            preemptible=args.preemptible)
    return ResourceConfig(
        cpu=args.worker_cpu, ram=args.worker_memory, disk=args.worker_disk,
        regions=[args.region] if args.region else None,
        preemptible=args.preemptible)


def cmd_generate(args: argparse.Namespace) -> None:
    from fray import ResourceConfig
    from zephyr import Dataset, ZephyrContext

    cif_column = args.cif_text_column or args.cif_uri_column
    columns = _resolve_columns(args.input, cif_column)

    rows = Dataset.from_files(args.input).load_parquet(columns=columns)
    if args.num_docs is not None:
        # Global cap; collapse to one shard so the take is deterministic.
        rows = rows.reshard(1).take_per_shard(args.num_docs)

    out_rows = rows.map_shard(partial(
        generate_rows.generate_shard,
        cif_uri_column=args.cif_uri_column,
        cif_text_column=args.cif_text_column,
        device=args.device,
        max_batch=args.max_batch,
        max_batch_tokens=args.max_batch_tokens,
        max_context=args.max_context,
        fetch_concurrency=args.fetch_concurrency,
        on_error=args.on_error,
        structure_name=NAME,
    ))

    # One output file per input shard; omit {shard} to collapse to a single file.
    if "{shard" not in args.out:
        out_rows = out_rows.reshard(1)

    suffix = os.path.splitext(args.out)[1]
    match suffix:
        case ".parquet":
            ds = out_rows.write_parquet(args.out)
        case ".jsonl" | ".json":
            ds = out_rows.write_jsonl(args.out)
        case _:
            typing.assert_never(suffix)

    ctx = ZephyrContext(
        max_workers=args.max_workers,
        resources=_worker_resources(args),
        coordinator_resources=ResourceConfig(cpu=1, ram="2g"),
        # Cold XLA compile (once per bucket) can exceed the default heartbeat;
        # give the first shard room before the worker is declared dead.
        heartbeat_timeout=args.heartbeat_timeout,
    )
    ctx.execute(ds)
    print(f"[{NAME}] wrote {args.out}", file=sys.stderr)


def cmd_vocab(args: argparse.Namespace) -> None:
    from vocab import all_tokens

    toks = all_tokens()
    print(f"{NAME} vocabulary: {len(toks)} tokens")
    print("first 8:", toks[:8])
    print("last 4 :", toks[-4:])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description=f"{NAME} on zephyr — generate documents from an input manifest.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("generate", help="Generate bio2token-v1 documents at scale.")
    p.add_argument("--input", required=True,
                   help="Input-manifest parquet glob (one row per structure), "
                        "e.g. gs://.../manifest/val/shard_*.parquet.")
    p.add_argument("--out", required=True,
                   help="Output path (.parquet/.jsonl). Include {shard:05d}-of-"
                        "{total:05d} to write one file per input shard.")
    p.add_argument("--cif-uri-column", default=generate_rows.DEFAULT_CIF_URI_COLUMN,
                   help="Column with a per-row structure URI fetched by the "
                        f"worker (default {generate_rows.DEFAULT_CIF_URI_COLUMN!r}).")
    p.add_argument("--cif-text-column", default=None,
                   help="Fallback: column holding inline mmCIF text (local "
                        "testing). Overrides --cif-uri-column.")
    p.add_argument("--num-docs", type=int, default=None,
                   help="Global cap on documents (single merged output file).")
    p.add_argument("--max-context", type=int, default=None,
                   help="Skip structures whose document exceeds this token count.")

    # Neural-tokenizer knobs.
    p.add_argument("--device", default="xla", choices=["xla", "cpu", "cuda"],
                   help="Torch device for the encoder (default xla for TPU).")
    p.add_argument("--max-batch", type=int, default=DEFAULT_MAX_BATCH,
                   help="Max structure *count* per batched forward pass "
                        f"(default {DEFAULT_MAX_BATCH}).")
    p.add_argument("--max-batch-tokens", type=int, default=DEFAULT_MAX_BATCH_TOKENS,
                   help="Max padded tokens (B*bucket_len) per forward pass — the "
                        "real HBM governor (big buckets -> small batches). Default "
                        f"{DEFAULT_MAX_BATCH_TOKENS} (~8.6 GB scan peak, fits v6e/v5p).")
    p.add_argument("--fetch-concurrency", type=int, default=32,
                   help="Concurrent per-row URI fetches per shard (overlaps GCS "
                        "GETs with each other; default 32).")
    p.add_argument("--on-error", default="raise", choices=["raise", "skip"],
                   help="Per-row parse-failure policy. Default 'raise' (fail "
                        "loud); flip to 'skip' only after a smoke characterizes "
                        "the expected drop rate.")

    # TPU worker resources. The default (tpu-type, zone) pair must name a real
    # pool in the cluster config (iris/config/marin.yaml -> tpu_pools); v6e-4 in
    # us-east5-b is a small, large-capacity slice. Co-locate --out's region with
    # the zone so per-row writes stay in-region.
    p.add_argument("--tpu-type", default="v6e-4",
                   help="TPU slice type for xla workers (default v6e-4). Must be "
                        "offered by the cluster in --zone; see the cluster's "
                        "tpu_pools config.")
    p.add_argument("--zone", default="us-east5-b",
                   help="Pin TPU workers to this zone so a large pool can't spill; "
                        "co-locate the output bucket's region with it.")
    # CPU worker resources (only used when --device cpu).
    p.add_argument("--worker-cpu", type=float, default=1)
    p.add_argument("--worker-memory", default="4g")
    p.add_argument("--worker-disk", default="32g")
    p.add_argument("--region", default="us-central1",
                   help="Region pin for CPU workers (ignored for TPU; use --zone).")
    p.add_argument("--max-workers", type=int, default=None,
                   help="Cap concurrent Zephyr workers (default: cluster default).")
    p.add_argument("--preemptible", action=argparse.BooleanOptionalAction, default=True,
                   help="Request preemptible/spot workers (cheaper + more capacity).")
    p.add_argument("--heartbeat-timeout", type=float, default=600.0,
                   help="Seconds before a silent worker is declared dead. Raised "
                        "from the 120s default to absorb cold XLA compile.")
    p.set_defaults(func=cmd_generate)

    p_vocab = sub.add_parser("vocab", help="Print the vocabulary.")
    p_vocab.set_defaults(func=cmd_vocab)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
