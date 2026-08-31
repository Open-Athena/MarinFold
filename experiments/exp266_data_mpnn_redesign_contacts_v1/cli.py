# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage B driver — redesign + document generation at scale on Iris/Zephyr.

Consumes the Stage-A manifest (``select_backbones.py``: one row per kept
decontaminated backbone, with ``gcs_uri`` + provenance) and, for each row,
fetches the structure, redesigns it 8 ways with ProteinMPNN, and emits one
contacts-v1 document per design. The per-row work lives in
``generate_rows.py``; exp266 re-implements no document logic (issue #266).

One job, not two: see ``generate_rows``'s module docstring for why the
redesign runs on the same CPU workers as the document generation rather
than on a separate GPU cluster.

The heavy ``marin-zephyr`` runtime is imported here; ``generate_rows`` and
``redesign`` stay import-light so they unit-test without it.
"""

from __future__ import annotations

import argparse
import os
import sys
import typing
from functools import partial

import fsspec
from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext

from marinfold.document_structures.contacts_v1 import GenerationConfig
from marinfold.document_structures.contacts_v1.vocab import CONTEXT_LENGTH, NAME

import generate_rows
from redesign import DESIGN_TEMPERATURES

# Manifest columns the worker needs plus provenance carried onto each output
# row. ``entry_id`` and the cif source are required; the rest are optional.
_DESIRED_COLUMNS = (
    "entry_id",
    "seq_len",
    "struct_cluster_id",
    "seq_cluster_id",
    "split",
    "round",
    "native_contacts_emitted",
    "native_sha1",
)


def _resolve_columns(input_path: str, cif_column: str) -> list[str]:
    """Peek the first input parquet once, on the controller, and keep the
    columns it actually has — so every shard loads the same list and the
    output schema is stable across shards."""
    import pyarrow.parquet as pq

    fs, _ = fsspec.core.url_to_fs(input_path)
    matches = sorted(fs.glob(input_path))
    if not matches:
        raise FileNotFoundError(f"no input files match {input_path}")
    with fsspec.open(fs.unstrip_protocol(matches[0]), "rb") as handle:
        present = set(pq.ParquetFile(handle).schema_arrow.names)
    if "entry_id" not in present:
        raise ValueError(f"{matches[0]}: input is missing required 'entry_id'")
    if cif_column not in present:
        raise ValueError(f"{matches[0]}: input is missing cif column {cif_column!r}")
    columns = [c for c in _DESIRED_COLUMNS if c in present]
    if cif_column not in columns:
        columns.append(cif_column)
    return columns


def _config_from_args(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        min_seq_separation=args.min_seq_separation,
        min_contact_degree=args.min_contact_degree,
    )


def cmd_generate(args: argparse.Namespace) -> None:
    cfg = _config_from_args(args)
    cif_column = args.cif_text_column or args.cif_uri_column
    columns = _resolve_columns(args.input, cif_column)

    rows = Dataset.from_files(args.input).load_parquet(columns=columns)
    if args.num_backbones is not None:
        # Global cap for smoke runs; collapse to one shard so the take is
        # deterministic.
        rows = rows.reshard(1).take_per_shard(args.num_backbones)

    out_rows = rows.map_shard(partial(
        generate_rows.generate_shard,
        cif_uri_column=args.cif_uri_column,
        cif_text_column=args.cif_text_column,
        context_length=args.context_length,
        config=cfg,
        fetch_concurrency=args.fetch_concurrency,
        structure_name=NAME,
        device=args.device,
        temperatures=tuple(args.temperatures),
    ))

    # One output file per input manifest shard: parallel writers, and a failed
    # shard is re-runnable without touching the rest (exp53's rerun_missing).
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
        resources=ResourceConfig(
            cpu=args.worker_cpu, ram=args.worker_memory, disk=args.worker_disk,
            regions=[args.region] if args.region else None,
            preemptible=args.preemptible,
        ),
    )
    ctx.execute(ds)
    print(f"[exp266] wrote {args.out}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate", help="redesign + generate documents")

    p.add_argument("--input", required=True,
                   help="Stage-A manifest glob, e.g. 'gs://.../manifest-*.parquet'.")
    p.add_argument("--out", required=True,
                   help="Output path; include '{shard:05d}-of-{total:05d}' for "
                        "one parquet per input shard.")
    p.add_argument("--num-backbones", type=int, default=None,
                   help="Smoke cap on input backbones (not output documents).")

    p.add_argument("--cif-uri-column", default="gcs_uri",
                   help="Manifest column holding the per-row structure URI. "
                        "Reading the pointer (not an inline cif column) is "
                        "~2000x less manifest I/O — see exp53.")
    p.add_argument("--cif-text-column", default=None,
                   help="Inline mmCIF column; local testing only.")

    p.add_argument("--temperatures", type=float, nargs="+",
                   default=list(DESIGN_TEMPERATURES),
                   help="ProteinMPNN sampling temperature per design slot. "
                        "The default ladder spans near-native to diverse and "
                        "is recorded per row so training can subset it.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Where ProteinMPNN runs. 'cpu' keeps this a single "
                        "Iris job; 'cuda' is ~18x faster per sequence but "
                        "needs the backbones staged to the GPU cluster.")

    p.add_argument("--context-length", type=int, default=CONTEXT_LENGTH)
    p.add_argument("--min-seq-separation", type=int, default=6)
    p.add_argument("--min-contact-degree", type=float, default=0.001)

    p.add_argument("--fetch-concurrency", type=int, default=4,
                   help="Per-shard fetch threads. Low on purpose: the CPU step "
                        "here is seconds per row, so the GCS GET is already "
                        "hidden and more threads only contend for the core.")
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--worker-cpu", type=int, default=1)
    p.add_argument("--worker-memory", default="4g")
    p.add_argument("--worker-disk", default="32g")
    p.add_argument("--region", default="us-central1",
                   help="Pin workers to this region so a large pool can't "
                        "spill cross-continent (exp53's straggler tail).")
    p.add_argument("--preemptible", action=argparse.BooleanOptionalAction,
                   default=True)
    p.set_defaults(func=cmd_generate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
