# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Zephyr/Iris driver for exp266's two GCP-side stages.

    cli.py stage     Stage A2 — AFDB mmCIF -> compact backbone parquet (GCP).
    cli.py generate  Stage B fallback — backbones -> documents on CPU workers.

`stage` is the one that has to run on **GCP**: AFDB's bucket is requester-pays
and only the GCP Iris workers' service account can read it. Its output is what
crosses to CoreWeave, where the real Stage B runs on idle prepaid H100s via
`dispatch_redesign_cw.py`.

`generate` is the all-CPU fallback for when the GPU fleet is busy — same
worker code, `--device cpu`, ~19.5 h on the 735 idle vCPUs of
cw-us-east-02a's `cpu-genoa` pool or on the GCP pool.

The heavy `marin-zephyr` runtime is imported here; `stage_rows`, `generate_rows`
and `redesign` stay import-light so they unit-test without it and run unchanged
inside the CoreWeave task image.
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
import stage_rows
from redesign import DESIGN_TEMPERATURES

# Stage-A manifest columns the staging worker needs, plus provenance carried
# onto each backbone row.
_STAGE_COLUMNS = (
    "entry_id",
    "seq_len",
    "struct_cluster_id",
    "seq_cluster_id",
    "split",
    "round",
    "native_contacts_emitted",
    "native_sha1",
)

# Staged backbone columns the document worker needs.
_GENERATE_COLUMNS = (
    "entry_id",
    "chain_id",
    "resnum_start",
    "sequence",
    "coords_milli",
    "ca_plddt",
    "struct_cluster_id",
    "seq_cluster_id",
    "split",
    "round",
    "native_contacts_emitted",
    "native_sha1",
)


def _resolve_columns(input_path: str, desired: tuple[str, ...],
                     required: tuple[str, ...]) -> list[str]:
    """Peek the first input parquet once, on the controller, and keep the
    desired columns it actually has — so every shard loads the same list and
    the output schema is stable across shards."""
    import pyarrow.parquet as pq

    fs, _ = fsspec.core.url_to_fs(input_path)
    matches = sorted(fs.glob(input_path))
    if not matches:
        raise FileNotFoundError(f"no input files match {input_path}")
    with fsspec.open(fs.unstrip_protocol(matches[0]), "rb") as handle:
        present = set(pq.ParquetFile(handle).schema_arrow.names)
    missing = set(required) - present
    if missing:
        raise ValueError(f"{matches[0]}: input is missing required column(s) {missing}")
    return [c for c in desired if c in present]


def _execute(out_rows, args, out: str) -> None:
    # One output file per input shard: parallel writers, and a failed shard is
    # re-runnable without touching the rest (exp53's rerun_missing).
    if "{shard" not in out:
        out_rows = out_rows.reshard(1)

    suffix = os.path.splitext(out)[1]
    match suffix:
        case ".parquet":
            ds = out_rows.write_parquet(out)
        case ".jsonl" | ".json":
            ds = out_rows.write_jsonl(out)
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
    print(f"[exp266] wrote {out}", file=sys.stderr)


def cmd_stage(args: argparse.Namespace) -> None:
    cif_column = args.cif_text_column or args.cif_uri_column
    columns = _resolve_columns(
        args.input, _STAGE_COLUMNS + (cif_column,), ("entry_id", cif_column)
    )
    rows = Dataset.from_files(args.input).load_parquet(columns=columns)
    if args.num_backbones is not None:
        rows = rows.reshard(1).take_per_shard(args.num_backbones)

    out_rows = rows.map_shard(partial(
        stage_rows.stage_shard,
        cif_uri_column=args.cif_uri_column,
        cif_text_column=args.cif_text_column,
        fetch_concurrency=args.fetch_concurrency,
    ))
    _execute(out_rows, args, args.out)


def cmd_generate(args: argparse.Namespace) -> None:
    columns = _resolve_columns(
        args.input, _GENERATE_COLUMNS,
        ("entry_id", "sequence", "coords_milli", "ca_plddt", "chain_id", "resnum_start"),
    )
    rows = Dataset.from_files(args.input).load_parquet(columns=columns)
    if args.num_backbones is not None:
        rows = rows.reshard(1).take_per_shard(args.num_backbones)

    out_rows = rows.map_shard(partial(
        generate_rows.generate_shard,
        context_length=args.context_length,
        config=GenerationConfig(
            min_seq_separation=args.min_seq_separation,
            min_contact_degree=args.min_contact_degree,
        ),
        structure_name=NAME,
        device=args.device,
        temperatures=tuple(args.temperatures),
    ))
    _execute(out_rows, args, args.out)


def _add_cluster_args(p: argparse.ArgumentParser, *, fetch_concurrency: int | None) -> None:
    if fetch_concurrency is not None:
        p.add_argument("--fetch-concurrency", type=int, default=fetch_concurrency,
                       help="Per-shard fetch threads.")
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--worker-cpu", type=int, default=1)
    p.add_argument("--worker-memory", default="4g")
    p.add_argument("--worker-disk", default="32g")
    p.add_argument("--region", default="us-central1",
                   help="Pin workers to this region so a large pool can't spill "
                        "cross-continent (exp53's straggler tail).")
    p.add_argument("--preemptible", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-backbones", type=int, default=None,
                   help="Smoke cap on input backbones.")
    p.add_argument("--out", required=True,
                   help="Output path; include '{shard:05d}-of-{total:05d}' for "
                        "one parquet per input shard.")
    p.add_argument("--input", required=True, help="Input parquet glob.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    stage = sub.add_parser(
        "stage", help="AFDB mmCIF -> compact backbone parquet (must run on GCP)")
    _add_cluster_args(stage, fetch_concurrency=32)
    stage.add_argument("--cif-uri-column", default="gcs_uri",
                       help="Manifest column holding the per-row structure URI. "
                            "Reading the pointer (not an inline cif column) is "
                            "~2000x less manifest I/O — see exp53.")
    stage.add_argument("--cif-text-column", default=None,
                       help="Inline mmCIF column; local testing only.")
    stage.set_defaults(func=cmd_stage)

    gen = sub.add_parser(
        "generate", help="staged backbones -> redesigned documents (CPU fallback)")
    _add_cluster_args(gen, fetch_concurrency=None)
    gen.add_argument("--temperatures", type=float, nargs="+",
                     default=list(DESIGN_TEMPERATURES),
                     help="ProteinMPNN sampling temperature per design slot.")
    gen.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    gen.add_argument("--context-length", type=int, default=CONTEXT_LENGTH)
    gen.add_argument("--min-seq-separation", type=int, default=6)
    gen.add_argument("--min-contact-degree", type=float, default=0.001)
    gen.set_defaults(func=cmd_generate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
