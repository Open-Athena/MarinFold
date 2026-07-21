# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage 1 driver — generate contacts-v1 documents + reusable pyconfind contacts
from the ESMFold2-Atlas distillation set, at scale on Iris/Zephyr.

Consumes the in-region GCS mirror of ``open-athena/esm-atlas-esmfold2-distill``
(see ``mirror_source.py`` / Stage 0): ``structures/parts/part_*.parquet``, each
row carrying an **inline** ``cif_content`` mmCIF, ``entry_id``, and provenance
(``seq_cluster_id``, ``ptm``, ``plddt_std``, ``cluster_size``, ``source``,
``split``). For each structure we run pyconfind **once** and emit one combined
row (contacts-v1 document + the raw contacts record) — see ``generate_rows.py``.

Region discipline (this experiment's whole point): the mirror, the workers, and
the output all live in **us-central1**, so there are no GCP inter-region hops.
The marin cluster has no preemptible CPU scale group, so we run on-demand
(``--no-preemptible``, the exp105 lesson); ``--region`` is repeatable if you
must widen the on-demand pool, but widening it means workers cross-region-READ
the ~2 TB mirror, which defeats the purpose — keep it single-region.

The heavy ``marin-zephyr`` runtime is imported here; ``generate_rows`` stays
import-light so it unit-tests without it.
"""

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

# Columns we read from each ESM-Atlas part: the inline cif + id are required;
# the rest are provenance passthrough (kept if present).
_DESIRED_COLUMNS: tuple[str, ...] = (
    "entry_id",
    generate_rows.DEFAULT_CIF_TEXT_COLUMN,   # "cif_content"
    "seq_cluster_id",
    "cluster_size",
    "ptm",
    "plddt_std",
    "source",
    "split",
)


def _assembly_arg(text: str) -> int | str | None:
    """Parse ``--assembly`` into pyconfind's ``int | str | None`` surface."""
    stripped = text.strip()
    if stripped.lower() == "none":
        return None
    try:
        return int(stripped)
    except ValueError:
        if not stripped:
            raise argparse.ArgumentTypeError(
                "assembly must be an integer, a named assembly, or 'none'"
            ) from None
        return stripped


def _config_from_args(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        native_only=args.native_only,
        contact_distance=args.contact_distance,
        dcut=args.dcut,
        clash_distance=args.clash_distance,
        assembly=args.assembly,
        min_seq_separation=args.min_seq_separation,
        min_contact_degree=args.min_contact_degree,
    )


def _resolve_columns(input_path: str, cif_column: str) -> list[str]:
    """Peek the first input parquet and keep the desired columns it actually has.

    Done once at submission time so every shard's ``load_parquet(columns=...)``
    uses the same stable list. ``entry_id`` and the cif column are required;
    everything else is opportunistic provenance passthrough.
    """
    import pyarrow.parquet as pq

    fs, _ = fsspec.core.url_to_fs(input_path)
    matches = sorted(fs.glob(input_path))
    if not matches:
        raise FileNotFoundError(f"No parquet files match {input_path!r}")
    with fsspec.open(fs.unstrip_protocol(matches[0]), "rb") as f:
        present = set(pq.ParquetFile(f).schema_arrow.names)
    if "entry_id" not in present:
        raise ValueError(f"{matches[0]}: input is missing required 'entry_id' "
                         f"(found {sorted(present)})")
    if cif_column not in present:
        raise ValueError(f"{matches[0]}: input is missing cif column {cif_column!r} "
                         f"(found {sorted(present)})")
    columns = [c for c in _DESIRED_COLUMNS if c in present]
    if cif_column not in columns:
        columns.append(cif_column)
    return columns


def cmd_generate(args: argparse.Namespace) -> None:
    cfg = _config_from_args(args)
    cif_column = args.cif_text_column
    columns = _resolve_columns(args.input, cif_column)

    rows = Dataset.from_files(args.input).load_parquet(columns=columns)
    if args.num_docs is not None:
        # Global cap; collapse to one shard so the take is deterministic.
        # Use for the iris smoke + pilot runs.
        rows = rows.reshard(1).take_per_shard(args.num_docs)

    out_rows = rows.map_shard(partial(
        generate_rows.generate_shard,
        cif_text_column=cif_column,
        context_length=args.context_length,
        config=cfg,
        structure_name=NAME,
    ))

    # One output file per input part; omit {shard} to collapse to a single file.
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

    regions = args.region if args.region else ["us-central1"]
    ctx = ZephyrContext(
        max_workers=args.max_workers,
        resources=ResourceConfig(
            cpu=args.worker_cpu, ram=args.worker_memory, disk=args.worker_disk,
            regions=regions,
            preemptible=args.preemptible,
        ),
    )
    ctx.execute(ds)
    print(f"[{NAME}] wrote {args.out}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description=f"{NAME} on zephyr — generate documents + reusable contacts "
                    "from the ESM-Atlas distillation set.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("generate", help="Generate documents + contacts at scale.")
    p.add_argument("--input", type=str, required=True,
                   help="ESM-Atlas parts glob (the in-region GCS mirror), e.g. "
                        "gs://marin-us-central1/protein-structure/MarinFold/"
                        "exp139_esm_atlas_contacts_v1/source/structures/parts/"
                        "part_*.parquet.")
    p.add_argument("--cif-text-column", type=str,
                   default=generate_rows.DEFAULT_CIF_TEXT_COLUMN,
                   help="Column holding inline mmCIF text "
                        f"(default {generate_rows.DEFAULT_CIF_TEXT_COLUMN!r}).")
    p.add_argument("--num-docs", type=int, default=None,
                   help="Global cap on rows produced (single merged file). "
                        "Use for the iris smoke + pilot runs.")
    p.add_argument("--context-length", type=int, default=CONTEXT_LENGTH,
                   help=f"Token budget per document (default {CONTEXT_LENGTH}).")
    p.add_argument("--out", type=str, required=True,
                   help="Output path (.parquet/.jsonl). Include a {shard} "
                        "placeholder (e.g. analyzed-{shard:05d}-of-{total:05d}"
                        ".parquet) to write one file per input part.")

    # Zephyr worker resources (1 CPU/worker, scale via max-workers). ESM-Atlas
    # parts are ~600 MB with inline cif, so give the worker more RAM headroom
    # than the AFDB URI-fetch pipelines (which streamed tiny manifest rows).
    p.add_argument("--worker-cpu", type=float, default=1)
    p.add_argument("--worker-memory", type=str, default="8g")
    p.add_argument("--worker-disk", type=str, default="32g")
    p.add_argument("--max-workers", type=int, default=None,
                   help="Cap concurrent Zephyr workers (default: cluster default "
                        "or ZEPHYR_MAX_WORKERS).")
    p.add_argument("--region", action="append", default=None,
                   help="GCP region(s) to place workers in (repeatable). Defaults "
                        "to ['us-central1'] — region-local to the mirror + output. "
                        "WARNING: adding regions makes those workers cross-region "
                        "READ the ~2 TB mirror; keep single-region unless you have "
                        "a reason and have weighed the egress.")
    p.add_argument("--preemptible", action=argparse.BooleanOptionalAction, default=False,
                   help="Request preemptible/spot workers. Default False: the marin "
                        "cluster has NO preemptible CPU scale group, so a "
                        "preemptible request creates zero autoscaler demand and the "
                        "job strands. Use on-demand.")

    # Generation knobs (contacts-v1 SPEC defaults).
    cfg = GenerationConfig()
    p.add_argument("--min-seq-separation", type=int, default=cfg.min_seq_separation)
    p.add_argument("--min-contact-degree", type=float, default=cfg.min_contact_degree)
    p.add_argument("--contact-distance", type=float, default=cfg.contact_distance)
    p.add_argument("--dcut", type=float, default=cfg.dcut)
    p.add_argument("--clash-distance", type=float, default=cfg.clash_distance)
    p.add_argument("--native-only", action=argparse.BooleanOptionalAction,
                   default=cfg.native_only)
    p.add_argument("--assembly", type=_assembly_arg, default=cfg.assembly)
    p.set_defaults(func=cmd_generate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
