# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute contacts-set-v1 targets for a small slice of exp139 analyzed shards.

This Zephyr job reads the reusable pyconfind-output rows produced for exp139
and emits one row per protein with fixed-width contacts-set-v1 arrays:
``present[L,16]``, ``signed_coarse[L,16]``, ``fine[L,16]`` and
``degree[L,16]``.  The job is intentionally slice-oriented: use ``--num-shards``
small first, inspect the resulting parquet, then scale only after the target
contract is settled.
"""

import argparse
import os
import re
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

from fray.types import ResourceConfig
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

CONTACT_SLOTS = 16
MAX_ABS_DELTA = 2048
FINE_BINS = 16
COARSE_BINS_PER_SIGN = MAX_ABS_DELTA // FINE_BINS


def encode_delta(delta: int) -> tuple[int, int]:
    """Encode a nonzero signed relative offset as ``(signed_coarse, fine)``."""
    if delta == 0:
        raise ValueError("delta=0 is a self-contact and cannot be encoded")
    abs_delta = abs(delta)
    if abs_delta > MAX_ABS_DELTA:
        raise ValueError(f"abs(delta) must be <= {MAX_ABS_DELTA}, got {delta}")
    abs0 = abs_delta - 1
    coarse = abs0 // FINE_BINS
    fine = abs0 % FINE_BINS
    signed_coarse = coarse + (COARSE_BINS_PER_SIGN if delta > 0 else 0)
    return signed_coarse, fine

DEFAULT_INPUT = (
    "s3://marin-us-east-02a/protein-structure/MarinFold/"
    "exp139_esm_atlas_contacts_v1/analyzed/analyzed-*-of-03338.parquet"
)
DEFAULT_OUTPUT = (
    "s3://marin-us-east-02a/protein-structure/MarinFold/"
    "exp177_contacts_set_v1_targets/slices/2026.09.15.1/"
    "shard-{shard:05d}-of-{total:05d}.parquet"
)
TOTAL_INPUT_SHARDS = 3338
SHARD_RE = re.compile(r"analyzed-(\d+)-of-\d+\.parquet$")

ANALYZED_COLUMNS = (
    "entry_id",
    "seq_len",
    "global_plddt",
    "num_contacts",
    "residue_resname",
    "contact_seq_i",
    "contact_seq_j",
    "contact_degree",
)


@dataclass(frozen=True)
class ResidueContact:
    """One candidate contact incident on a residue."""

    delta: int
    degree: float


def _as_list(value: Any) -> list[Any]:
    """Coerce pyarrow/numpy/list scalars to a plain Python list."""
    if value is None:
        return []
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _top_contacts(contacts: list[ResidueContact], *, max_contacts: int, overflow_policy: str) -> list[ResidueContact]:
    if len(contacts) <= max_contacts:
        return sorted(contacts, key=lambda contact: contact.delta)
    if overflow_policy == "error":
        raise ValueError(f"residue has {len(contacts)} contacts; max supported is {max_contacts}")
    if overflow_policy != "keep_strongest":
        raise ValueError(f"unknown overflow_policy={overflow_policy!r}")
    strongest = sorted(contacts, key=lambda contact: (-contact.degree, abs(contact.delta), contact.delta))[:max_contacts]
    return sorted(strongest, key=lambda contact: contact.delta)


def target_row_from_analyzed(
    row: Mapping[str, Any],
    *,
    source_shard: int,
    min_seq_separation: int,
    min_contact_degree: float,
    overflow_policy: str,
) -> dict[str, Any]:
    """Convert one analyzed contact row to fixed-width contacts-set-v1 targets."""
    entry_id = str(row["entry_id"])
    residue_resname = [str(x) for x in _as_list(row["residue_resname"])]
    seq_len = int(row.get("seq_len") or len(residue_resname))
    if seq_len != len(residue_resname):
        raise ValueError(f"{entry_id}: seq_len={seq_len} but residue_resname has {len(residue_resname)} entries")

    contacts_by_residue: list[list[ResidueContact]] = [[] for _ in range(seq_len)]
    raw_i = _as_list(row["contact_seq_i"])
    raw_j = _as_list(row["contact_seq_j"])
    raw_degree = _as_list(row["contact_degree"])
    if not (len(raw_i) == len(raw_j) == len(raw_degree)):
        raise ValueError(f"{entry_id}: contact arrays have inconsistent lengths")

    contacts_considered = 0
    contacts_used_undirected = 0
    for left, right, degree_value in zip(raw_i, raw_j, raw_degree):
        i = int(left)
        j = int(right)
        degree = float(degree_value)
        if not (0 <= i < seq_len and 0 <= j < seq_len):
            raise ValueError(f"{entry_id}: contact ({i}, {j}) outside seq_len={seq_len}")
        if i == j:
            raise ValueError(f"{entry_id}: self-contact at {i}")
        contacts_considered += 1
        if abs(i - j) < min_seq_separation or degree < min_contact_degree:
            continue
        # Validate the format can represent both directions before mutating.
        encode_delta(j - i)
        encode_delta(i - j)
        contacts_by_residue[i].append(ResidueContact(delta=j - i, degree=degree))
        contacts_by_residue[j].append(ResidueContact(delta=i - j, degree=degree))
        contacts_used_undirected += 1

    present: list[list[bool]] = []
    signed_coarse: list[list[int]] = []
    fine: list[list[int]] = []
    degrees: list[list[float]] = []
    contacts_per_residue_before_cap: list[int] = []
    max_contacts_before_cap = 0
    overflow_residues = 0
    overflow_contacts_dropped = 0

    for residue_contacts in contacts_by_residue:
        before_count = len(residue_contacts)
        contacts_per_residue_before_cap.append(before_count)
        max_contacts_before_cap = max(max_contacts_before_cap, before_count)
        if before_count > CONTACT_SLOTS:
            overflow_residues += 1
            overflow_contacts_dropped += before_count - CONTACT_SLOTS
        kept = _top_contacts(residue_contacts, max_contacts=CONTACT_SLOTS, overflow_policy=overflow_policy)
        residue_present = [False] * CONTACT_SLOTS
        residue_coarse = [0] * CONTACT_SLOTS
        residue_fine = [0] * CONTACT_SLOTS
        residue_degree = [0.0] * CONTACT_SLOTS
        for slot, contact in enumerate(kept):
            coarse, fine_bin = encode_delta(contact.delta)
            residue_present[slot] = True
            residue_coarse[slot] = int(coarse)
            residue_fine[slot] = int(fine_bin)
            residue_degree[slot] = float(contact.degree)
        present.append(residue_present)
        signed_coarse.append(residue_coarse)
        fine.append(residue_fine)
        degrees.append(residue_degree)

    return {
        "entry_id": entry_id,
        "source_shard": int(source_shard),
        "seq_len": int(seq_len),
        "global_plddt": float(row["global_plddt"]) if row.get("global_plddt") is not None else None,
        "residue_resname": residue_resname,
        "min_seq_separation": int(min_seq_separation),
        "min_contact_degree": float(min_contact_degree),
        "raw_contacts": int(len(raw_i)),
        "contacts_considered": int(contacts_considered),
        "contacts_used_undirected": int(contacts_used_undirected),
        "contacts_used_directed": int(2 * contacts_used_undirected),
        "max_contacts_per_residue_before_cap": int(max_contacts_before_cap),
        "overflow_residues": int(overflow_residues),
        "overflow_contacts_dropped": int(overflow_contacts_dropped),
        "present": present,
        "signed_coarse": signed_coarse,
        "fine": fine,
        "degree": degrees,
        "contacts_per_residue_before_cap": contacts_per_residue_before_cap,
    }


def targets_shard(
    items: Iterator[Mapping[str, Any]],
    shard_info,
    *,
    min_seq_separation: int,
    min_contact_degree: float,
    overflow_policy: str,
    max_rows_per_shard: int | None,
) -> Iterator[dict[str, Any]]:
    """Zephyr map_shard body."""
    emitted = 0
    for row in items:
        if max_rows_per_shard is not None and emitted >= max_rows_per_shard:
            break
        yield target_row_from_analyzed(
            row,
            source_shard=int(shard_info.shard_idx),
            min_seq_separation=min_seq_separation,
            min_contact_degree=min_contact_degree,
            overflow_policy=overflow_policy,
        )
        emitted += 1


def _input_files(input_pattern: str, *, num_shards: int | None, start_shard: int) -> Dataset[str]:
    if num_shards is None:
        return Dataset.from_files(input_pattern)
    if "analyzed-*-of-03338.parquet" not in input_pattern:
        raise ValueError("--num-shards requires the default analyzed-* input pattern")
    return Dataset.from_list(
        [
            input_pattern.replace("analyzed-*-of-03338.parquet", f"analyzed-{index:05d}-of-03338.parquet")
            for index in range(start_shard, start_shard + num_shards)
        ]
    )


def run(args: argparse.Namespace) -> None:
    input_files = _input_files(args.input, num_shards=args.num_shards, start_shard=args.start_shard)
    rows = input_files.load_parquet(columns=list(ANALYZED_COLUMNS))
    out_rows = rows.map_shard(
        partial(
            targets_shard,
            min_seq_separation=args.min_seq_separation,
            min_contact_degree=args.min_contact_degree,
            overflow_policy=args.overflow_policy,
            max_rows_per_shard=args.max_rows_per_shard,
        )
    )
    ds = out_rows.write_parquet(args.output)
    ctx = ZephyrContext(
        max_workers=args.max_workers,
        resources=ResourceConfig(
            cpu=args.worker_cpu,
            ram=args.worker_memory,
            disk=args.worker_disk,
            preemptible=args.preemptible,
        ),
    )
    ctx.execute(ds)
    print(f"[exp177] wrote contacts-set-v1 target slice to {args.output}", file=sys.stderr)


def _optional_positive_int(value: str) -> int | None:
    """Parse a positive integer or ``none`` for uncapped shard processing."""
    if value.lower() in {"none", "null", "all"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive, or one of: none, null, all")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--num-shards", type=int, default=1, help="Debug cap on input shards.")
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--max-rows-per-shard", type=_optional_positive_int, default=128)
    parser.add_argument("--min-seq-separation", type=int, default=6)
    parser.add_argument("--min-contact-degree", type=float, default=0.001)
    parser.add_argument("--overflow-policy", choices=("error", "keep_strongest"), default="keep_strongest")
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("EXP177_TARGETS_MAX_WORKERS", "4")))
    parser.add_argument("--worker-cpu", type=float, default=1.0)
    parser.add_argument("--worker-memory", default="4GB")
    parser.add_argument("--worker-disk", default="16GB")
    parser.add_argument("--preemptible", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_shards is not None and args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
