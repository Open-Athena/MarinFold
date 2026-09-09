# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Snapshot and check existing, region-local tokenized cache ledgers."""

import argparse
import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

from .recipe import CORPORA, REFERENCE_SHA, TOKENIZER

EXPERIMENT = Path(__file__).resolve().parent
ROOT = EXPERIMENT.parents[1]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inspect_cache(path: str, name: str) -> dict:
    """Read a small ledger; validate completeness and reference training counts."""
    body = StoragePath(path.rstrip("/") + "/shard_ledger.json").read_bytes()
    ledger = json.loads(body)
    corpus, documents, tokens = CORPORA[name]
    if not ledger["is_finished"]:
        raise ValueError(f"Incomplete {name} cache")
    if documents is not None and (
        ledger["total_num_rows"],
        ledger["field_counts"]["input_ids"],
    ) != (documents, tokens):
        raise ValueError(
            f"{name} is not the reference decontaminated cache: counts differ"
        )
    return {
        "cache_dir": path,
        "ledger_sha256": sha256(body),
        "ledger": ledger,
        "source_corpus": "hf://buckets/open-athena/MarinFold/data/document_structures/"
        + corpus,
    }


def source_identity() -> dict:
    sources = sorted(EXPERIMENT.glob("*.py")) + [
        ROOT / "experiments/exp232_sweep_cv1_decontam/training_contract.py",
        ROOT / "scripts/history.py",
        ROOT / "scripts/_lib.py",
    ]
    code = b"".join(
        str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes() + b"\0"
        for path in sources
    )
    return {
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "code_sha256": sha256(code),
        "runtime_packages": {
            name: version(name)
            for name in ("marin-levanter", "marin-haliax", "jax", "transformers")
        },
        "uv_lock_sha256": sha256((EXPERIMENT / "uv.lock").read_bytes()),
        "reference_sha": REFERENCE_SHA,
        "tokenizer": TOKENIZER,
    }


def verify_manifest(manifest: dict) -> None:
    """Fail if the source/dependency pin or any selected cache ledger changed."""
    if manifest["source"] != source_identity():
        raise ValueError("Source or lockfile differs from the frozen input manifest")
    for name in CORPORA:
        pinned = manifest["inputs"][name]
        if inspect_cache(pinned["cache_dir"], name) != pinned:
            raise ValueError(f"The {name} input ledger changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in CORPORA:
        parser.add_argument(
            f"--{name}",
            required=True,
            help="Completed tokenized split cache (contains shard_ledger.json)",
        )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = {
        "source": source_identity(),
        "inputs": {name: inspect_cache(getattr(args, name), name) for name in CORPORA},
    }
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
