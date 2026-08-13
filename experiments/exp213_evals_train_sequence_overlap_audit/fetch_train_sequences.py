# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 1 — materialise every sequence `contacts-v1-exp199-1.5B` was trained on.

exp199's CoreWeave arm saw a 50/50 mixture of two contacts-v1 corpora for ~1.1
epochs of the smaller-weighted one, so both are "the train set" in full. Both
live on the public ``open-athena/MarinFold`` bucket and both are the same
document format, so one streaming path handles them:

* ``afdb`` — ``data/document_structures/contacts_v1/train/``: 2,067 shards
  (13 GB), ~4.13 M documents from ``timodonnell/afdb-24M``'s train split
  (issue #53). Contact labels come from **AlphaFold2** structures.
* ``esm_atlas`` — ``data/document_structures/contacts_v1_esm_atlas/train/``:
  3,338 shards (133 GB), ~66.76 M ESM-Atlas monomers (issue #139), the
  40 %-identity linclust representatives. Contact labels come from **ESMFold2**
  structures.

Neither corpus carries a ``sequence`` column — the sequence is *inside* the
document — so each is recovered with ``contacts_v1.read.sequence_from_document``,
the generator's exact inverse (:mod:`validate_sequences` checks the result
against AlphaFold DB). The corpus is also the only source that is *by
construction* what the model saw: ESM-Atlas's ``selected_manifest.parquet``
carries no sequences at all, and its ``structures/parts/`` are 2.08 TB.

Shards are streamed — download, parse to a FASTA part, delete the parquet — so
peak disk is a few GB even though 146 GB passes through. Output is one FASTA
per arm under ``--work`` plus a JSON of counts, headed ``{arm}|{local_id}`` so
:mod:`search_overlap` can attribute every hit back to an arm. ~17 Gaa total;
``--work`` needs ~80 GB free (the mmseqs DB in step 2 lands beside it).

**Why the downloads shell out to the ``hf`` CLI.** Bucket repos need
``huggingface_hub>=1.5``, and ``marinfold`` (for the document reader) pins
``transformers<5``, which pins ``huggingface_hub<1``. The two cannot share a
venv, so bucket I/O goes through the ``hf`` binary — the ``hf buckets cp`` path
``AGENTS.md`` prescribes — and only the parsing runs in this environment.

Resumable (an existing FASTA part short-circuits its shard) and per-arm::

    uv run python fetch_train_sequences.py --arm afdb      --work /data/exp213_overlap
    uv run python fetch_train_sequences.py --arm esm_atlas --work /data/exp213_overlap
    uv run python fetch_train_sequences.py --arm both --limit-shards 2 --work /tmp/exp213_smoke
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pyarrow.parquet as pq

from marinfold.document_structures.contacts_v1.read import sequence_from_document
from overlap_lib import ARM_AFDB, ARM_ESM, ARMS, fasta_header

BUCKET = "open-athena/MarinFold"

#: Sequences this short can't carry a meaningful alignment and mmseqs' k-mer
#: prefilter can't index them either; they are noise in a leakage measure.
MIN_SEQ_LEN = 10


@dataclass(frozen=True)
class Corpus:
    """One training corpus: where its shards live and how they're named."""

    arm: str
    prefix: str
    shard_name: str      # .format(shard_index)
    n_shards: int
    label: str

    def remote(self, shard: int) -> str:
        return f"{self.prefix}/{self.shard_name.format(shard)}"


CORPORA = {
    ARM_AFDB: Corpus(
        arm=ARM_AFDB,
        prefix="data/document_structures/contacts_v1/train",
        shard_name="contacts_v1-{:05d}-of-02067.parquet",
        n_shards=2067,
        label="AFDB / AlphaFold2 labels (issue #53)",
    ),
    ARM_ESM: Corpus(
        arm=ARM_ESM,
        prefix="data/document_structures/contacts_v1_esm_atlas/train",
        shard_name="shard-{:05d}-of-03338.parquet",
        n_shards=3338,
        label="ESM Atlas / ESMFold2 labels (issue #139)",
    ),
}


@lru_cache(maxsize=1)
def hf_cli() -> str:
    """Path to an ``hf`` binary that understands buckets (huggingface_hub>=1.5).

    This venv's own ``hf`` comes from the ``huggingface_hub<1`` that
    ``transformers<5`` pins, and has no ``buckets`` subcommand — so the venv's
    ``bin`` is skipped and every other ``hf`` on ``PATH`` is probed. Override
    with ``$HF_CLI`` if the right binary lives somewhere unusual.
    """
    candidates = []
    override = os.environ.get("HF_CLI")
    if override:
        candidates.append(override)
    venv_bin = str(Path(sys.executable).parent)
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if directory and directory != venv_bin:
            candidate = Path(directory) / "hf"
            if candidate.is_file() and os.access(candidate, os.X_OK):
                candidates.append(str(candidate))
    for candidate in candidates:
        probe = subprocess.run([candidate, "buckets", "--help"],
                               capture_output=True, check=False)
        if probe.returncode == 0:
            print(f"[hf] using {candidate}", flush=True)
            return candidate
    raise SystemExit(
        "no `hf` CLI with bucket support found (needs huggingface_hub>=1.5). "
        f"Tried: {candidates or '<none on PATH>'}. Set $HF_CLI to override."
    )


def hf_cp(remote: str, local: Path) -> None:
    """``hf buckets cp`` one public file down, atomically."""
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_name(local.name + ".partial")
    subprocess.run(
        [hf_cli(), "buckets", "cp", "-q", f"hf://buckets/{BUCKET}/{remote}", str(tmp)],
        check=True, stdout=subprocess.DEVNULL,
    )
    tmp.rename(local)


def _wrap(seq: str, width: int = 60) -> str:
    return "\n".join(seq[i : i + width] for i in range(0, len(seq), width))


def _shard_job(job: tuple[Corpus, int, Path, Path]) -> tuple[int, int, int]:
    """Download one corpus shard, write its sequences, delete the parquet.

    Returns ``(shard, n_written, n_skipped)``. Resumable: an existing FASTA
    part short-circuits the whole shard without a download.
    """
    corpus, shard, staging, parts_dir = job
    out = parts_dir / f"{shard:05d}.fasta"
    if out.exists():
        return shard, 0, 0

    parquet = staging / f"{shard:05d}.parquet"
    if not parquet.exists():
        hf_cp(corpus.remote(shard), parquet)

    table = pq.read_table(
        parquet, columns=["document", "entry_id", "seq_len", "n_term_index"]
    )
    rows = zip(
        table.column("document").to_pylist(),
        table.column("entry_id").to_pylist(),
        table.column("seq_len").to_pylist(),
        table.column("n_term_index").to_pylist(),
    )
    written = skipped = 0
    tmp = out.with_name(out.name + ".partial")
    with tmp.open("w") as fh:
        for row, (doc, entry_id, seq_len, n_term) in enumerate(rows):
            sequence = sequence_from_document(doc, seq_len, n_term)
            if len(sequence) < MIN_SEQ_LEN:
                skipped += 1
                continue
            header = fasta_header(corpus.arm, f"{shard:05d}_{row}_{entry_id}")
            fh.write(f">{header}\n{_wrap(sequence)}\n")
            written += 1
    tmp.rename(out)
    parquet.unlink(missing_ok=True)
    return shard, written, skipped


def fetch_arm(corpus: Corpus, work: Path, limit_shards: int | None,
              workers: int) -> dict:
    parts_dir = work / f"fasta_parts_{corpus.arm}"
    staging = work / f"_staging_{corpus.arm}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)

    n_shards = min(limit_shards, corpus.n_shards) if limit_shards else corpus.n_shards
    print(f"[{corpus.arm}] {n_shards} of {corpus.n_shards} shards — {corpus.label}",
          flush=True)

    jobs = [(corpus, s, staging, parts_dir) for s in range(n_shards)]
    t0 = time.time()
    written = skipped = done = 0
    # Threads, not processes: each job is download-bound, and pyarrow + the
    # regex scan release the GIL for most of the rest.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _, n_written, n_skipped in pool.map(_shard_job, jobs):
            written += n_written
            skipped += n_skipped
            done += 1
            if done % 100 == 0 or done == len(jobs):
                rate = done / max(time.time() - t0, 1e-9)
                eta = (len(jobs) - done) / rate if rate else 0
                print(f"[{corpus.arm}] {done}/{len(jobs)} shards, "
                      f"{written:,} new sequences, {time.time() - t0:.0f}s "
                      f"(eta {eta / 60:.0f} min)", flush=True)
    shutil.rmtree(staging, ignore_errors=True)

    parts = sorted(parts_dir.glob("*.fasta"))
    if len(parts) != len(jobs):
        raise SystemExit(f"[{corpus.arm}] expected {len(jobs)} fasta parts, "
                         f"found {len(parts)}")
    out = work / f"train_{corpus.arm}.fasta"
    _concat(parts, out)
    return {"arm": corpus.arm, "label": corpus.label, "prefix": corpus.prefix,
            "shards": len(jobs), "written_this_run": written,
            "skipped_short": skipped, "sequences": count_records(out),
            "fasta": str(out), "fasta_bytes": out.stat().st_size}


def _concat(parts: list[Path], out: Path) -> None:
    tmp = out.with_name(out.name + ".partial")
    t0 = time.time()
    with tmp.open("wb") as dst:
        for part in parts:
            with part.open("rb") as src:
                while chunk := src.read(1 << 24):
                    dst.write(chunk)
    tmp.rename(out)
    print(f"[concat] {len(parts)} parts -> {out} "
          f"({out.stat().st_size / 1e9:.1f} GB, {time.time() - t0:.0f}s)", flush=True)


def count_records(fasta: Path) -> int:
    """Number of ``>`` records, counted without loading the file."""
    n = 0
    tail_is_newline = True
    with fasta.open("rb") as fh:
        while chunk := fh.read(1 << 24):
            if tail_is_newline and chunk[:1] == b">":
                n += 1
            n += chunk.count(b"\n>")
            tail_is_newline = chunk.endswith(b"\n")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True,
                    help="scratch dir for FASTAs + the mmseqs DB (needs ~80 GB)")
    ap.add_argument("--arm", choices=[*ARMS, "both"], default="both")
    ap.add_argument("--limit-shards", type=int, default=None,
                    help="smoke test: only this many shards per arm")
    ap.add_argument("--workers", type=int, default=16,
                    help="parallel shard downloads")
    args = ap.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    arms = ARMS if args.arm == "both" else (args.arm,)
    stats = {arm: fetch_arm(CORPORA[arm], args.work, args.limit_shards, args.workers)
             for arm in arms}

    stats_path = args.work / "train_sequences_stats.json"
    merged = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    merged.update(stats)
    merged["total_sequences"] = sum(
        v["sequences"] for k, v in merged.items() if k in ARMS
    )
    stats_path.write_text(json.dumps(merged, indent=2))
    print(json.dumps(merged, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
