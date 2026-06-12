# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build ColabFold MMseqs2 MSAs for the candidate sequences (no Modal).

The MSA search runs on the free public ColabFold server
(``https://api.colabfold.com``) regardless of whether you wrap it in Modal;
exp12 used Modal only because it was also running Protenix GPU inference. For
the **MSA-depth axis** all we need is the alignment, so this talks to the
ColabFold MMseqs2 API directly (the standard ``run_mmseqs2`` submit/poll/
download protocol) and writes one ``.a3m`` per candidate to ``data/msa/``.

Search settings mirror what the Protenix-MSA baseline consumes so the Neff is
apples-to-apples: ``use_env=True`` (UniRef30 + ColabFoldDB environmental),
``use_filter=True`` (the ColabFold default ``env`` mode).

Idempotent (skips candidates whose ``.a3m`` already exists) and polite to the
shared service (deduplicates sequences, submits in chunks with backoff polling
and a pause between chunks). Then compute depth with the local tool::

    uv run python fetch_msa_colabfold.py --limit 3     # smoke
    uv run python fetch_msa_colabfold.py               # all 450
    uv run python msa_depth.py dir data/msa --layout flat --out data/candidate_msa_depth.csv
"""

import argparse
import csv
import hashlib
import random
import tarfile
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
HOST = "https://api.colabfold.com"
# Identify ourselves to the shared service (ColabFold asks for this).
UA = "MarinFold-exp65 (github.com/Open-Athena/MarinFold; contact admin@jacobsilterra.com)"
HEADERS = {"User-Agent": UA}


def _submit(seqs: list[str], mode: str, n: int = 101) -> dict:
    query = "".join(f">{n + i}\n{s}\n" for i, s in enumerate(seqs))
    r = requests.post(f"{HOST}/ticket/msa", data={"q": query, "mode": mode},
                      headers=HEADERS, timeout=120)
    try:
        return r.json()
    except requests.exceptions.JSONDecodeError:
        return {"status": "ERROR"}


def _status(ticket_id: str) -> dict:
    r = requests.get(f"{HOST}/ticket/{ticket_id}", headers=HEADERS, timeout=120)
    try:
        return r.json()
    except requests.exceptions.JSONDecodeError:
        return {"status": "ERROR"}


def _download(ticket_id: str, path: Path) -> None:
    r = requests.get(f"{HOST}/result/download/{ticket_id}", headers=HEADERS, timeout=600)
    r.raise_for_status()
    path.write_bytes(r.content)


def _poll_backoff(sleep_floor: int = 5) -> None:
    time.sleep(sleep_floor + random.randint(0, 5))


def run_mmseqs2(seqs: list[str], workdir: Path, *, use_env: bool = True,
                use_filter: bool = True) -> list[str]:
    """Submit ``seqs`` to the ColabFold API and return one a3m string each.

    Faithful port of ColabFold's ``run_mmseqs2`` submit/poll/download +
    per-query a3m demux (queries in the returned files are separated by a NUL
    byte before each query header). Caches the downloaded tarball in
    ``workdir`` so a re-run of the same chunk doesn't resubmit.
    """
    mode = ("env" if use_env else "all") if use_filter else (
        "env-nofilter" if use_env else "nofilter")
    workdir.mkdir(parents=True, exist_ok=True)
    tar_path = workdir / "out.tar.gz"

    # Deduplicate (the API keys on sequence); remember each input's query id.
    seqs_unique = sorted(set(seqs))
    n0 = 101
    ms = [n0 + seqs_unique.index(s) for s in seqs]

    if not tar_path.exists():
        out = _submit(seqs_unique, mode, n0)
        while out.get("status") in ("UNKNOWN", "RATELIMIT"):
            _poll_backoff()
            out = _submit(seqs_unique, mode, n0)
        if out.get("status") == "MAINTENANCE":
            raise RuntimeError("ColabFold API is in maintenance; try again later")
        if out.get("status") == "ERROR" or "id" not in out:
            raise RuntimeError(f"ColabFold submit failed: {out}")
        ticket = out["id"]
        while out.get("status") in ("UNKNOWN", "RUNNING", "PENDING"):
            _poll_backoff()
            out = _status(ticket)
        if out.get("status") != "COMPLETE":
            raise RuntimeError(f"ColabFold job {ticket} ended status={out.get('status')}")
        _download(ticket, tar_path)

    a3m_files = [workdir / "uniref.a3m"]
    if use_env:
        a3m_files.append(workdir / "bfd.mgnify30.metaeuk30.smag30.a3m")
    with tarfile.open(tar_path) as tf:
        try:
            tf.extractall(workdir, filter="data")  # safe-extraction filter (py>=3.11.4)
        except TypeError:
            tf.extractall(workdir)

    # Demux: accumulate each query's lines across both a3m files.
    blocks: dict[int, list[str]] = {}
    for a3m_file in a3m_files:
        update_m, m = True, None
        for line in a3m_file.read_text().splitlines(keepends=True):
            if not line:
                continue
            if "\x00" in line:
                line = line.replace("\x00", "")
                update_m = True
            if line.startswith(">") and update_m:
                m = int(line[1:].rstrip())
                update_m = False
                blocks.setdefault(m, [])
            if m is not None:
                blocks[m].append(line)
    joined = {k: "".join(v) for k, v in blocks.items()}
    # A query with zero hits in both DBs is omitted from the returned a3m;
    # that's a genuine orphan -> its MSA is just itself (Neff == 1). Emit a
    # self-only a3m rather than KeyError, keying the sequence off its query id.
    out: list[str] = []
    for s, m in zip(seqs, ms):
        out.append(joined.get(m, f">{m}\n{s}\n"))
    return out


def load_todo(candidates_csv: Path, msa_dir: Path, limit: int | None) -> list[dict]:
    with candidates_csv.open() as fh:
        rows = list(csv.DictReader(fh))
    if limit is not None:
        rows = rows[:limit]
    return [r for r in rows if not (msa_dir / f"{r['stem']}.a3m").exists()]


def run(candidates_csv: Path, msa_dir: Path, scratch: Path, chunk_size: int,
        pause: int, limit: int | None, use_env: bool) -> None:
    msa_dir.mkdir(parents=True, exist_ok=True)
    todo = load_todo(candidates_csv, msa_dir, limit)
    have = len(list(msa_dir.glob("*.a3m")))
    print(f"{len(todo)} MSAs to fetch ({have} already present); chunk={chunk_size}")
    if not todo:
        print("nothing to do.")
        return

    for ci in range(0, len(todo), chunk_size):
        chunk = todo[ci:ci + chunk_size]
        # Content-address the scratch dir on the chunk's sequences. Keying it on
        # the chunk *index* alone is unsafe: a later run with a different todo
        # list reuses chunk_000's cached out.tar.gz and silently demuxes the
        # previous run's sequences into the new stems. The hash makes the cache
        # correct (identical chunk -> reuse; any change -> fresh fetch).
        key = hashlib.sha1("\n".join(sorted(r["sequence"] for r in chunk)).encode()).hexdigest()[:12]
        workdir = scratch / f"chunk_{ci // chunk_size:03d}_{key}"
        print(f"[chunk {ci // chunk_size}] submitting {len(chunk)} sequences ...", flush=True)
        a3ms = run_mmseqs2([r["sequence"] for r in chunk], workdir, use_env=use_env)
        for r, a3m in zip(chunk, a3ms):
            (msa_dir / f"{r['stem']}.a3m").write_text(a3m)
        done = len(list(msa_dir.glob("*.a3m")))
        print(f"  wrote {len(chunk)} a3m ({done} total on disk)")
        if ci + chunk_size < len(todo):
            time.sleep(pause)  # be polite between chunks
    print(f"Done. a3m files in {msa_dir}/")
    print("Next: uv run python msa_depth.py dir data/msa --layout flat "
          "--out data/candidate_msa_depth.csv")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates-csv", type=Path, default=HERE / "data" / "candidate_sequences.csv")
    p.add_argument("--msa-dir", type=Path, default=HERE / "data" / "msa")
    p.add_argument("--scratch", type=Path, default=HERE / "tmp" / "colabfold")
    p.add_argument("--chunk-size", type=int, default=50, help="Sequences per API job (be polite).")
    p.add_argument("--pause", type=int, default=10, help="Seconds to wait between chunks.")
    p.add_argument("--limit", type=int, default=None, help="First N candidates (smoke test).")
    p.add_argument("--no-env", dest="use_env", action="store_false",
                   help="UniRef only (drop the environmental DB); default uses env.")
    args = p.parse_args()
    run(args.candidates_csv, args.msa_dir, args.scratch, args.chunk_size,
        args.pause, args.limit, args.use_env)


if __name__ == "__main__":
    main()
