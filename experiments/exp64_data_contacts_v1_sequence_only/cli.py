# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp64 driver — generate the sequence-only contacts-v1 corpus from UniRef50.

Local, embarrassingly parallel: one worker process per UniRef50 ``.fasta.zst``
shard (61 of them), each calling :func:`generate_rows.process_shard` — which
calls straight into marinfold's ``generate_sequence_only_document`` (issue #64;
no document logic re-implemented here). No pyconfind, no structure I/O — the
input is just sequences, so the whole corpus is built on a single box.

Subcommands::

    # Quick format sample (one shard, capped) for inspection.
    python cli.py generate --out ~/exp64_out --shards 0 --limit-per-shard 2000

    # Full run: all 61 shards, downloaded on demand, resumable.
    python cli.py generate --out ~/exp64_out --workers 32

    # Eyeball generated documents + corpus stats.
    python cli.py inspect --out ~/exp64_out --num 5

    # Build + save the unified tokenizer (contacts-v1 + the sequence-only token).
    python cli.py tokenizer --save-local ~/exp64_out/tokenizer

Outputs ``<out>/<split>/uniref50-<shard>-<chunk>.parquet`` plus a ``_done/``
resume marker per finished shard and ``data/generation_counts.csv`` /
``summary.json`` in the experiment dir.
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

from marinfold import build_tokenizer
from marinfold.document_structures.contacts_v1.vocab import (
    SEQUENCE_ONLY_DOC_TYPE_TOKEN,
    all_domain_tokens,
)

import generate_rows
from generate_rows import SPLITS, STRUCTURE_NAME, process_shard

DEFAULT_REPO = "LiteFold/UniRef50"
SHARD_SUBDIR = "sequences/sequence_uniref50_uniref50.fasta.gz"
TOTAL_SHARDS = 61
EXP_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Shard selection + download
# --------------------------------------------------------------------------


def parse_shard_spec(spec: str | None, total: int) -> list[int]:
    """Parse ``--shards`` ("0-4,7,10" / "all" / None) into a sorted index list."""
    if spec is None or spec.strip().lower() in ("", "all"):
        return list(range(total))
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
    return sorted(i for i in out if 0 <= i < total)


def shard_repo_filename(i: int) -> str:
    return f"{SHARD_SUBDIR}/shard-{i:06d}.fasta.zst"


def resolve_shard_path(
    i: int, *, repo: str, shards_dir: str | None, cache_dir: str
) -> Path:
    """Local path to shard ``i`` — found under ``shards_dir`` or downloaded."""
    if shards_dir:
        matches = sorted(Path(shards_dir).rglob(f"shard-{i:06d}.fasta.zst"))
        if not matches:
            raise FileNotFoundError(
                f"shard {i} (shard-{i:06d}.fasta.zst) not found under {shards_dir}"
            )
        return matches[0]
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=repo, repo_type="dataset",
        filename=shard_repo_filename(i), local_dir=cache_dir,
    ))


# --------------------------------------------------------------------------
# Worker (module-level so multiprocessing can pickle it)
# --------------------------------------------------------------------------


def _run_shard(task: dict) -> dict:
    i = task["shard_index"]
    out_dir = Path(task["out_dir"])
    marker = out_dir / "_done" / f"shard-{i:05d}.json"
    if marker.exists() and not task["overwrite"]:
        rec = json.loads(marker.read_text())
        rec["skipped_existing"] = True
        return rec

    fasta_path = resolve_shard_path(
        i, repo=task["repo"], shards_dir=task["shards_dir"],
        cache_dir=task["cache_dir"],
    )
    t0 = time.monotonic()
    counts = process_shard(
        fasta_path, shard_index=i, out_dir=out_dir,
        rows_per_file=task["rows_per_file"],
        val_per_mille=task["val_per_mille"],
        test_per_mille=task["test_per_mille"],
        limit=task["limit"],
    )
    rec = counts.as_dict()
    rec["elapsed_seconds"] = round(time.monotonic() - t0, 1)
    rec["skipped_existing"] = False
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(rec))

    # Free disk during the full run: drop the (large) downloaded shard once
    # its documents are written. Never deletes user-provided --shards-dir files.
    if task["cleanup_input"] and not task["shards_dir"]:
        try:
            fasta_path.unlink()
        except OSError:
            pass
    return rec


# --------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = parse_shard_spec(args.shards, TOTAL_SHARDS)
    workers = args.workers or min(32, (os.cpu_count() or 4))
    tasks = [
        {
            "shard_index": i, "out_dir": str(out_dir),
            "repo": args.repo, "shards_dir": args.shards_dir,
            "cache_dir": str(Path(args.cache_dir).expanduser()),
            "rows_per_file": args.rows_per_file,
            "val_per_mille": args.val_per_mille,
            "test_per_mille": args.test_per_mille,
            "limit": args.limit_per_shard,
            "overwrite": args.overwrite,
            "cleanup_input": args.cleanup_input,
        }
        for i in shards
    ]
    print(
        f"[exp64] generating sequence-only docs from {len(shards)} shard(s) "
        f"with {workers} worker(s) -> {out_dir}",
        file=sys.stderr,
    )

    records: list[dict] = []
    t0 = time.monotonic()
    # imap_unordered streams completions so progress shows as shards finish.
    with mp.Pool(processes=workers) as pool:
        for n, rec in enumerate(pool.imap_unordered(_run_shard, tasks), start=1):
            records.append(rec)
            tag = "cached" if rec.get("skipped_existing") else "done"
            print(
                f"[exp64] ({n}/{len(tasks)}) shard {rec['shard_index']:05d} {tag}: "
                f"{rec.get('written', 0):,} docs, {rec.get('tokens', 0):,} tokens"
                + (f", {rec['elapsed_seconds']}s" if "elapsed_seconds" in rec else ""),
                file=sys.stderr,
            )

    _write_run_summary(records, out_dir=out_dir, wall_seconds=time.monotonic() - t0,
                       shards=shards, workers=workers)


def _write_run_summary(
    records: list[dict], *, out_dir: Path, wall_seconds: float,
    shards: list[int], workers: int,
) -> None:
    """Aggregate per-shard counts to data/generation_counts.csv + summary.json."""
    data_dir = EXP_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    records = sorted(records, key=lambda r: r["shard_index"])

    cols = ["shard_index", "seen", "written", "skipped_length",
            "skipped_unserializable", "files_written", "tokens"]
    cols += [f"{s}_rows" for s in SPLITS] + [f"{s}_tokens" for s in SPLITS]
    with open(data_dir / "generation_counts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for rec in records:
            w.writerow(rec)

    def total(key: str) -> int:
        return sum(int(r.get(key, 0)) for r in records)

    summary = {
        "structure": STRUCTURE_NAME,
        "source": DEFAULT_REPO,
        "shards_processed": shards,
        "workers": workers,
        "wall_seconds": round(wall_seconds, 1),
        "seen": total("seen"),
        "written": total("written"),
        "skipped_length": total("skipped_length"),
        "skipped_unserializable": total("skipped_unserializable"),
        "tokens": total("tokens"),
        "rows_per_split": {s: total(f"{s}_rows") for s in SPLITS},
        "tokens_per_split": {s: total(f"{s}_tokens") for s in SPLITS},
        "output_dir": str(out_dir),
    }
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"\n[exp64] DONE: {summary['written']:,} docs "
        f"({summary['tokens']:,} tokens) from {summary['seen']:,} sequences; "
        f"dropped {summary['skipped_length']:,} out-of-range. "
        f"splits={summary['rows_per_split']}. "
        f"wall={summary['wall_seconds']}s. "
        f"counts -> {data_dir / 'generation_counts.csv'}",
        file=sys.stderr,
    )


# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------


def cmd_tokenizer(args: argparse.Namespace) -> None:
    tok = build_tokenizer(all_domain_tokens())
    print(
        f"[exp64] built unified tokenizer: {len(tok)} tokens "
        f"(last = {SEQUENCE_ONLY_DOC_TYPE_TOKEN} @ id "
        f"{tok.convert_tokens_to_ids(SEQUENCE_ONLY_DOC_TYPE_TOKEN)})",
        file=sys.stderr,
    )
    if args.save_local:
        out = Path(args.save_local).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(str(out))
        print(f"[exp64] saved tokenizer to {out}", file=sys.stderr)
    if args.push:
        tok.push_to_hub(args.push, private=args.private)
        print(f"[exp64] pushed tokenizer to {args.push}", file=sys.stderr)
    if not args.save_local and not args.push:
        print("Pass --save-local DIR or --push REPO to persist the tokenizer.")


# --------------------------------------------------------------------------
# inspect
# --------------------------------------------------------------------------


def cmd_inspect(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    out_dir = Path(args.out).expanduser()
    for split in SPLITS:
        files = sorted((out_dir / split).glob("*.parquet"))
        n_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        print(f"[{split}] {len(files)} file(s), {n_rows:,} rows")

    split_dir = out_dir / args.split
    files = sorted(split_dir.glob("*.parquet"))
    if not files:
        print(f"\nNo parquet files under {split_dir}", file=sys.stderr)
        return
    table = pq.ParquetFile(files[0]).read_row_group(0)
    rows = table.slice(0, args.num).to_pylist()
    print(f"\n=== first {len(rows)} document(s) from {files[0].name} ===")
    for r in rows:
        print(f"\n--- {r['entry_id']}  seq_len={r['seq_len']}  "
              f"num_tokens={r['num_tokens']}  split={r['split']} ---")
        doc = r["document"]
        print(doc if len(doc) <= args.max_chars else doc[: args.max_chars] + " …")


# --------------------------------------------------------------------------
# argparse
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python cli.py",
        description="exp64 — sequence-only contacts-v1 corpus from UniRef50.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate documents (sharded parquet).")
    g.add_argument("--out", required=True, help="Output dir for <split>/*.parquet.")
    g.add_argument("--repo", default=DEFAULT_REPO,
                   help=f"HF dataset to pull shards from (default {DEFAULT_REPO}).")
    g.add_argument("--shards-dir", default=None,
                   help="Local dir holding shard-*.fasta.zst (skips download).")
    g.add_argument("--cache-dir", default="~/exp64_uniref50_shards",
                   help="Where downloaded shards are cached.")
    g.add_argument("--shards", default="all",
                   help="Shard spec: 'all', '0-4', '0,3,7' (default all 61).")
    g.add_argument("--workers", type=int, default=None,
                   help="Worker processes (default min(32, cpu_count)).")
    g.add_argument("--rows-per-file", type=int, default=200_000,
                   help="Max rows per output parquet file (default 200k).")
    g.add_argument("--val-per-mille", type=int, default=5,
                   help="Per-mille of entries to val (default 5 = 0.5%%).")
    g.add_argument("--test-per-mille", type=int, default=5,
                   help="Per-mille of entries to test (default 5 = 0.5%%).")
    g.add_argument("--limit-per-shard", type=int, default=None,
                   help="Cap records read per shard (for quick samples).")
    g.add_argument("--overwrite", action="store_true",
                   help="Reprocess shards even if a _done marker exists.")
    g.add_argument("--cleanup-input", action="store_true",
                   help="Delete each downloaded shard after processing (saves disk).")
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("tokenizer", help="Build/save/push the unified tokenizer.")
    t.add_argument("--save-local", default=None, help="save_pretrained() to this dir.")
    t.add_argument("--push", default=None, help="Push to this HF repo.")
    t.add_argument("--private", action="store_true", help="If --push, make it private.")
    t.set_defaults(func=cmd_tokenizer)

    n = sub.add_parser("inspect", help="Print documents + corpus stats.")
    n.add_argument("--out", required=True, help="Output dir to read.")
    n.add_argument("--split", default="train", choices=SPLITS, help="Split to sample.")
    n.add_argument("--num", type=int, default=5, help="Documents to print.")
    n.add_argument("--max-chars", type=int, default=2000, help="Truncate each doc.")
    n.set_defaults(func=cmd_inspect)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
