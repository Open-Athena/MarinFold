# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute ColabFold MSA depth for exp324 targets using the public MMseqs API.

This is intentionally lightweight: it calls the same ColabFold API mode that
Protenix uses, computes raw depth + Neff from the returned non-pairing A3M, and
writes only per-protein depth rows. The temporary A3Ms are deleted after each
protein unless --keep-a3m-dir is provided.
"""

import argparse
import io
import json
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import fsspec
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp12_data_protenix_foldbench_monomers"))
import msa_depth as md

HOST_URL = "https://api.colabfold.com"
USER_AGENT = "MarinFold-exp324-msa-depth/0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=Path("experiments/exp324_evals_esmfold2_pdb_sample/data/sample_10000_targets.parquet"))
    parser.add_argument("--out-prefix", default="s3://marin-us-east-02a/protein-structure/MarinFold/exp324_esmfold2_pdb_sample/msa_depth_colabfold")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tmp-dir", type=Path, default=Path("/tmp/exp324_msa_depth"))
    parser.add_argument("--keep-a3m-dir", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_fasta_string(text: str) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    name: str | None = None
    for raw in text.replace("\x00", "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            name = line[1:].strip()
            records[name] = []
        elif name is not None:
            records[name].append(line)
    return {key: "".join(value) for key, value in records.items()}


def _json_response(response: requests.Response) -> dict[str, Any]:
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(f"server did not reply with JSON: status={response.status_code} text={response.text[:500]!r}") from exc


def submit(sequence: str, stem: str) -> dict[str, Any]:
    query = f">query_0\n{sequence}\n"
    while True:
        response = requests.post(
            f"{HOST_URL}/ticket/msa",
            data={"q": query, "mode": "env", "email": ""},
            timeout=10,
            headers={"User-Agent": USER_AGENT},
            auth=HTTPBasicAuth(None, None),
        )
        out = _json_response(response)
        if out.get("status") in {"UNKNOWN", "RATELIMIT"}:
            print(f"[{stem}] submit {out.get('status')}; sleeping 60s", flush=True)
            time.sleep(60)
            continue
        return out


def poll(ticket_id: str, stem: str) -> dict[str, Any]:
    while True:
        response = requests.get(
            f"{HOST_URL}/ticket/{ticket_id}",
            timeout=10,
            headers={"User-Agent": USER_AGENT},
            auth=HTTPBasicAuth(None, None),
        )
        out = _json_response(response)
        status = out.get("status")
        if status in {"UNKNOWN", "RUNNING", "PENDING"}:
            print(f"[{stem}] status={status}; sleeping 60s", flush=True)
            time.sleep(60)
            continue
        return out


def download(ticket_id: str) -> bytes:
    response = requests.get(
        f"{HOST_URL}/result/download/{ticket_id}",
        timeout=30,
        headers={"User-Agent": USER_AGENT},
        auth=HTTPBasicAuth(None, None),
    )
    response.raise_for_status()
    return response.content


def combined_non_pairing_a3m(tar_bytes: bytes, sequence: str, work_dir: Path) -> str:
    work_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        tar.extractall(work_dir)
    env_path = work_dir / "bfd.mgnify30.metaeuk30.smag30.a3m"
    uniref_path = work_dir / "uniref.a3m"
    env_records = parse_fasta_string(env_path.read_text().replace("\x00", "")) if env_path.exists() else {}
    uniref_records = parse_fasta_string(uniref_path.read_text().replace("\x00", "")) if uniref_path.exists() else {}
    lines = [">query", sequence]
    for records in (env_records, uniref_records):
        for key, value in records.items():
            if key.startswith("query_"):
                continue
            lines.extend([f">{key}", value])
    return "\n".join(lines) + "\n"


def compute_one(stem: str, sequence: str, tmp_root: Path, keep_a3m_dir: bool) -> dict[str, Any]:
    started = time.monotonic()
    safe = stem.replace("/", "_")
    work_dir = tmp_root / safe
    if work_dir.exists():
        shutil.rmtree(work_dir)
    row: dict[str, Any] = {"stem": stem, "seq_len": len(sequence), "found": False, "status": "error", "error": ""}
    try:
        submitted = submit(sequence, stem)
        if submitted.get("status") in {"ERROR", "MAINTENANCE"}:
            raise RuntimeError(f"submit failed: {submitted}")
        ticket_id = submitted["id"]
        final = poll(ticket_id, stem)
        if final.get("status") != "COMPLETE":
            raise RuntimeError(f"ticket {ticket_id} ended with {final}")
        tar_bytes = download(ticket_id)
        a3m = combined_non_pairing_a3m(tar_bytes, sequence, work_dir)
        depth = md.msa_depth(a3m)
        row.update(
            {
                "found": True,
                "status": "ok",
                "ticket_id": ticket_id,
                "a3m_bytes": len(a3m),
                "n_seqs": depth.n_seqs,
                "query_len": depth.query_len,
                **{f"n_eff_{threshold}": value for threshold, value in depth.n_eff.items()},
            }
        )
        if keep_a3m_dir:
            (work_dir / "non_pairing.combined.a3m").write_text(a3m)
    except Exception as exc:  # noqa: BLE001 - preserve per-protein failures.
        row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        row["elapsed_seconds"] = time.monotonic() - started
        if not keep_a3m_dir and work_dir.exists():
            shutil.rmtree(work_dir)
    return row


def write_csv(df: pd.DataFrame, uri: str) -> None:
    with fsspec.open(uri, "wt") as handle:
        df.to_csv(handle, index=False)


def main() -> int:
    args = parse_args()
    out_uri = f"{args.out_prefix.rstrip('/')}/shards/shard-{args.shard_index:03d}-of-{args.num_shards:03d}.csv"
    summary_uri = f"{args.out_prefix.rstrip('/')}/shards/shard-{args.shard_index:03d}-of-{args.num_shards:03d}.summary.json"
    fs, _, [path] = fsspec.get_fs_token_paths(out_uri)
    if fs.exists(path) and not args.overwrite:
        print(f"output exists, skipping: {out_uri}")
        return 0

    targets = pd.read_parquet(args.targets).sort_values(["L", "stem"], ascending=[False, True], ignore_index=True)
    shard = targets.iloc[args.shard_index :: args.num_shards].copy()
    if args.limit is not None:
        shard = shard.head(args.limit)
    print(f"[msa-depth] shard {args.shard_index}/{args.num_shards}: rows={len(shard)} out={out_uri}", flush=True)

    rows = []
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    for n, record in enumerate(shard.to_dict("records"), start=1):
        stem = record["stem"]
        print(f"[msa-depth] {n}/{len(shard)} {stem} L={record['L']}", flush=True)
        row = compute_one(stem, record["input_seq"], args.tmp_dir, args.keep_a3m_dir)
        rows.append(row)
        print(f"[msa-depth] done {stem} status={row['status']} n_seqs={row.get('n_seqs')} elapsed={row['elapsed_seconds']:.1f}s", flush=True)

    frame = pd.DataFrame(rows)
    write_csv(frame, out_uri)
    summary = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "rows": int(len(frame)),
        "ok": int((frame.status == "ok").sum()) if len(frame) else 0,
        "errors": int((frame.status != "ok").sum()) if len(frame) else 0,
        "out": out_uri,
    }
    with fsspec.open(summary_uri, "wt") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
