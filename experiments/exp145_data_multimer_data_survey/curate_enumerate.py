# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Enumerate every AFDB dimer passing the release's own quality gate.

``profile_confidence.py`` estimates gate yields from byte-range samples. This
script settles them: it streams both metadata CSVs end to end (6.00 GB homodimer,
2.45 GB heterodimer) and writes out every passing row, so the downstream fetch
works from an exact ID list rather than a projection.

The gate is the release's own, ``ipSAE >= 0.6 and pDockQ2 >= 0.23``, taking the
max over the two chain orderings. The heterodimer table ships the verdict as
``passes_quality_threshold`` (and the inputs as ``max_ipSAE`` /
``max_pDockQ2_AB``); the homodimer table ships neither, which is why the gate is
recomputed from the raw columns for both. Where the declared column exists it is
compared row by row and any disagreement is fatal, not a warning: the whole basis
for applying this gate to homodimers is that the recomputation is exact.

Nothing here downloads a structure. Output is one gzipped CSV per set holding the
model ID, both UniProt accessions, gene names, taxon IDs, the scores the gate
used, and ``local_tar_name`` (which locates the model inside the FTP tar shards
if we ever want the bulk path instead of per-file GETs).

Usage::

    python3 curate_enumerate.py --out data/curation
    python3 curate_enumerate.py --out /tmp/pilot --max-bytes 200_000_000
"""

import argparse
import csv
import gzip
import io
import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterator

FTP_BASE = "https://ftp.ebi.ac.uk/pub/databases/alphafold/collaborations/nvda/"

IPSAE_MIN = 0.6
PDOCKQ2_MIN = 0.23

# Read size for the streaming pass. Large enough that per-read overhead vanishes
# against an 8.45 GB transfer, small enough to keep memory flat.
CHUNK_BYTES = 8 << 20

# Columns kept for every passing row. Deliberately narrow: the full tables carry
# ~40 ipSAE intermediates (d0chn, n0res, nres1 ...) that only matter if we want to
# recompute the scores ourselves, and carrying them would multiply the output.
OUT_COLUMNS = [
    "model_entity_id", "kind",
    "uniprot_1", "uniprot_2", "gene_1", "gene_2", "tax_1", "tax_2",
    "ipTM", "ipSAE_max", "pDockQ2_max", "pDockQ", "LIS_max",
    "n_clash_backbone", "n_interactions", "local_tar_name",
]

SPECS: dict[str, dict[str, Any]] = {
    "homodimer": {
        "csv": "homodimer_metadata.csv",
        "uniprot": ("uniprotAccession", "uniprotAccession"),
        "gene": ("gene", "gene"),
        "tax": ("taxId", "taxId"),
    },
    "heterodimer": {
        "csv": "heterodimer_metadata.csv",
        "uniprot": ("uniprot_ac_1", "uniprot_ac_2"),
        "gene": ("gene_name_1", "gene_name_2"),
        "tax": ("tax_id_1", "tax_id_2"),
    },
}


def _f(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key) or "nan")
    except ValueError:
        return float("nan")


def _pair_max(row: dict[str, str], a: str, b: str) -> float:
    return max(_f(row, a), _f(row, b))


def passes_gate(row: dict[str, str]) -> bool:
    """The release's gate, recomputed from raw columns for both tables."""
    return (
        _pair_max(row, "ipSAE_AB", "ipSAE_BA") >= IPSAE_MIN
        and _pair_max(row, "pDockQ2_AB", "pDockQ2_BA") >= PDOCKQ2_MIN
    )


def stream_rows(url: str, max_bytes: int | None) -> Iterator[tuple[dict[str, str], int]]:
    """Yield (row, bytes_consumed) streaming the CSV without buffering it.

    Holds one partial line between reads. Yields the byte total so the caller can
    report progress against Content-Length without a second pass.
    """
    with urllib.request.urlopen(urllib.request.Request(url), timeout=300) as response:
        columns: list[str] | None = None
        pending = b""
        consumed = 0
        while True:
            chunk = response.read(CHUNK_BYTES)
            if not chunk:
                break
            consumed += len(chunk)
            pending += chunk
            lines = pending.split(b"\n")
            pending = lines.pop()
            if columns is None:
                columns = lines.pop(0).decode().split(",")
            if not lines:
                continue
            text = b"\n".join(lines).decode("utf-8", "replace")
            for row in csv.DictReader(io.StringIO(text), fieldnames=columns):
                yield row, consumed
            if max_bytes is not None and consumed >= max_bytes:
                return
        if pending and columns is not None:
            text = pending.decode("utf-8", "replace")
            for row in csv.DictReader(io.StringIO(text), fieldnames=columns):
                yield row, consumed


def project(row: dict[str, str], kind: str) -> dict[str, Any]:
    spec = SPECS[kind]
    a, b = spec["uniprot"]
    ga, gb = spec["gene"]
    ta, tb = spec["tax"]
    return {
        "model_entity_id": row["modelEntityId"],
        "kind": kind,
        "uniprot_1": row.get(a, ""),
        "uniprot_2": row.get(b, ""),
        "gene_1": row.get(ga, ""),
        "gene_2": row.get(gb, ""),
        "tax_1": row.get(ta, ""),
        "tax_2": row.get(tb, ""),
        "ipTM": _f(row, "ipTM"),
        "ipSAE_max": _pair_max(row, "ipSAE_AB", "ipSAE_BA"),
        "pDockQ2_max": _pair_max(row, "pDockQ2_AB", "pDockQ2_BA"),
        "pDockQ": _f(row, "pDockQ"),
        "LIS_max": _pair_max(row, "LIS_AB", "LIS_BA"),
        "n_clash_backbone": _f(row, "N_clash_backbone"),
        "n_interactions": _f(row, "numberOfInteractions"),
        "local_tar_name": row.get("local_tar_name", ""),
    }


def enumerate_set(kind: str, out_dir: Path, max_bytes: int | None) -> dict[str, Any]:
    spec = SPECS[kind]
    url = FTP_BASE + spec["csv"]
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=120) as r:
        total_bytes = int(r.headers["Content-Length"])
    path = out_dir / f"confident_{kind}s.csv.gz"

    n_rows = n_pass = n_declared_true = n_disagree = 0
    bytes_read = 0
    start = time.time()
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        for row, consumed in stream_rows(url, max_bytes):
            n_rows += 1
            bytes_read = consumed
            keep = passes_gate(row)
            declared = row.get("passes_quality_threshold")
            if declared is not None:
                declared_bool = declared.strip() == "true"
                n_declared_true += declared_bool
                if declared_bool != keep:
                    # Fatal by design: applying this gate to the homodimers is
                    # only licensed by the recomputation being exact.
                    raise RuntimeError(
                        f"{kind} {row['modelEntityId']}: declared "
                        f"{declared_bool}, recomputed {keep}"
                    )
            if not keep:
                continue
            n_pass += 1
            writer.writerow(project(row, kind))
            if n_pass % 100_000 == 0:
                pct = 100 * consumed / total_bytes
                print(f"  [{kind}] {pct:5.1f}% {n_rows:,} rows {n_pass:,} pass", flush=True)

    elapsed = time.time() - start
    stats = {
        "csv": spec["csv"],
        "csv_bytes": total_bytes,
        "complete": max_bytes is None,
        "rows_read": n_rows,
        "rows_passing": n_pass,
        "pass_rate": round(n_pass / max(n_rows, 1), 6),
        "declared_true": n_declared_true if "hetero" in kind else None,
        "declared_vs_recomputed_disagreements": n_disagree,
        "elapsed_seconds": round(elapsed, 1),
        "bytes_read": bytes_read,
        "mb_per_second": round(bytes_read / max(elapsed, 1e-9) / 1e6, 2),
        "output": str(path),
        "output_bytes": path.stat().st_size,
    }
    print(f"[{kind}] {json.dumps(stats, indent=2)}", flush=True)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/curation"))
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="stop each stream after this many bytes (pilot mode; output is partial)",
    )
    parser.add_argument("--kinds", nargs="+", default=["heterodimer", "homodimer"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meta = {
        "gate": f"ipSAE_max >= {IPSAE_MIN} and pDockQ2_max >= {PDOCKQ2_MIN}",
        "sets": {kind: enumerate_set(kind, args.out, args.max_bytes) for kind in args.kinds},
    }
    (args.out / "curation_enumerate_summary.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
