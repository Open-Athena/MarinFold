# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch a small sample of high-confidence AFDB dimers.

The AFDB complex release (``collaborations/nvda/``) is 48.8 TB of tar shards, so
nothing here touches them. Two much cheaper facts make a sample possible:

1. The FTP metadata CSVs (``homodimer_metadata.csv`` 6.0 GB,
   ``heterodimer_metadata.csv`` 2.45 GB) support HTTP range requests, so a few
   hundred kB of windows is enough to harvest model IDs and their scores.
2. Every model is individually downloadable from the ordinary AFDB file endpoint,
   ``https://alphafold.ebi.ac.uk/files/AF-<id>-model_v1.cif``. Measured over ten
   sampled models: 115 kB on the wire when gzip is requested, 466 kB of mmCIF
   decoded. The server only compresses if the client sends
   ``Accept-Encoding: gzip``, which urllib does not do by default.

Windows are spread evenly across each CSV on purpose. The rows are blocked by
organism, so consecutive rows share a species and confidence correlates strongly
with which block you land in; a single contiguous read would sample one organism.

``MAX_STRUCTURES`` is a hard ceiling that raises rather than truncates. This script
is for smoke tests and spot checks. A bulk download of the confident slice is
~1.96 M homodimers plus ~70 k heterodimers, which at the sizes above is **roughly
230 GB gzipped and 950 GB decoded**; that needs a deliberate separate job with
rate limiting, not an accidentally large ``--n-hetero``.

Usage::

    python3 fetch_dimer_sample.py --n-hetero 30 --n-homo 15 --out sample
"""

import argparse
import csv
import gzip
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator

FTP_BASE = "https://ftp.ebi.ac.uk/pub/databases/alphafold/collaborations/nvda/"
FILES_BASE = "https://alphafold.ebi.ac.uk/files/"

# Every model in the release carries version 1 (the API reports latestVersion 1
# for the complex entities). There is no version column in the metadata CSVs.
MODEL_VERSION = 1

# Refuse to be turned into a bulk downloader. See the module docstring.
MAX_STRUCTURES = 250

WINDOW_BYTES = 300_000


def _get(url: str, headers: dict[str, str] | None = None, timeout: int = 180) -> bytes:
    """GET a URL, transparently gunzipping a gzip-encoded body.

    urllib neither requests nor decodes Content-Encoding, so callers that want
    compression ask for it explicitly and this function sniffs the gzip magic
    number on the way back rather than trusting headers (the FTP host and the
    files host set them differently).

    Range requests deliberately do NOT ask for gzip: a byte range against a
    compressed representation would not be the byte range the caller means.
    """
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    if body[:2] == b"\x1f\x8b":
        return gzip.decompress(body)
    return body


def content_length(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=120) as response:
        return int(response.headers["Content-Length"])


def iter_windows(url: str, n_windows: int) -> Iterator[list[dict[str, str]]]:
    """Yield parsed CSV rows from ``n_windows`` evenly spaced byte windows.

    The first and last lines of each window are partial records and are dropped,
    which costs two rows out of roughly a thousand.
    """
    total = content_length(url)
    header = _get(url, {"Range": "bytes=0-4095"}).split(b"\n")[0].decode()
    columns = header.split(",")
    for i in range(n_windows):
        start = int(total * (i + 0.5) / n_windows)
        end = start + WINDOW_BYTES - 1
        chunk = _get(url, {"Range": f"bytes={start}-{end}"})
        lines = chunk.split(b"\n")[1:-1]
        if not lines:
            continue
        text = b"\n".join(lines).decode("utf-8", "replace")
        yield list(csv.DictReader(io.StringIO(text), fieldnames=columns))


def _to_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key) or "nan")
    except ValueError:
        return float("nan")


def _pair_max(row: dict[str, str], a: str, b: str) -> float:
    return max(_to_float(row, a), _to_float(row, b))


def is_confident(row: dict[str, str]) -> bool:
    """The release's own gate: ipSAE >= 0.6 and pDockQ2 >= 0.23.

    The heterodimer table ships this verdict as ``passes_quality_threshold``; the
    homodimer table ships no verdict column at all. Recomputing it from the raw
    score columns therefore gives both sets one criterion. ``profile_confidence.py``
    checks the recomputation against the declared column and found 123,597 of
    123,597 heterodimer rows in agreement, zero disagreements in either
    direction, so this is the authors' gate rather than an approximation of it.

    It passes 0.92% of heterodimers (~70 k) and 9.1% of homodimers (~1.96 M).
    """
    return (
        _pair_max(row, "ipSAE_AB", "ipSAE_BA") >= 0.6
        and _pair_max(row, "pDockQ2_AB", "pDockQ2_BA") >= 0.23
    )


def passes_iptm(row: dict[str, str], ipTM_min: float) -> bool:
    """Alternative gate on ipTM alone, for ``--gate iptm``.

    Not the same population: ipTM >= 0.8 passes 5.9% of homodimers against the
    authors' gate's 9.1%, and the two sets overlap only partly, because ipSAE
    rescales the interface score by the size of the interface being scored.
    """
    return _to_float(row, "ipTM") >= ipTM_min


def harvest(
    csv_name: str,
    keep: Callable[[dict[str, str]], bool],
    n_wanted: int,
    n_windows: int,
) -> list[dict[str, str]]:
    """Collect up to ``n_wanted`` rows passing ``keep``, spread across windows."""
    url = FTP_BASE + csv_name
    per_window = max(1, n_wanted // n_windows + 1)
    kept: list[dict[str, str]] = []
    scanned = 0
    for rows in iter_windows(url, n_windows):
        scanned += len(rows)
        hits = [r for r in rows if keep(r)]
        kept.extend(hits[:per_window])
        print(
            f"  [{csv_name}] window: {len(rows)} rows, {len(hits)} pass, "
            f"{len(kept)}/{n_wanted} collected",
            flush=True,
        )
        if len(kept) >= n_wanted:
            break
    print(f"  [{csv_name}] scanned {scanned:,} rows to collect {len(kept)}")
    return kept[:n_wanted]


def download_model(model_id: str, out_dir: Path, suffix: str) -> Path:
    """Download one model coordinate file. Returns the written path."""
    url = f"{FILES_BASE}{model_id}-model_v{MODEL_VERSION}.{suffix}"
    path = out_dir / f"{model_id}.{suffix}"
    if path.exists() and path.stat().st_size > 0:
        return path
    # Asking for gzip cuts the transfer about 4x (115 kB against 466 kB measured);
    # _get gunzips it, so the file on disk is plain mmCIF either way.
    path.write_bytes(_get(url, {"Accept-Encoding": "gzip"}))
    return path


def build_record(row: dict[str, str], kind: str) -> dict[str, Any]:
    """Flatten a metadata row into the manifest schema shared by both sets."""
    if kind == "heterodimer":
        partners = [row.get("uniprot_ac_1", ""), row.get("uniprot_ac_2", "")]
        genes = [row.get("gene_name_1", ""), row.get("gene_name_2", "")]
        taxa = [row.get("tax_id_1", ""), row.get("tax_id_2", "")]
    else:
        partners = [row.get("uniprotAccession", "")] * 2
        genes = [row.get("gene", "")] * 2
        taxa = [row.get("taxId", "")] * 2
    return {
        "model_entity_id": row["modelEntityId"],
        "kind": kind,
        "uniprot_1": partners[0],
        "uniprot_2": partners[1],
        "gene_1": genes[0],
        "gene_2": genes[1],
        "tax_1": taxa[0],
        "tax_2": taxa[1],
        "ipTM": _to_float(row, "ipTM"),
        "ipSAE_AB": _to_float(row, "ipSAE_AB"),
        "ipSAE_BA": _to_float(row, "ipSAE_BA"),
        "pDockQ": _to_float(row, "pDockQ"),
        "pDockQ2_AB": _to_float(row, "pDockQ2_AB"),
        "n_clash_backbone": _to_float(row, "N_clash_backbone"),
        "n_interactions": _to_float(row, "numberOfInteractions"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-hetero", type=int, default=30, help="heterodimers to fetch")
    parser.add_argument("--n-homo", type=int, default=15, help="homodimers to fetch")
    parser.add_argument("--windows", type=int, default=10, help="byte windows per CSV")
    parser.add_argument(
        "--gate",
        choices=["authors", "iptm"],
        default="authors",
        help="confidence gate: the release's ipSAE/pDockQ2 criterion, or ipTM alone",
    )
    parser.add_argument("--iptm-min", type=float, default=0.8, help="only used by --gate iptm")
    parser.add_argument("--format", choices=["cif", "pdb"], default="cif")
    parser.add_argument("--out", type=Path, default=Path("sample"))
    args = parser.parse_args()

    total = args.n_hetero + args.n_homo
    if total > MAX_STRUCTURES:
        # Deliberately fatal. See MAX_STRUCTURES in the module docstring.
        raise SystemExit(
            f"refusing to fetch {total} structures; this script is capped at "
            f"{MAX_STRUCTURES}. A bulk pull of the confident slice needs its own "
            f"job with rate limiting."
        )

    struct_dir = args.out / "structures"
    struct_dir.mkdir(parents=True, exist_ok=True)

    if args.gate == "authors":
        keep, label = is_confident, "ipSAE >= 0.6 and pDockQ2 >= 0.23"
    else:
        keep, label = lambda r: passes_iptm(r, args.iptm_min), f"ipTM >= {args.iptm_min}"

    print(f"harvesting heterodimer IDs (gate: {label})")
    hetero = harvest("heterodimer_metadata.csv", keep, args.n_hetero, args.windows)
    print(f"harvesting homodimer IDs (gate: {label})")
    homo = harvest("homodimer_metadata.csv", keep, args.n_homo, args.windows)

    records = [build_record(r, "heterodimer") for r in hetero]
    records += [build_record(r, "homodimer") for r in homo]

    print(f"\ndownloading {len(records)} structures to {struct_dir}")
    start = time.time()
    downloaded, failed = [], []
    for i, record in enumerate(records, 1):
        model_id = record["model_entity_id"]
        try:
            path = download_model(model_id, struct_dir, args.format)
        except urllib.error.HTTPError as exc:
            # One dead ID should not abandon the rest of the sample; record it so
            # the manifest and the structure dir stay consistent with each other.
            print(f"  [{i}/{len(records)}] {model_id} FAILED {exc.code}")
            failed.append({**record, "error": f"HTTP {exc.code}"})
            continue
        record["path"] = str(path.relative_to(args.out))
        record["bytes"] = path.stat().st_size
        downloaded.append(record)
        print(f"  [{i}/{len(records)}] {model_id} {record['kind']:12s} {record['bytes']:>8,} B")
    elapsed = time.time() - start

    manifest = args.out / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(downloaded[0].keys()))
        writer.writeheader()
        writer.writerows(downloaded)

    summary = {
        "n_heterodimers": sum(1 for r in downloaded if r["kind"] == "heterodimer"),
        "n_homodimers": sum(1 for r in downloaded if r["kind"] == "homodimer"),
        "n_failed": len(failed),
        "total_bytes": sum(r["bytes"] for r in downloaded),
        "download_seconds": round(elapsed, 1),
        "format": args.format,
        "gate": args.gate,
        "gate_description": label,
    }
    (args.out / "fetch_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{json.dumps(summary, indent=2)}")
    print(f"manifest: {manifest}")
    if failed:
        print(f"{len(failed)} downloads failed", file=sys.stderr)


if __name__ == "__main__":
    main()
