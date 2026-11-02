# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Profile the confidence-score distributions of the AFDB dimer release.

Answers "how many complexes survive gate X" for a grid of thresholds on every
interface score the release ships (ipTM, pDockQ, pDockQ2, ipSAE, LIS), plus the
authors' own combined gate, without downloading either metadata CSV in full
(6.00 GB homodimer, 2.45 GB heterodimer).

Method: HTTP range requests at ``--windows`` evenly spaced byte offsets, keeping
only complete lines from each window. The two facts that make this sound:

1. The CSVs are fixed-schema and near-fixed-width (279.0 bytes per homodimer row,
   323.1 per heterodimer row, measured over 200 windows), so
   ``total_bytes / mean_row_bytes`` estimates the row count to about 1%.
2. Nothing else about a row correlates with its byte offset *within* an organism
   block, so a window is a fair sample of its block.

What is NOT sound is treating windows as independent draws from the whole file.
**Rows are blocked by organism**, and confidence varies enormously between
organisms, so a window is a cluster sample. Every output therefore carries the
per-window spread (min / median / max window pass rate) next to the pooled rate:
the pooled rate is the point estimate, and the spread is what says how much to
trust it. Use many small windows rather than few large ones for the same reason.

Usage::

    python3 profile_confidence.py --windows 200 --window-bytes 200000 --out data

Writes four files into ``--out``:

- ``nvda_confidence_gates.csv``: pass rate per metric per threshold, with the
  per-window spread and an estimated absolute count.
- ``nvda_confidence_quantiles.csv``: the underlying distributions, including the
  fraction scoring exactly zero and the maximum observed value.
- ``nvda_confidence_windows.csv``: one row per window (offset, dominant organism,
  headline pass rates), which is the evidence for the blocking caveat.
- ``nvda_confidence_sampling.json``: row-count estimates, redundancy counts, and
  the check of the recomputed gate against the declared verdict column.
"""

import argparse
import csv
import io
import json
import math
import statistics
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

FTP_BASE = "https://ftp.ebi.ac.uk/pub/databases/alphafold/collaborations/nvda/"

# Row = one predicted dimer. Values are the score columns as parsed floats; a
# missing or unparseable value becomes NaN and is excluded from that metric's
# denominator (reported as n_scored) rather than silently counted as a failure.
Row = dict[str, str]
Extractor = Callable[[Row], float]

# Every metric is swept on the same grid so the output reads as a matrix and a
# reader can compare metrics at a fixed threshold. The grid includes each score's
# own literature threshold:
#   ipTM      0.8 is AlphaFold-Multimer's own "confident interface" line.
#   pDockQ    0.23 = acceptable (DockQ >= 0.23), 0.5 = high confidence
#             (Bryant et al. 2022, https://doi.org/10.1038/s41467-022-28865-w).
#   pDockQ2   0.23 is the threshold the release itself uses.
#   ipSAE     0.6 is the release's gate; the ipSAE preprint favours ~0.75.
#   LIS       0.203 is the LIS paper's cutoff (Kim et al. 2024), added as an extra
#             because rounding it to 0.2 would misreport that specific gate.
COMMON_GRID = (0.23, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9)
EXTRA_THRESHOLDS: dict[str, tuple[float, ...]] = {"LIS_max": (0.203,)}

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def _get(url: str, headers: dict[str, str] | None = None, timeout: int = 180) -> bytes:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def content_length(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=120) as response:
        return int(response.headers["Content-Length"])


def fetch_window(url: str, columns: list[str], start: int, size: int) -> list[Row]:
    """Read one byte window and parse its complete lines.

    The first and last lines of a window are partial records and are dropped,
    which costs two rows out of roughly seven hundred.
    """
    chunk = _get(url, {"Range": f"bytes={start}-{start + size - 1}"})
    lines = chunk.split(b"\n")[1:-1]
    if not lines:
        return []
    text = b"\n".join(lines).decode("utf-8", "replace")
    return list(csv.DictReader(io.StringIO(text), fieldnames=columns))


def sample_windows(
    url: str, n_windows: int, window_bytes: int, workers: int
) -> tuple[list[list[Row]], list[int], int, float]:
    """Sample ``n_windows`` evenly spaced windows.

    Returns (windows, their byte offsets, file size in bytes, mean bytes per
    row). Windows are fetched concurrently but returned in offset order so runs
    are reproducible.
    """
    total = content_length(url)
    header = _get(url, {"Range": "bytes=0-8191"}).split(b"\n")[0].decode()
    columns = header.split(",")
    offsets = [int(total * (i + 0.5) / n_windows) for i in range(n_windows)]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        fetched = list(pool.map(lambda o: fetch_window(url, columns, o, window_bytes), offsets))

    kept = [(o, w) for o, w in zip(offsets, fetched) if w]
    offsets = [o for o, _ in kept]
    windows = [w for _, w in kept]
    n_rows = sum(len(w) for w in windows)
    # Bytes per row measured from the sample itself, header excluded: the row
    # count estimate is total_bytes / this.
    sampled_bytes = sum(
        len(",".join(row.get(c) or "" for c in columns)) + 1 for w in windows for row in w
    )
    return windows, offsets, total, sampled_bytes / max(n_rows, 1)


def _f(row: Row, key: str) -> float:
    try:
        return float(row.get(key) or "nan")
    except ValueError:
        return float("nan")


def _pair(row: Row, a: str, b: str, how: Callable[[float, float], float]) -> float:
    x, y = _f(row, a), _f(row, b)
    if math.isnan(x) or math.isnan(y):
        return float("nan")
    return how(x, y)


# ipSAE and pDockQ2 are directional: computed once per chain order. Reporting
# both the max (the permissive reading, and the one the release's own
# max_ipSAE / max_pDockQ2_AB columns take) and the min (the symmetric reading,
# which demands both directions agree) is the point of having two rows.
EXTRACTORS: dict[str, Extractor] = {
    "ipTM": lambda r: _f(r, "ipTM"),
    "pDockQ": lambda r: _f(r, "pDockQ"),
    "pDockQ2_max": lambda r: _pair(r, "pDockQ2_AB", "pDockQ2_BA", max),
    "ipSAE_max": lambda r: _pair(r, "ipSAE_AB", "ipSAE_BA", max),
    "ipSAE_min": lambda r: _pair(r, "ipSAE_AB", "ipSAE_BA", min),
    "LIS_max": lambda r: _pair(r, "LIS_AB", "LIS_BA", max),
}


def authors_gate(row: Row) -> bool:
    """The release's own heterodimer criterion: ipSAE >= 0.6 and pDockQ2 >= 0.23.

    Recomputed from the raw score columns rather than read off the heterodimer
    table's ``passes_quality_threshold``, so the identical criterion can be
    applied to homodimers, whose table has no verdict column.
    """
    return (
        _pair(row, "ipSAE_AB", "ipSAE_BA", max) >= 0.6
        and _pair(row, "pDockQ2_AB", "pDockQ2_BA", max) >= 0.23
    )


def clean_gate(row: Row) -> bool:
    """ipTM >= 0.8 plus no backbone clashes: a confident *and* physical model."""
    return _f(row, "ipTM") >= 0.8 and _f(row, "N_clash_backbone") == 0.0


COMBINED: dict[str, Callable[[Row], bool]] = {
    "authors_gate (ipSAE>=0.6 & pDockQ2>=0.23)": authors_gate,
    "ipTM>=0.8 & zero backbone clashes": clean_gate,
    "declared passes_quality_threshold": lambda r: (
        (r.get("passes_quality_threshold") or "").strip() == "true"
    ),
}


def gate_stats(windows: list[list[Row]], predicate: Callable[[Row], bool]) -> dict[str, Any]:
    """Pooled pass rate plus the per-window spread that the blocking demands."""
    rates, n_pass, n_rows = [], 0, 0
    for window in windows:
        hits = sum(1 for row in window if predicate(row))
        n_pass += hits
        n_rows += len(window)
        rates.append(hits / len(window))
    return {
        "n_rows": n_rows,
        "n_pass": n_pass,
        "pass_rate": n_pass / max(n_rows, 1),
        "window_min": min(rates, default=0.0),
        "window_median": statistics.median(rates) if rates else 0.0,
        "window_max": max(rates, default=0.0),
    }


def gate_agreement(windows: list[list[Row]]) -> dict[str, int] | None:
    """Check ``authors_gate`` against the heterodimer table's own verdict column.

    The whole point of recomputing the gate is to apply it to homodimers, which
    ship no verdict column. That is only legitimate if the recomputation
    reproduces the declared verdict where a declared verdict exists, so compare
    them row by row rather than trusting matching totals.
    """
    rows = [row for window in windows for row in window if "passes_quality_threshold" in row]
    if not rows:
        return None
    declared = [(row.get("passes_quality_threshold") or "").strip() == "true" for row in rows]
    recomputed = [authors_gate(row) for row in rows]
    return {
        "n_compared": len(rows),
        "agree": sum(1 for d, r in zip(declared, recomputed) if d == r),
        "declared_true_recomputed_false": sum(
            1 for d, r in zip(declared, recomputed) if d and not r
        ),
        "declared_false_recomputed_true": sum(
            1 for d, r in zip(declared, recomputed) if r and not d
        ),
    }


def quantiles(values: Iterable[float]) -> dict[str, float]:
    scored = sorted(v for v in values if not math.isnan(v))
    if not scored:
        return {}
    # frac_zero and max matter as much as the percentiles here: ipSAE and LIS are
    # heavily zero-inflated (a failed interface scores exactly 0), and pDockQ's
    # sigmoid saturates well below 1, so a "pDockQ >= 0.8" gate selects nothing
    # for reasons that have nothing to do with the data.
    out = {
        "n_scored": len(scored),
        "mean": statistics.fmean(scored),
        "frac_zero": sum(1 for v in scored if v == 0.0) / len(scored),
        "max": scored[-1],
    }
    for q in QUANTILES:
        # Nearest-rank; the sample is large enough that interpolation is noise.
        out[f"p{int(q * 100)}"] = scored[min(int(q * len(scored)), len(scored) - 1)]
    return out


def redundancy(name: str, windows: list[list[Row]]) -> dict[str, Any]:
    """Measure how much of a set is the same protein appearing more than once.

    Distinct model IDs say nothing about distinct *structures*: the homodimer set
    is built from UniProt, which carries one accession per strain, so a conserved
    bacterial enzyme appears once per sequenced strain with a different accession
    and the same gene name. Runs of an identical gene name at consecutive byte
    offsets are the visible symptom, so measure the run lengths too, not just the
    distinct-value counts.

    Gene name is a loose key (``rplC`` in E. coli and in S. aureus are homologs,
    not copies), so these counts bound the redundancy from one side only. The
    structural answer is what Foldseek clustering gives.
    """
    id_key, gene_key, tax_key = (
        ("uniprotAccession", "gene", "taxId")
        if name == "homodimer"
        else ("modelEntityId", None, "tax_id_1")
    )
    rows = [row for window in windows for row in window]
    out: dict[str, Any] = {
        "n_rows": len(rows),
        "distinct_ids": len({row.get(id_key) for row in rows}),
    }
    if gene_key is None:
        # Heterodimers are keyed on a pair; the comparable key is the gene pair.
        pairs = {(row.get("gene_name_1"), row.get("gene_name_2")) for row in rows}
        out["distinct_gene_pairs"] = len(pairs)
        confident = [row for row in rows if authors_gate(row)]
        out["confident_rows"] = len(confident)
        out["confident_distinct_gene_pairs"] = len(
            {(row.get("gene_name_1"), row.get("gene_name_2")) for row in confident}
        )
        return out

    out["distinct_genes"] = len({row.get(gene_key) for row in rows})
    out["distinct_gene_taxon"] = len({(row.get(gene_key), row.get(tax_key)) for row in rows})
    out["distinct_organisms"] = len({row.get("organismScientificName") for row in rows})

    # Run lengths are computed per window: rows are only adjacent within a window.
    runs: list[int] = []
    for window in windows:
        current, length = None, 0
        for row in window:
            if row.get(gene_key) == current:
                length += 1
                continue
            if current is not None:
                runs.append(length)
            current, length = row.get(gene_key), 1
        if current is not None:
            runs.append(length)
    out["max_gene_run"] = max(runs, default=0)
    out["rows_in_repeated_gene_runs"] = round(
        sum(r for r in runs if r >= 2) / max(sum(runs), 1), 4
    )

    confident = [row for row in rows if authors_gate(row)]
    out["confident_rows"] = len(confident)
    out["confident_distinct_genes"] = len({row.get(gene_key) for row in confident})
    out["confident_distinct_gene_taxon"] = len(
        {(row.get(gene_key), row.get(tax_key)) for row in confident}
    )
    return out


def window_detail(name: str, windows: list[list[Row]], offsets: list[int]) -> list[dict]:
    """One row per window: where it landed, what organism it hit, how it scored.

    This is the evidence for the blocking caveat rather than an assertion of it.
    If confidence tracked organism, the headline gates will vary by orders of
    magnitude between windows and the extremes will be single-species runs.
    """
    organism_key = "organismScientificName" if name == "homodimer" else "tax_id_1"
    detail = []
    for index, (offset, window) in enumerate(zip(offsets, windows)):
        organisms = [row.get(organism_key) or "?" for row in window]
        top = max(set(organisms), key=organisms.count)
        detail.append({
            "dataset": name,
            "window": index,
            "byte_offset": offset,
            "n_rows": len(window),
            "top_organism": top,
            "top_organism_share": round(organisms.count(top) / len(window), 4),
            "iptm_ge_0.8": round(sum(1 for r in window if _f(r, "ipTM") >= 0.8) / len(window), 4),
            "authors_gate": round(sum(1 for r in window if authors_gate(r)) / len(window), 4),
        })
    return detail


def profile(
    name: str, csv_name: str, args: argparse.Namespace
) -> tuple[list[dict], list[dict], list[dict], dict]:
    url = FTP_BASE + csv_name
    print(f"[{name}] sampling {args.windows} x {args.window_bytes // 1000} kB from {csv_name}")
    windows, offsets, total_bytes, bytes_per_row = sample_windows(
        url, args.windows, args.window_bytes, args.workers
    )
    n_rows = sum(len(w) for w in windows)
    est_total = round(total_bytes / bytes_per_row)
    print(
        f"[{name}] {n_rows:,} rows from {len(windows)} windows; "
        f"{bytes_per_row:.1f} B/row -> {est_total:,} rows in the file"
    )

    gates: list[dict] = []
    for metric, extractor in EXTRACTORS.items():
        grid = sorted(set(COMMON_GRID + EXTRA_THRESHOLDS.get(metric, ())))
        # Present but all-NaN means the column does not exist in this table.
        if all(math.isnan(extractor(row)) for window in windows for row in window):
            print(f"[{name}] {metric}: column absent, skipped")
            continue
        for threshold in grid:
            stats = gate_stats(windows, lambda r, e=extractor, t=threshold: e(r) >= t)
            gates.append({"dataset": name, "metric": metric, "threshold": threshold, **stats})
    for label, predicate in COMBINED.items():
        stats = gate_stats(windows, predicate)
        if stats["n_pass"] == 0 and "passes_quality_threshold" in label:
            continue  # column absent (homodimers)
        gates.append({"dataset": name, "metric": label, "threshold": "", **stats})

    for gate in gates:
        gate["estimated_passing"] = round(gate["pass_rate"] * est_total)

    dists = []
    for metric, extractor in EXTRACTORS.items():
        q = quantiles(extractor(row) for window in windows for row in window)
        if q:
            dists.append({"dataset": name, "metric": metric, **q})

    meta = {
        "csv": csv_name,
        "csv_bytes": total_bytes,
        "windows": len(windows),
        "window_bytes": args.window_bytes,
        "rows_sampled": n_rows,
        "bytes_per_row": round(bytes_per_row, 1),
        "estimated_total_rows": est_total,
        "sampled_fraction": round(n_rows / max(est_total, 1), 6),
    }
    meta["redundancy"] = redundancy(name, windows)
    print(f"[{name}] redundancy: {meta['redundancy']}")
    agreement = gate_agreement(windows)
    if agreement:
        meta["authors_gate_vs_declared"] = agreement
        print(f"[{name}] recomputed gate vs declared column: {agreement}")
    return gates, dists, window_detail(name, windows, offsets), meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=int, default=200)
    parser.add_argument("--window-bytes", type=int, default=200_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    all_gates, all_dists, all_windows, meta = [], [], [], {}
    for name, csv_name in (
        ("homodimer", "homodimer_metadata.csv"),
        ("heterodimer", "heterodimer_metadata.csv"),
    ):
        gates, dists, detail, m = profile(name, csv_name, args)
        all_gates += gates
        all_dists += dists
        all_windows += detail
        meta[name] = m

    def write(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}
                )
        print(f"wrote {path} ({len(rows)} rows)")

    write(args.out / "nvda_confidence_gates.csv", all_gates)
    write(args.out / "nvda_confidence_quantiles.csv", all_dists)
    write(args.out / "nvda_confidence_windows.csv", all_windows)
    (args.out / "nvda_confidence_sampling.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    for dataset in ("homodimer", "heterodimer"):
        print(f"\n=== {dataset} ===")
        print(f"{'metric':<44}{'pass %':>8}{'window min-max':>18}{'estimated':>14}")
        for gate in all_gates:
            if gate["dataset"] != dataset:
                continue
            label = gate["metric"]
            if gate["threshold"] != "":
                label = f"{label} >= {gate['threshold']}"
            print(
                f"{label:<44}{gate['pass_rate'] * 100:>7.2f}%"
                f"{gate['window_min'] * 100:>9.1f}-{gate['window_max'] * 100:<8.1f}"
                f"{gate['estimated_passing']:>14,}"
            )


if __name__ == "__main__":
    main()
