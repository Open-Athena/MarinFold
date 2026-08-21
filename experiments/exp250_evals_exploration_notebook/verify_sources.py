# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Check every published artifact ``notebooks/evals_exploration.ipynb`` reads.

The notebook has no dependencies of its own beyond the bucket: if a cell starts
failing, the likely cause is that one of these files moved, lost a column, or
changed shape — not the notebook. This script fetches each one, asserts the
columns and row counts the notebook relies on, and writes
``data/source_check.json``.

Deliberately dependency-light: pandas / pyarrow and the standard library, the
same surface the notebook itself uses. Run it from this directory::

    uv run --with pandas --with pyarrow python verify_sources.py
"""

import argparse
import io
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

BUCKET = "https://huggingface.co/buckets/open-athena/MarinFold/resolve"
EXP245 = f"{BUCKET}/data/contacts-v1-foldbench-monomers-exp245"
EXP89 = f"{BUCKET}/data/contacts-v1-model-eval-exp89"
EXP247 = f"{BUCKET}/data/contacts-v1-protein-properties-exp247"
EXP199_COOLDOWN_ROWS = (
    f"{BUCKET}/data/contacts-v1-model-eval-exp199/replicates/cooldown-v2-20260815-01/derived/"
    "prot-exp199-cw-cv1-p06-cool-s01/step-290400/contact_eval_cw_p06_cool_step290400_rows.csv.gz"
)
# The legacy set's 554 input sequences are not on the bucket; this in-repo FASTA is what the
# notebook prompts with, verified byte-identical against #89's own prompts (see README section 2).
LEGACY_FASTA = Path(__file__).resolve().parents[1] / (
    "exp94_evals_sequence_knn_baseline/data/eval_queries.fasta")


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()


def check_parquet(url: str, columns: set[str], rows: int) -> dict:
    table = pq.read_table(io.BytesIO(fetch(url)))
    missing = sorted(columns - set(table.column_names))
    return {"rows": table.num_rows, "expected_rows": rows, "missing_columns": missing,
            "ok": not missing and table.num_rows == rows}


def check_csv(url: str, columns: set[str], rows: int | None = None, **read_options) -> dict:
    frame = pd.read_csv(io.BytesIO(fetch(url)), **read_options)
    missing = sorted(columns - set(frame.columns))
    return {"rows": len(frame), "expected_rows": rows, "missing_columns": missing,
            "ok": not missing and (rows is None or len(frame) == rows)}


def check_jsonl(url: str, keys: set[str], rows: int) -> dict:
    lines = fetch(url).decode().splitlines()
    missing = sorted(keys - set(json.loads(lines[0]))) if lines else sorted(keys)
    return {"rows": len(lines), "expected_rows": rows, "missing_columns": missing,
            "ok": not missing and len(lines) == rows}


def check_fasta(path: Path, records: int) -> dict:
    if not path.exists():
        return {"rows": 0, "expected_rows": records, "missing_columns": ["<file>"], "ok": False}
    count = sum(1 for line in path.read_text().splitlines() if line.startswith(">"))
    return {"rows": count, "expected_rows": records, "missing_columns": [], "ok": count == records}


def checks() -> dict[str, dict]:
    """Every source, with the shape the notebook assumes of it."""
    return {
        "exp245 targets": check_parquet(
            f"{EXP245}/eval_targets_foldbench_monomers.parquet",
            {"dataset", "stem", "L", "input_seq"}, rows=333),
        "exp245 eval sets": check_csv(
            f"{EXP245}/eval_sets.csv",
            {"stem", "eval_set", "designed", "is_viral", "kingdom", "title", "deposit_date",
             "exp199_best_identity", "exp199_stratum"}, rows=334),
        "exp245 ground truth": check_jsonl(
            f"{EXP245}/gt_universe_scored.jsonl",
            {"dataset", "stem", "L", "resolved", "contacts"}, rows=333),
        "exp245 per-protein scores": check_csv(
            f"{EXP245}/per_protein.csv.gz",
            {"stem", "range", "cut", "predictor", "precision"}, rows=10_316,
            compression="gzip"),
        "exp89 ground truth": check_jsonl(
            f"{EXP89}/gt_universe.jsonl",
            {"dataset", "stem", "L", "resolved", "contacts"}, rows=554),
        "exp89 per-protein scores": check_csv(
            f"{EXP89}/contact_precision_all.csv",
            {"dataset", "stem", "model", "mode", "predictor", "range", "cut", "precision"},
            rows=73_128),
        "exp199 cooldown rows": check_csv(
            EXP199_COOLDOWN_ROWS,
            {"dataset", "stem", "predictor", "range", "cut", "precision"}, compression="gzip"),
        "exp247 features": check_csv(
            f"{EXP247}/protein_features.csv",
            {"stem", "relative_contact_order", "frac_helix", "frac_sheet", "msa_log_depth",
             "knn_best_identity", "n_pfam"}, rows=314),
        "exp94 legacy sequences": check_fasta(LEGACY_FASTA, records=554),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/source_check.json"))
    arguments = parser.parse_args()

    results = {}
    for name, check in checks().items():
        results[name] = check
        status = "ok " if check["ok"] else "FAIL"
        detail = "" if check["ok"] else f"  missing={check['missing_columns']}"
        print(f"[{status}] {name}: {check['rows']} rows "
              f"(expected {check['expected_rows']}){detail}")

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(
        {"checked_utc": datetime.now(UTC).isoformat(), "results": results}, indent=2) + "\n")
    failures = [name for name, check in results.items() if not check["ok"]]
    if failures:
        print(f"\n{len(failures)} source(s) no longer match what the notebook assumes: {failures}")
        return 1
    print(f"\nall {len(results)} sources ok -> {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
