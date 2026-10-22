# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch UniRef cluster sizes as a cheap homolog-depth proxy for exp324.

This is not an MSA depth measurement. It is a cheap sequence-family-size proxy:
for each UniProt accession already resolved in ``sample_10000_biology_features``
we query UniRef and record UniRef100/90/50 cluster sizes. For proteins with
multiple accessions, stem-level features take the max cluster size at each
identity threshold and keep the accession that supplied it.
"""

import argparse
import concurrent.futures
import io
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

URL = "https://rest.uniprot.org/uniref/search"
FIELDS = "id,name,count,identity"
IDENTITY_TO_COL = {1.0: "uniref100_size", 0.9: "uniref90_size", 0.5: "uniref50_size"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biology", type=Path, default=Path("experiments/exp324_evals_esmfold2_pdb_sample/data/sample_10000_biology_features.parquet"))
    parser.add_argument("--cache", type=Path, default=Path("experiments/exp324_evals_esmfold2_pdb_sample/data/annotation_cache/uniref_cluster_sizes.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("experiments/exp324_evals_esmfold2_pdb_sample/data/sample_10000_uniref_depth_proxy.csv"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def split_accessions(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def read_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out[row["accession"]] = row
    return out


def parse_tsv(accession: str, text: str) -> dict[str, Any]:
    base: dict[str, Any] = {"accession": accession, "status": "ok", "error": ""}
    if text.startswith("Error messages"):
        return {"accession": accession, "status": "error", "error": text[:500]}
    frame = pd.read_csv(io.StringIO(text), sep="\t")
    for _, row in frame.iterrows():
        identity = float(row.get("Identity"))
        col = IDENTITY_TO_COL.get(identity)
        if col is None:
            continue
        base[col] = int(row.get("Size"))
        base[col.replace("_size", "_cluster_id")] = row.get("Cluster ID")
        base[col.replace("_size", "_cluster_name")] = row.get("Cluster Name")
    for col in IDENTITY_TO_COL.values():
        base.setdefault(col, None)
        base.setdefault(col.replace("_size", "_cluster_id"), "")
        base.setdefault(col.replace("_size", "_cluster_name"), "")
    return base


def fetch_one(accession: str, retries: int = 4) -> dict[str, Any]:
    params = {"query": f"({accession})", "format": "tsv", "fields": FIELDS, "size": 10}
    for attempt in range(retries + 1):
        try:
            response = requests.get(URL, params=params, timeout=30, headers={"User-Agent": "MarinFold-exp324-uniref-depth/0.1"})
            if response.status_code == 200:
                return parse_tsv(accession, response.text)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            return {"accession": accession, "status": "error", "error": f"HTTP {response.status_code}: {response.text[:500]}"}
        except Exception as exc:  # noqa: BLE001
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            return {"accession": accession, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    raise AssertionError("unreachable")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def stem_level(biology: pd.DataFrame, accession_rows: pd.DataFrame) -> pd.DataFrame:
    by_acc = accession_rows.set_index("accession").to_dict("index")
    rows: list[dict[str, Any]] = []
    for record in biology[["stem", "uniprot_accessions"]].to_dict("records"):
        accs = split_accessions(record["uniprot_accessions"])
        row: dict[str, Any] = {"stem": record["stem"], "uniprot_accessions": ";".join(accs), "n_uniprot_accessions": len(accs)}
        for col in IDENTITY_TO_COL.values():
            best_size = pd.NA
            best_acc = ""
            best_cluster = ""
            values = []
            for acc in accs:
                acc_row = by_acc.get(acc)
                if not acc_row:
                    continue
                value = acc_row.get(col)
                if pd.isna(value):
                    continue
                value = int(value)
                values.append(value)
                if pd.isna(best_size) or value > int(best_size):
                    best_size = value
                    best_acc = acc
                    best_cluster = str(acc_row.get(col.replace("_size", "_cluster_id"), ""))
            row[col] = best_size
            row[col.replace("_size", "_best_accession")] = best_acc
            row[col.replace("_size", "_best_cluster_id")] = best_cluster
            row[col.replace("_size", "_mean")] = sum(values) / len(values) if values else pd.NA
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    biology = pd.read_parquet(args.biology)
    accessions = sorted({acc for value in biology["uniprot_accessions"] for acc in split_accessions(value)})
    if args.limit:
        accessions = accessions[: args.limit]
    cache = read_cache(args.cache)
    missing = [acc for acc in accessions if acc not in cache]
    print(f"accessions={len(accessions)} cached={len(accessions)-len(missing)} missing={len(missing)}")

    completed: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, acc): acc for acc in missing}
        for n, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            row = future.result()
            completed.append(row)
            if len(completed) >= 100:
                append_jsonl(args.cache, completed)
                cache.update({r["accession"]: r for r in completed})
                completed = []
            if n % 200 == 0 or n == len(missing):
                print(f"fetched {n}/{len(missing)}")
    if completed:
        append_jsonl(args.cache, completed)
        cache.update({r["accession"]: r for r in completed})

    accession_rows = pd.DataFrame([cache[acc] for acc in accessions if acc in cache])
    stem_rows = stem_level(biology, accession_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stem_rows.to_csv(args.out, index=False)
    accession_rows.to_csv(args.out.with_name(args.out.stem + ".accessions.csv"), index=False)
    print(
        json.dumps(
            {
                "stems": int(len(stem_rows)),
                "stems_with_uniref50": int(stem_rows["uniref50_size"].notna().sum()),
                "accessions": int(len(accession_rows)),
                "accession_errors": int((accession_rows["status"] != "ok").sum()) if "status" in accession_rows else 0,
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
