# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the eval manifest that ``contact_eval.py`` consumes.

One row per protein: ``dataset, stem, gt_cif, gt_chain, input_seq`` (+ any
strata columns to carry through). ``input_seq`` is the exact sequence
Protenix was given — the canonical ``_entity_poly.pdbx_seq_one_letter_code_can``
(unresolved residues included), which is also the distogram's index space.

Two builders:
  - ``foldbench`` — from the exp12 HF bucket sync (manifest.csv + gt/).
  - ``exp65`` — from the exp65 candidate CSVs + fetched structures
    (wired up in the exp65 input-prep step).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import gemmi


def extract_canonical_sequence(cif_path: Path, chain_id: str | None = None) -> str:
    """Canonical one-letter sequence (incl. unresolved residues, modified->parent).

    Reads ``_entity_poly.pdbx_seq_one_letter_code_can`` via gemmi's cif
    reader (same path as exp12's prepare_inputs). For multi-entity files
    we pick the polypeptide(L) entity whose ``pdbx_strand_id`` lists
    ``chain_id`` when given, else the first polypeptide(L) entity.
    """
    doc = gemmi.cif.read(str(cif_path))
    block = doc.sole_block()
    rows = list(block.find(
        "_entity_poly.",
        ["entity_id", "type", "pdbx_seq_one_letter_code_can", "?pdbx_strand_id"],
    ))
    pep = [r for r in rows if "polypeptide(L)" in r[1]]
    if not pep:
        raise ValueError(f"no polypeptide(L) entity in {cif_path}")
    chosen = pep[0]
    if chain_id is not None and len(pep) > 1:
        for r in pep:
            try:
                strands = r.str(3)
            except Exception:  # noqa: BLE001 — column may be absent
                strands = ""
            if chain_id in {s.strip() for s in strands.split(",")}:
                chosen = r
                break
    return "".join(chosen.str(2).split())


def build_foldbench(hf_dir: Path, out_csv: Path) -> int:
    """Eval manifest for the FoldBench-100 set synced from the HF bucket."""
    hf_dir = Path(hf_dir)
    manifest = list(csv.DictReader((hf_dir / "manifest.csv").open()))
    rows: list[dict] = []
    for r in manifest:
        stem, chain = r["stem"], r["chain_id"]
        gt = hf_dir / "gt" / f"{stem}.cif"
        if not gt.exists():
            continue
        if not (hf_dir / "best" / "single_seq" / stem / "distogram.npz").exists():
            continue
        try:
            seq = extract_canonical_sequence(gt, chain)
        except Exception as e:  # noqa: BLE001
            print(f"skip {stem}: {e}")
            continue
        rows.append(dict(
            dataset="foldbench100", stem=stem, gt_cif=f"gt/{stem}.cif",
            gt_chain=chain, input_seq=seq, n_residues=r["n_residues"],
        ))
    _write(out_csv, rows)
    return len(rows)


def _write(out_csv: Path, rows: list[dict]) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}: {len(rows)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("foldbench")
    pf.add_argument("--hf-dir", type=Path, required=True)
    pf.add_argument("--out", type=Path, required=True)
    pf.set_defaults(func=lambda a: build_foldbench(a.hf_dir, a.out))
    args = ap.parse_args()
    args.func(args)
