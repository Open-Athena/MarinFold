# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Report the corpus-quality numbers for a batch of redesigned documents.

The three things #266's success criteria ask for, computed from a documents
parquet plus the staged backbones it came from:

* **contacts per residue, designed vs native** — the artifact the sequence
  sensitivity probe predicted (contact degree collapses for small side chains,
  so an Ala-rich design distribution would systematically shorten documents).
* **amino-acid composition drift** vs the native sequences.
* **per-temperature identity / score / density**, so the ladder can be
  subsetted later without regenerating.

Stratified by length, because the effect is not length-uniform: on the 200
shortest AFDB entries (mean L 47, low pLDDT) designed density came out *above*
native, the opposite of the corpus-wide direction.

    uv run python analyze_documents.py --documents docs.parquet \\
        --backbones backbones.parquet --out data/
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import pyarrow.parquet as pq

LENGTH_BINS = ((0, 100), (100, 200), (200, 400), (400, 800), (800, 10**6))


def _bin(length: int) -> str:
    for lo, hi in LENGTH_BINS:
        if lo <= length < hi:
            return f"{lo}-{hi if hi < 10**6 else '+'}"
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--documents", required=True)
    ap.add_argument("--backbones", required=True)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--label", default="rep")
    args = ap.parse_args()

    docs = pq.read_table(args.documents).to_pylist()
    bb = {r["entry_id"]: r for r in pq.read_table(args.backbones).to_pylist()}
    args.out.mkdir(parents=True, exist_ok=True)

    n_bb = len({d["entry_id"] for d in docs})
    print(f"{len(docs)} documents over {n_bb} backbones "
          f"({len(docs) / n_bb:.2f} per backbone); "
          f"{len({d['document'] for d in docs})} distinct")

    # --- density, overall and by length ---
    rows = []
    for label, keep in [("all", lambda L: True)] + [
        (_bin(lo), (lambda lo=lo, hi=hi: lambda L: lo <= L < hi)()) for lo, hi in LENGTH_BINS
    ]:
        nat, des = [], []
        for e, r in bb.items():
            L = len(r["sequence"])
            if not keep(L) or r.get("native_contacts_emitted") is None:
                continue
            nat.append(r["native_contacts_emitted"] / L)
        for d in docs:
            L = len(bb[d["entry_id"]]["sequence"])
            if keep(L):
                des.append(d["contacts_emitted"] / L)
        if not nat or not des:
            continue
        n, s = sum(nat) / len(nat), sum(des) / len(des)
        rows.append({"stratum": label, "n_backbones": len(nat), "n_documents": len(des),
                     "native_per_residue": round(n, 4), "designed_per_residue": round(s, 4),
                     "ratio": round(s / n, 4)})
        print(f"  {label:>8}  n={len(nat):5d}  native {n:.3f}  designed {s:.3f}  "
              f"ratio {s / n:.3f}")
    _write(args.out / f"density_{args.label}.csv", rows)

    # --- composition ---
    native_counts = collections.Counter("".join(r["sequence"] for r in bb.values()))
    design_counts = collections.Counter()
    for d in docs:
        design_counts.update(_sequence_of(d, bb))
    n_nat, n_des = sum(native_counts.values()), sum(design_counts.values())
    comp = []
    for aa in sorted(set(native_counts) | set(design_counts)):
        nat = 100 * native_counts[aa] / n_nat
        des = 100 * design_counts[aa] / n_des
        comp.append({"amino_acid": aa, "native_pct": round(nat, 2),
                     "design_pct": round(des, 2), "delta_pct": round(des - nat, 2)})
    comp.sort(key=lambda r: -abs(r["delta_pct"]))
    print("\n  largest composition shifts: " +
          ", ".join(f"{r['amino_acid']}{r['delta_pct']:+.2f}" for r in comp[:6]))
    _write(args.out / f"composition_{args.label}.csv", comp)

    # --- per-temperature ---
    by_t = collections.defaultdict(list)
    for d in docs:
        by_t[d["mpnn_temperature"]].append(d)
    temps = []
    print("\n     T      n   identity   score   contacts/res")
    for t in sorted(by_t):
        ds = by_t[t]
        ident = sum(d["identity_to_native"] for d in ds) / len(ds)
        score = sum(d["mpnn_score"] for d in ds) / len(ds)
        dens = sum(d["contacts_emitted"] / len(bb[d["entry_id"]]["sequence"])
                   for d in ds) / len(ds)
        temps.append({"temperature": t, "n": len(ds),
                      "mean_identity_to_native": round(ident, 4),
                      "mean_mpnn_score": round(score, 4),
                      "contacts_per_residue": round(dens, 4)})
        print(f"  {t:4.1f} {len(ds):6d}   {ident:.3f}    {score:.3f}      {dens:.3f}")
    _write(args.out / f"temperature_{args.label}.csv", temps)


def _sequence_of(doc: dict, bb: dict) -> str:
    """Recover a design's sequence from its own document's sequence section.

    Read back out of the document rather than trusting a side channel: this is
    the sequence the corpus actually asserts, so a composition drift measured
    here is a property of the shipped data.
    """
    from marinfold.document_structures.contacts_v1.read import sequence_from_document

    return sequence_from_document(
        doc["document"], doc["seq_len"], doc["n_term_index"]
    )


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
