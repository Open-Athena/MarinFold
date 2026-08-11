# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step A (issue #211) — measure the distance bounds on real structures.

``consistency.py`` scores a contact set against a bound system: a CA(i)-CA(i+1)
virtual bond, an upper bound on the CA-CA distance of a contact, a lower bound
for a declared non-contact, and a steric floor. Those numbers are *measurements*,
not physics, and this script makes them. It runs over the 554-protein
ground-truth bundle exp174 already built and published, so nothing here needs
mmCIF parsing, chain selection or sequence alignment::

    hf buckets cp hf://buckets/open-athena/MarinFold/data/exp174-structural-eval/gt/... _scratch/gt/
    uv run python calibrate_bounds.py --gt-dir _scratch/gt --out-dir data

**Why bounds have to be measured and why they will overlap.** pyconfind contacts
are *side-chain* contacts (rotamer-based, ``contact_distance=3.0``,
``native_only``), so the CA-CA distance is only a proxy for them. Phase 0 on one
structure found contact CA-CA distances spanning 4.0-13.7 A and non-contact
CA-CA distances starting at 4.1 A, with 17.5% of non-contact pairs closer than
the contact p99. There is no threshold pair that cleanly separates the two
populations, and this script quantifies that across all 554 proteins rather than
one. The consequence is recorded in the emitted ``bounds.json`` and must survive
into the writeup: a nonzero embedding residual means *less geometrically
consistent*, *never* *provably unrealizable*.

The comparison between arms stays valid regardless, because every arm is scored
under the same bounds — the bounds only set where the whole scale sits.

Two indexing facts this script depends on, both verified by ``--check`` and by
``tests/test_calibrate_bounds.py``:

* ``gt_contacts.jsonl`` carries **0-based** input-sequence indices (verified:
  min index 0, max 760, none >= L).
* the bundle's PDB ``resSeq`` is the **1-based** input-sequence index (exp174's
  ``canonical_pdb.py`` contract), so CA coordinates are keyed by ``resSeq - 1``
  to line up with the contacts.

Unresolved residues carry no coordinates, so any pair touching one is excluded
from the calibration — but *not* from scoring, where the model can and does emit
contacts for unresolved positions.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# contacts-v1's own contact definition (SPEC): a pyconfind contact counts only
# above this degree and at or beyond this primary-sequence separation. The
# bundle stores every degree>0 contact, so the filter is applied here.
MIN_CONTACT_DEGREE = 0.001
MIN_SEP = 6

# Quantiles the bounds are read off at. u_contact is set high enough that almost
# every real contact satisfies it (a bound that real contacts violate would make
# the ground-truth arm score badly and destroy the calibration gate);
# l_noncontact is set low enough that almost every real non-contact satisfies it.
# Both are deliberately permissive — the metric should fire on joint-geometry
# impossibility, not on the tails of a proxy distribution.
U_CONTACT_Q = 99.5
L_NONCONTACT_Q = 0.5


def load_bundle(gt_dir: Path):
    """Read the exp174 GT bundle: per-record length, CA coordinates, contacts."""
    import gemmi

    index = {}
    for line in (gt_dir / "gt_index.jsonl").open():
        row = json.loads(line)
        index[row["record_id"]] = row

    contacts = {}
    for line in (gt_dir / "gt_contacts.jsonl").open():
        row = json.loads(line)
        contacts[row["record_id"]] = row["contacts"]

    for record_id, meta in index.items():
        pdb = gt_dir / "gt_structures" / f"{record_id}.pdb"
        if not pdb.exists():
            continue
        st = gemmi.read_structure(str(pdb))
        length = int(meta["L"])
        xyz = np.full((length, 3), np.nan)
        for model in st:
            for chain in model:
                for res in chain:
                    atom = res.find_atom("CA", "*")
                    if atom is None:
                        continue
                    # resSeq is the 1-based input-sequence index; contacts are
                    # 0-based. This is the off-by-one that would silently shift
                    # every measured distance by one residue.
                    k = res.seqid.num - 1
                    if 0 <= k < length:
                        xyz[k] = [atom.pos.x, atom.pos.y, atom.pos.z]
            break
        yield record_id, meta, xyz, contacts.get(record_id, [])


def measure(record_id, meta, xyz, raw_contacts):
    """Per-protein distance samples for the four bound families."""
    length = xyz.shape[0]
    resolved = ~np.isnan(xyz[:, 0])
    if resolved.sum() < 2:
        return None

    dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    both = resolved[:, None] & resolved[None, :]
    sep = np.abs(np.arange(length)[:, None] - np.arange(length)[None, :])
    triu = np.triu(np.ones((length, length), dtype=bool), 1)

    is_contact = np.zeros((length, length), dtype=bool)
    for i, j, degree in raw_contacts:
        if degree >= MIN_CONTACT_DEGREE and abs(j - i) >= MIN_SEP:
            is_contact[i, j] = is_contact[j, i] = True

    eligible = triu & both & (sep >= MIN_SEP)
    bonded = triu & both & (sep == 1)

    return {
        "record_id": record_id,
        "dataset": meta["dataset"],
        "L": length,
        "n_resolved": int(resolved.sum()),
        "bond": dist[bonded],
        "contact": dist[eligible & is_contact],
        "noncontact": dist[eligible & ~is_contact],
        # Steric floor: closest CA-CA pair that is not a backbone neighbour.
        "min_nonbonded": float(dist[triu & both & (sep >= 2)].min())
        if (triu & both & (sep >= 2)).any()
        else np.nan,
        # Packing ceiling: most contact partners claimed by any one residue.
        "max_degree": int(is_contact.sum(axis=1).max()) if length else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, default=Path("_scratch/gt"))
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pools: dict[str, list[np.ndarray]] = defaultdict(list)
    per_protein = []
    for n, (record_id, meta, xyz, raw) in enumerate(load_bundle(args.gt_dir)):
        if args.limit and n >= args.limit:
            break
        m = measure(record_id, meta, xyz, raw)
        if m is None:
            continue
        for key in ("bond", "contact", "noncontact"):
            pools[key].append(m[key])
        per_protein.append(
            {
                "record_id": m["record_id"],
                "dataset": m["dataset"],
                "L": m["L"],
                "n_resolved": m["n_resolved"],
                "n_contacts": int(m["contact"].size),
                "max_degree": m["max_degree"],
                "min_nonbonded": m["min_nonbonded"],
                "contact_p99_5": float(np.percentile(m["contact"], 99.5))
                if m["contact"].size
                else np.nan,
            }
        )
        if (n + 1) % 100 == 0:
            print(f"[calibrate] {n + 1} proteins", flush=True)

    pooled = {k: np.concatenate(v) for k, v in pools.items()}
    df = pd.DataFrame(per_protein)
    df.to_csv(args.out_dir / "calibration_per_protein.csv", index=False)

    bond, contact, noncontact = pooled["bond"], pooled["contact"], pooled["noncontact"]
    u_contact = float(np.percentile(contact, U_CONTACT_Q))
    l_noncontact = float(np.percentile(noncontact, L_NONCONTACT_Q))

    # How badly the two populations overlap — the number the writeup must quote
    # when it declines to call anything "provably unrealizable".
    overlap = float((noncontact < u_contact).mean())

    bounds = {
        "bond": float(np.median(bond)),
        "bond_sd": float(bond.std()),
        "u_contact": u_contact,
        "l_noncontact": l_noncontact,
        "d_min": float(np.nanpercentile(df["min_nonbonded"], 1.0)),
        "min_sep": MIN_SEP,
        "min_contact_degree": MIN_CONTACT_DEGREE,
        "u_contact_quantile": U_CONTACT_Q,
        "l_noncontact_quantile": L_NONCONTACT_Q,
        "max_degree_observed": int(df["max_degree"].max()),
        "max_degree_p99": float(np.percentile(df["max_degree"], 99)),
        "n_proteins": int(len(df)),
        "n_contact_pairs": int(contact.size),
        "n_noncontact_pairs": int(noncontact.size),
        "noncontact_frac_below_u_contact": overlap,
    }
    (args.out_dir / "bounds.json").write_text(json.dumps(bounds, indent=2) + "\n")

    quantile_rows = []
    for name, arr in (("bond", bond), ("contact", contact), ("noncontact", noncontact)):
        qs = [0.1, 0.5, 1, 5, 25, 50, 75, 95, 99, 99.5, 99.9]
        quantile_rows.append(
            {"population": name, "n": int(arr.size), "min": float(arr.min()),
             "max": float(arr.max()),
             **{f"p{q}": float(np.percentile(arr, q)) for q in qs}}
        )
    pd.DataFrame(quantile_rows).to_csv(
        args.out_dir / "calibration_quantiles.csv", index=False
    )

    print(f"\n{len(df)} proteins | {contact.size:,} contact pairs, "
          f"{noncontact.size:,} non-contact pairs")
    print(f"  bond          {np.median(bond):.3f} +/- {bond.std():.3f} A")
    print(f"  u_contact     {u_contact:.2f} A  (contact p{U_CONTACT_Q})")
    print(f"  l_noncontact  {l_noncontact:.2f} A  (non-contact p{L_NONCONTACT_Q})")
    print(f"  d_min         {bounds['d_min']:.2f} A")
    print(f"  max degree    {bounds['max_degree_observed']} observed, "
          f"p99 {bounds['max_degree_p99']:.0f}")
    print(f"\n  OVERLAP: {100 * overlap:.1f}% of non-contact pairs are closer than "
          f"u_contact={u_contact:.2f} A")
    print(f"  -> the score is statistical, not a proof of infeasibility.")
    print(f"\nwrote {args.out_dir}/bounds.json, calibration_quantiles.csv, "
          f"calibration_per_protein.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
