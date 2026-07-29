# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step A — build the full-atom ground-truth bundle for the 554-protein eval set.

Every MarinFold eval so far has needed ground-truth *contacts* for this set;
issue #174 is the first that needs ground-truth **coordinates**. This script
produces them, in the same 554-protein universe exp74/exp78/exp89 use and in
the canonical file contract of ``canonical_pdb.py``, so the scorer never has
to touch an mmCIF, a chain id, or a sequence alignment again.

Per protein:

1. Read the eval manifest row (stem, input sequence, GT mmCIF, preferred chain).
2. Extract the single protein chain (``chain_alignment.extract_single_chain``).
3. Run contacts-and-crops-v1's own :func:`analyze_coordinates` on it. This is
   the format's own parse path — canonical residue names, the 37-name heavy-atom
   vocabulary, hydrogens and out-of-vocabulary atoms dropped, alt-loc doubles
   collapsed — so "an atom the ground truth has" means exactly "an atom a
   contacts-and-crops-v1 document could have mentioned". It also returns the
   pyconfind contacts, which cost nothing extra here and which the conditioning
   regimes discussed in ``PLANS.md`` will want.
4. Align the resolved residues to the input sequence (difflib, as exp78/exp89)
   and renumber every atom to its **1-based input-sequence index**.
5. Write ``gt_structures/<dataset>/<stem>.pdb``; append a row to
   ``gt_index.jsonl`` and the remapped contacts to ``gt_contacts.jsonl``.

Outputs (all under ``--out-dir``):

* ``gt_structures/<dataset>/<stem>.pdb`` — canonical GT coordinates, one per
  eval-set record. The ``<dataset>`` level is load-bearing, not decoration:
  ``7ur7_A`` and ``8ah9_A`` appear in **both** the FoldBench-100 manifest and
  exp65's ``denovo_pdb`` set, with *different* input sequences and different
  ground-truth files, so ``stem`` alone is not unique over the 554 records.
  The record key used everywhere downstream is ``<dataset>/<stem>``.
* ``gt_index.jsonl`` — one row per record: lengths, atom counts, alignment
  quality, and the manifest strata, for the scorer's stratified reporting.
* ``gt_contacts.jsonl`` — every degree>0 pyconfind contact in input-sequence
  coordinates (the same quantity as exp89's ``gt_universe.jsonl``).

Run in this experiment's venv (it has pyconfind via ``marinfold[contacts-v1]``)::

    uv run python prepare_gt_structures.py --out-dir _scratch/gt

The bundle is checkpoint-independent: build it once, publish it to the HF
bucket with ``publish_gt_bundle.py``, and every later scoring run pulls it.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from canonical_pdb import build_atom_array, write_structure
from chain_alignment import align_obs_to_ref, extract_single_chain, one_letter
from marinfold.document_structures.contacts_and_crops_v1 import analyze_coordinates

# The eval-set manifests and staged ground-truth structures live in the exp78
# checkout that built them (exp89's prepare_gt_universe.py points at the same
# paths). The manifests are committed in this repo; the structures are not
# (they are third-party PDB/CAMEO/CASP downloads), hence the absolute default.
EXP78 = Path(
    "/home/bizon/git/MarinFold-exp78/experiments/exp78_evals_esmfold_contacts"
)
EXP65 = Path(
    "/home/bizon/git/MarinFold-exp78/experiments/exp65_evals_low_msa_depth_proteins"
)

# pyconfind geometry knobs — MUST match
# marinfold.document_structures.contacts_and_crops_v1.GenerationConfig
# defaults, so ground-truth contacts are defined exactly as the training
# documents define them. Same values exp78/exp89 pinned.
PYCONFIND_KWARGS = dict(
    native_only=True,
    contact_distance=3.0,
    dcut=25.0,
    clash_distance=2.0,
    assembly=None,
)


def iter_manifest(manifest: Path, gt_root: Path, default_dataset: str):
    """Yield ``(dataset, stem, input_seq, gt_cif, gt_chain, strata)`` per row."""
    df = pd.read_csv(manifest)
    # Everything that is neither an input nor already a top-level index field
    # becomes a stratum (neff_tier, fold_verdict, seq_leakage, msa_neff, …).
    strata_cols = [
        c
        for c in df.columns
        if c not in {"stem", "gt_cif", "gt_chain", "input_seq", "n_residues"}
    ]
    for _, rec in df.iterrows():
        strata = {c: (None if pd.isna(rec[c]) else rec[c]) for c in strata_cols}
        dataset = strata.pop("dataset", None) or default_dataset
        yield (
            str(dataset),
            rec["stem"],
            rec["input_seq"],
            gt_root / rec["gt_cif"],
            rec.get("gt_chain"),
            strata,
        )


def build_one(stem: str, input_seq: str, gt_cif: Path, gt_chain: str | None):
    """Analyze one GT structure into a canonical AtomArray + its index row.

    Returns ``(atom_array, index_row, contacts)`` where ``contacts`` is the
    list of ``(i, j, degree)`` pyconfind contacts in input-sequence
    coordinates with ``i < j``.
    """
    structure, chain = extract_single_chain(gt_cif, prefer_chain=gt_chain)
    analyzed = analyze_coordinates(structure, entry_id=stem, **PYCONFIND_KWARGS)

    obs = "".join(one_letter(r.resname) for r in analyzed.residues)
    mapping = align_obs_to_ref(obs, input_seq)
    matched = sum(
        1 for k, c in enumerate(mapping) if c is not None and obs[k] == input_seq[c]
    )

    atoms: list[tuple[int, str, str, float, float, float, float]] = []
    for k, residue in enumerate(analyzed.residues):
        ref_index = mapping[k]
        if ref_index is None:
            continue
        for name, x, y, z in analyzed.atoms_by_seq_index.get(residue.seq_index, ()):
            atoms.append((ref_index + 1, residue.resname, name, x, y, z, 0.0))
    if not atoms:
        raise ValueError(f"{stem}: no in-vocabulary heavy atoms mapped to the input sequence")

    array = build_atom_array(atoms)
    contacts: list[tuple[int, int, float]] = []
    for c in analyzed.contacts:
        ci = mapping[c.seq_i] if c.seq_i < len(mapping) else None
        cj = mapping[c.seq_j] if c.seq_j < len(mapping) else None
        if ci is None or cj is None or ci == cj:
            continue
        lo, hi = (ci, cj) if ci < cj else (cj, ci)
        contacts.append((lo, hi, float(c.degree)))
    contacts.sort()

    mapped_positions = sorted({c for c in mapping if c is not None})
    row = dict(
        stem=stem,
        gt_chain=chain,
        L=len(input_seq),
        n_resolved=len(analyzed.residues),
        n_mapped=len(mapped_positions),
        alignment_identity=round(matched / len(obs), 4) if obs else 0.0,
        n_gt_atoms=int(len(array)),
        n_gt_ca=int((array.atom_name == "CA").sum()),
        n_gt_residues=int(len(set(array.res_id.tolist()))),
        global_plddt=float(analyzed.global_plddt),
        n_contacts=len(contacts),
    )
    return array, row, contacts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--foldbench-manifest",
        type=Path,
        default=EXP78 / "data/eval_manifest_foldbench.csv",
    )
    ap.add_argument(
        "--exp65-manifest", type=Path, default=EXP78 / "data/eval_manifest_exp65.csv"
    )
    ap.add_argument(
        "--foldbench-gt-root", type=Path, default=EXP78 / "_scratch/gt_foldbench"
    )
    ap.add_argument("--exp65-gt-root", type=Path, default=EXP65)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--limit", type=int, default=None, help="stop after N proteins (smoke runs)"
    )
    args = ap.parse_args(argv)

    structures_dir = args.out_dir / "gt_structures"
    structures_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        (args.foldbench_manifest, args.foldbench_gt_root, "foldbench100"),
        (args.exp65_manifest, args.exp65_gt_root, "exp65"),
    ]

    n_ok = n_fail = 0
    started = time.time()
    with (
        (args.out_dir / "gt_index.jsonl").open("w") as index_fh,
        (args.out_dir / "gt_contacts.jsonl").open("w") as contacts_fh,
    ):
        for manifest, gt_root, default_dataset in sources:
            for dataset, stem, input_seq, gt_cif, gt_chain, strata in iter_manifest(
                manifest, gt_root, default_dataset
            ):
                if args.limit is not None and n_ok >= args.limit:
                    break
                if not gt_cif.exists():
                    print(f"  {stem}: MISSING GT {gt_cif}", file=sys.stderr)
                    n_fail += 1
                    continue
                chain = None if (gt_chain is None or pd.isna(gt_chain)) else gt_chain
                try:
                    array, row, contacts = build_one(stem, input_seq, gt_cif, chain)
                except (ValueError, RuntimeError) as exc:
                    print(f"  {stem}: FAILED: {exc!r}", file=sys.stderr)
                    n_fail += 1
                    continue
                out_path = structures_dir / dataset / f"{stem}.pdb"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                write_structure(array, out_path)
                row["dataset"] = dataset
                row["record_id"] = f"{dataset}/{stem}"
                row["strata"] = strata
                index_fh.write(json.dumps(row) + "\n")
                contacts_fh.write(
                    json.dumps(
                        {
                            "record_id": row["record_id"],
                            "stem": stem,
                            "contacts": [[i, j, d] for (i, j, d) in contacts],
                        }
                    )
                    + "\n"
                )
                n_ok += 1
                if n_ok % 50 == 0:
                    print(f"  ...{n_ok} proteins done ({time.time() - started:.0f}s)", flush=True)

    print(
        f"[gt] wrote {n_ok} structures to {structures_dir} "
        f"({n_fail} failed/missing) in {time.time() - started:.0f}s"
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
