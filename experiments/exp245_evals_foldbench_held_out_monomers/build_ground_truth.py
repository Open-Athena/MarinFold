# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 2 -- pyconfind ground-truth contacts for all 334 FoldBench monomers.

199 of the 334 have never been scored and have no ground truth. Rather than
bolting new records onto #89's frozen universe and leaving the eval sets with
two provenances, this rebuilds **all 334** through one path and then proves the
path by re-deriving the 135 that #89 and #226 already froze:

* Contacts come from #89's ``pyconfind_contacts.compute_contacts``, imported
  through :mod:`upstream`, in ``native_only`` mode with the C++ confind geometry
  ``contacts_v1`` generates with. Nothing is re-implemented.
* Structures are RCSB ``-assembly1`` mmCIFs, the source exp12 used for the
  original 100 and #226 used for its 23.
* The chain passed as ``prefer_chain`` is the resolved **auth** asym id, not
  FoldBench's ``chain_id`` (which is the mmCIF *label* id for 10 entries).
* Records carry ``resolved`` -- the candidate-pair universe -- so every unit is
  scored over the same index space as the 554.

**The control.** Any frozen unit that was built from *the same input sequence*
must come back identical: same length, same resolved positions, same contact
pairs. #226 proved this for the FoldBench-100 (100/100); here it is a gate over
every overlapping unit.

Some frozen units are the same PDB stem built from a **different** sequence --
exp65's ``denovo_pdb`` records use its own construct sequences, which are
offset or truncated relative to FoldBench's entity sequence, so the same
structure re-indexes to different residue numbers. Those are classified by
comparing the recorded input sequences, not by a hardcoded exception list, and
reported as sequence variants with the evidence.

All 334 units carry the dataset label ``foldbench_monomer``. The eval sets are
one homogeneous universe, so nothing downstream has to know which of them used
to live under ``foldbench100`` or ``denovo_pdb``.

    uv run --extra gt python build_ground_truth.py
    uv run --extra gt python build_ground_truth.py --only-control   # just the gate
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

import upstream as U

DATA = U.DATA
DATASET = "foldbench_monomer"
GT_UNIVERSE = DATA / "gt_universe_foldbench_monomers.jsonl"
GT_MANIFEST = DATA / "gt_manifest.csv"
GT_REPORT = DATA / "gt_report.json"
CIF_CACHE = U.WORK / "cif"

RCSB_ASSEMBLY_URL = "https://files.rcsb.org/download/{pdb}-assembly1.cif"

#: Below this, the structure does not match the eval sequence and the chain
#: selection is wrong. #226's threshold, and every one of the 554 clears it.
MIN_ALIGNMENT_IDENTITY = 0.90
#: A protein with no contacts contributes nothing and would dilute any mean.
MIN_CONTACTS = 1



def fetch_assembly(pdb_id: str, cache_dir: Path, *, retries: int = 4) -> Path:
    """Download ``<pdb>-assembly1.cif``, cached on disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{pdb_id}-assembly1.cif"
    if out.exists() and out.stat().st_size > 0:
        return out
    url = RCSB_ASSEMBLY_URL.format(pdb=pdb_id)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=180) as handle:
                out.write_bytes(handle.read())
            return out
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"failed to fetch {url}") from exc
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def build_record(row, cif: Path, compute_contacts) -> dict:
    """One ``gt_universe.jsonl`` record in #89's schema."""
    auth = [c for c in str(row.auth_asym_ids).split(";") if c]
    gt = compute_contacts(
        cif, row.sequence, stem=row.stem, prefer_chain=auth[0] if auth else None)
    if gt.alignment_identity < MIN_ALIGNMENT_IDENTITY:
        raise ValueError(
            f"{row.stem}: alignment identity {gt.alignment_identity:.3f} below "
            f"{MIN_ALIGNMENT_IDENTITY} on chain {gt.chain} -- the structure does "
            "not match the eval sequence, so the chain selection is wrong."
        )
    if len(gt.contacts) < MIN_CONTACTS:
        raise ValueError(f"{row.stem}: pyconfind found no contacts")
    return dict(
        dataset=DATASET,
        stem=row.stem,
        L=int(gt.n_input_residues),
        n_resolved=int(gt.n_resolved_residues),
        gt_chain=gt.chain,
        gt_align_identity=round(float(gt.alignment_identity), 4),
        resolved=[int(p) for p in gt.resolved_positions],
        contacts=[[int(i), int(j), float(d)] for (i, j, d) in gt.contacts],
        strata={},
    )


def load_frozen_sequences() -> dict[tuple[str, str], str]:
    """``{(dataset, stem): input_seq}`` for the units whose sequence is recorded.

    Only the legacy 554 have a published targets file. The 23 units #226 added
    are absent from it and are treated as same-sequence, which is correct by
    construction: #226 built them from the very ``foldbench_targets.csv``
    sequences this experiment uses.
    """
    import pyarrow.parquet as pq

    path = U.require_pinned(U.LEGACY_TARGETS, U.LEGACY_TARGETS_SIZE)
    if U.sha256(path) != U.LEGACY_TARGETS_SHA256:
        raise SystemExit(f"{path} does not match the digest PR #244 pins")
    table = pq.read_table(path, columns=["dataset", "stem", "input_seq"]).to_pylist()
    return {(row["dataset"], row["stem"]): row["input_seq"] for row in table}


def load_frozen() -> dict[str, list[dict]]:
    """The frozen 577-unit universe, keyed by stem (a stem can have two units)."""
    frozen: dict[str, list[dict]] = {}
    if not U.FROZEN_GT_UNIVERSE.exists():
        raise SystemExit(
            f"frozen universe not found at {U.FROZEN_GT_UNIVERSE}; it is #226's "
            "published gt_universe_eval2.jsonl and the control needs it."
        )
    for line in U.FROZEN_GT_UNIVERSE.read_text().splitlines():
        record = json.loads(line)
        frozen.setdefault(record["stem"], []).append(record)
    return frozen


def compare(record: dict, frozen: list[dict], sequence: str,
            frozen_sequences: dict[tuple[str, str], str]) -> list[dict]:
    """Compare a rebuilt record against every frozen unit with the same stem."""
    results = []
    for other in frozen:
        key = (other["dataset"], other["stem"])
        frozen_sequence = frozen_sequences.get(key)
        same_sequence = frozen_sequence is None or frozen_sequence == sequence
        same_length = other["L"] == record["L"]
        same_resolved = list(other["resolved"]) == list(record["resolved"])
        mine = {(i, j) for i, j, _ in record["contacts"]}
        theirs = {(i, j) for i, j, _ in other["contacts"]}
        results.append({
            "stem": record["stem"],
            "frozen_dataset": other["dataset"],
            "sequence_recorded": frozen_sequence is not None,
            "same_input_sequence": same_sequence,
            "length_match": same_length,
            "resolved_match": same_resolved,
            "contacts_match": mine == theirs,
            "n_contacts_rebuilt": len(mine),
            "n_contacts_frozen": len(theirs),
            "jaccard": round(len(mine & theirs) / max(1, len(mine | theirs)), 6),
            "identical": same_length and same_resolved and mine == theirs,
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", type=Path, default=DATA / "eval_sets.csv")
    parser.add_argument("--cif-cache", type=Path, default=CIF_CACHE,
                        help="assembly mmCIFs are inputs, not artifacts; kept out of git")
    parser.add_argument("--only-control", action="store_true",
                        help="rebuild only the units the frozen universe covers")
    args = parser.parse_args()

    compute_contacts, geometry = U.exp89_contacts()
    sets = pd.read_csv(args.sets)
    frozen = load_frozen()
    frozen_sequences = load_frozen_sequences()
    if args.only_control:
        sets = sets[sets["stem"].isin(frozen)]
    print(f"[gt] {len(sets)} monomers; pyconfind geometry: {geometry}", flush=True)

    records, manifest, failures, controls = [], [], [], []
    for index, row in enumerate(sets.itertuples(), 1):
        try:
            cif = fetch_assembly(row.pdb_id, args.cif_cache)
            record = build_record(row, cif, compute_contacts)
        except Exception as exc:  # noqa: BLE001 - one bad structure must not
            # abort 333 good ones; every failure is named in the report and the
            # protein is dropped from its eval set rather than silently missing.
            failures.append({"stem": row.stem, "eval_set": row.eval_set,
                             "error": f"{type(exc).__name__}: {exc}"})
            print(f"  [{index:3d}/{len(sets)}] {row.stem}: FAILED {exc}", flush=True)
            continue

        records.append(record)
        comparisons = (
            compare(record, frozen[row.stem], row.sequence, frozen_sequences)
            if row.stem in frozen else []
        )
        controls.extend(comparisons)
        gating = [c for c in comparisons if c["same_input_sequence"]]
        control = max(gating, key=lambda c: c["jaccard"]) if gating else None
        manifest.append({
            "eval_set": row.eval_set, "dataset": DATASET, "stem": row.stem,
            "pdb_id": row.pdb_id, "gt_cif": cif.name, "gt_chain": record["gt_chain"],
            "foldbench_chain": row.chain_id, "input_seq": row.sequence,
            "n_residues": record["L"], "n_resolved": record["n_resolved"],
            "gt_align_identity": record["gt_align_identity"],
            "n_contacts": len(record["contacts"]),
            "in_frozen_universe": int(row.stem in frozen),
            "control_identical": None if control is None else int(control["identical"]),
        })
        flag = "" if control is None else (
            " control=identical" if control["identical"]
            else f" control=DIFFERS(jaccard={control['jaccard']:.4f})")
        if control is None and comparisons:
            flag = " control=sequence-variant-only"
        print(f"  [{index:3d}/{len(sets)}] {row.stem}: L={record['L']} "
              f"resolved={record['n_resolved']} chain={record['gt_chain']} "
              f"id={record['gt_align_identity']:.3f} "
              f"contacts={len(record['contacts'])}{flag}", flush=True)

    if not args.only_control:
        with GT_UNIVERSE.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        pd.DataFrame(manifest).to_csv(GT_MANIFEST, index=False)

    gated = [c for c in controls if c["same_input_sequence"]]
    variants = [c for c in controls if not c["same_input_sequence"]]
    unexpected = [c for c in gated if not c["identical"]]
    report = {
        "dataset_label": DATASET,
        "n_requested": int(len(sets)),
        "n_built": len(records),
        "failures": failures,
        "control": {
            "n_frozen_units_compared": len(controls),
            "n_same_sequence": len(gated),
            "n_same_sequence_identical": sum(1 for c in gated if c["identical"]),
            "n_sequence_variants": len(variants),
            "sequence_variants": variants,
            "unexpected_differences": unexpected,
            "passed": not unexpected,
        },
        "pyconfind_geometry": {k: str(v) for k, v in geometry.items()},
        "frozen_universe": str(U.FROZEN_GT_UNIVERSE),
    }
    if not args.only_control:
        report["universe_sha256"] = U.sha256(GT_UNIVERSE)
    GT_REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2)[:2000])
    if failures:
        print(f"[gt] {len(failures)} FAILED: "
              f"{[f['stem'] for f in failures]}", flush=True)
    print(f"[gt] {len(records)} records -> {GT_UNIVERSE}", flush=True)
    return 0 if report["control"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
