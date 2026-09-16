#!/usr/bin/env python
# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 0 — build the exp301 fold-switching eval universe and run the premise gate.

Source of the pairs: ``supporting tables/TableS1.xlsx`` from
https://github.com/ncbi/AF2_benchmark (Chakravarty et al. 2024, *Nat Commun*
15:7296; Zenodo 10.5281/zenodo.13221957). It lists **93 fold-switching pairs**,
each as ``Fold1`` / ``Fold2`` PDB+chain plus the sequence of the fold-switching
region.

For each pair this script:

1. Locates both mmCIFs (local RCSB mirror first, RCSB fetch as fallback).
2. Runs pyconfind on each named chain with exactly ``contacts_v1``'s geometry —
   marinfold's ``analyze_structure`` under exp74's ``PYCONFIND_KWARGS``, both
   imported rather than forked, so the contact definition here is the same one
   every other MarinFold contact number uses.
3. Merges the two chains' *observed* sequences into one **union reference**, so
   both folds' contacts land in a single coordinate frame and the model gets
   one input sequence per pair.
4. Restricts to residues **resolved in both** chains and derives the sets that
   every downstream metric is computed on:
       ``S`` = shared contacts, ``A`` = Fold1-unique, ``B`` = Fold2-unique.
   ``A`` and ``B`` together are the **discriminative universe**. Nothing in
   exp301 is ever computed on the global map, which the shared core dominates.
5. Runs the **premise gate**: a pair whose two contact maps barely differ cannot
   test anything, so ``|A|``, ``|B|``, Jaccard and the fold-switching region's
   share of the discriminative pairs are reported *before* any GPU is spent.

Writes ``data/foldswitch_universe.jsonl`` (one record per pair, the input to
Phase 1) and ``data/premise_gate.csv`` (the gate table, committed and read by
the README). Both are small enough to commit.

    uv run python prepare_inputs.py
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import sys
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import gemmi
import requests

HERE = Path(__file__).resolve().parent
# exp74 owns the ground-truth contact definition; import it rather than fork it.
sys.path.insert(0, str(HERE.parent / "exp74_evals_protenix_pyconfind_contacts"))
from pyconfind_contacts import (  # noqa: E402
    PYCONFIND_KWARGS,
    extract_single_chain,
)

from marinfold.document_structures.contacts_v1 import (  # noqa: E402
    GenerationConfig,
    analyze_structure,
)

DATA = HERE / "data"
CACHE = HERE / "_cache"
MIRROR = Path("/data/tim/af3-db/mmcif_files")

TABLE_S1_URL = (
    "https://raw.githubusercontent.com/ncbi/AF2_benchmark/main/"
    "supporting%20tables/TableS1.xlsx"
)

# The contacts_v1 thresholds, read off the generator config rather than
# restated, so a spec change propagates here instead of silently diverging.
_CFG = GenerationConfig()
MIN_DEGREE = _CFG.min_contact_degree
MIN_SEP = _CFG.min_seq_separation

# contacts-v1 addresses residues with <p0>..<p1999>; a longer chain has no
# document at all, so it cannot be scored.
MAX_RESIDUES = 2000

# Below this the "pair" is one fold observed twice and there is nothing to
# discriminate. Reported, not silently dropped.
MIN_DISCRIMINATIVE = 10

# The two chains must actually overlap. A pair whose chains cover different
# domains leaves a common universe far smaller than the reference, and its
# "unique" contacts are mostly residues the other structure never resolved.
MIN_COVERAGE = 0.5

# Above this pairwise identity the pair is an identical-sequence fold switch —
# the regime TableS1 describes (mean 99% / median 100%). Below it the pair is a
# homolog or point-mutant pair, which is a different question: the model is
# folding one sequence and being asked about another's contacts. Kept, flagged,
# and reported separately — never silently pooled.
IDENTICAL_SEQUENCE_THRESHOLD = 0.98

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


# --------------------------------------------------------------------------
# TableS1
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PairSpec:
    """One row of TableS1: two PDB+chain ids and the fold-switching region."""

    pair_id: str
    pdb1: str
    chain1: str
    pdb2: str
    chain2: str
    fs_region_seq: str


def _xlsx_rows(path: Path, sheet: int = 1) -> list[list[str]]:
    """Minimal xlsx reader — one sheet, shared strings resolved, as text."""
    z = zipfile.ZipFile(path)
    shared: list[str] = []
    try:
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall("m:si", _NS):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
    except KeyError:
        pass
    sh = ET.fromstring(z.read(f"xl/worksheets/sheet{sheet}.xml"))
    rows: list[list[str]] = []
    for row in sh.findall(".//m:row", _NS):
        vals: list[str] = []
        for c in row.findall("m:c", _NS):
            v = c.find("m:v", _NS)
            if v is None or v.text is None:
                vals.append("")
            elif c.get("t") == "s":
                vals.append(shared[int(v.text)])
            else:
                vals.append(v.text)
        rows.append(vals)
    return rows


def load_pairs() -> list[PairSpec]:
    """Download (once) and parse TableS1 into PairSpecs."""
    CACHE.mkdir(exist_ok=True)
    xlsx = CACHE / "TableS1.xlsx"
    if not xlsx.exists():
        resp = requests.get(TABLE_S1_URL, timeout=120)
        resp.raise_for_status()
        xlsx.write_bytes(resp.content)
    rows = _xlsx_rows(xlsx)
    header = [h.strip().lower() for h in rows[0][:3]]
    if header[:2] != ["fold1", "fold2"]:
        raise ValueError(f"unexpected TableS1 header: {rows[0][:3]!r}")

    pairs: list[PairSpec] = []
    for row in rows[1:]:
        if len(row) < 3 or not row[0].strip() or not row[1].strip():
            continue  # trailing blank / formatting rows
        f1, f2 = row[0].strip(), row[1].strip()
        # "6c6sD" -> ("6c6s", "D"). Chain ids in this table are single
        # characters appended to the 4-character entry id.
        if len(f1) != 5 or len(f2) != 5:
            raise ValueError(f"unexpected fold id format: {f1!r} / {f2!r}")
        pairs.append(
            PairSpec(
                pair_id=f"{f1.lower()}_{f2.lower()}",
                pdb1=f1[:4].lower(),
                chain1=f1[4],
                pdb2=f2[:4].lower(),
                chain2=f2[4],
                fs_region_seq=row[2].strip().upper(),
            )
        )
    return pairs


# --------------------------------------------------------------------------
# Structures
# --------------------------------------------------------------------------
def structure_path(pdb: str) -> Path:
    """Prefer the local mirror; fall back to RCSB and cache under ``_cache/cif``."""
    local = MIRROR / f"{pdb.lower()}.cif"
    if local.exists():
        return local
    cif_cache = CACHE / "cif"
    cif_cache.mkdir(parents=True, exist_ok=True)
    cached = cif_cache / f"{pdb.lower()}.cif"
    if not cached.exists():
        resp = requests.get(f"https://files.rcsb.org/download/{pdb.upper()}.cif", timeout=120)
        resp.raise_for_status()
        cached.write_bytes(resp.content)
    return cached


def _one_letter(resname: str) -> str:
    """Canonical 3-letter -> 1-letter via gemmi's residue table.

    ``analyze_structure`` has already canonicalized modified residues (MSE->MET,
    SEP->SER, ...), so this only ever sees standard names. Correctness is
    self-checked in :func:`union_reference`, which asserts that every mapped
    fold1 residue equals the reference residue it points at — a wrong mapping
    here would break that assertion rather than pass silently.
    """
    info = gemmi.find_tabulated_residue(resname)
    if info is None or not info.is_amino_acid():
        return "X"
    return info.one_letter_code.upper()


def analyze_chain(path: Path, pdb: str, chain: str):
    """pyconfind analysis of one named chain, plus its observed sequence.

    Geometry comes from exp74's ``PYCONFIND_KWARGS`` and marinfold's
    ``analyze_structure``, so the contact definition is the same one every
    other MarinFold contact number uses.
    """
    st, chosen = extract_single_chain(path, prefer_chain=chain)
    if chosen != chain:
        # extract_single_chain silently falls back to the longest polymer when
        # the requested chain is absent. Scoring the wrong chain would look
        # like a real result, so this is fatal for the pair, not a fallback.
        raise ValueError(f"chain {chain!r} not found in {pdb} (got {chosen!r})")
    analyzed = analyze_structure(st, entry_id=f"{pdb}_{chain}", **PYCONFIND_KWARGS)
    return analyzed, "".join(_one_letter(r.resname) for r in analyzed.residues)


def entry_metadata(path: Path) -> tuple[str, float | None]:
    """``(experimental method, resolution)`` for tiering. Resolution is None for NMR."""
    doc = gemmi.cif.read(str(path))
    block = doc.sole_block()
    method = (block.find_value("_exptl.method") or "").strip().strip("'\"") or "UNKNOWN"
    res: float | None = None
    for tag in ("_refine.ls_d_res_high", "_reflns.d_resolution_high", "_em_3d_reconstruction.resolution"):
        raw = block.find_value(tag)
        if raw and raw not in ("?", "."):
            try:
                res = float(raw)
            except ValueError:
                res = None
            if res is not None:
                break
    return method, res


# --------------------------------------------------------------------------
# The union reference frame
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class UnionFrame:
    """Two observed chains merged into one reference, with both mappings kept.

    Attributes:
        reference: the merged sequence — what the model is asked to fold.
        map1/map2: for each observed residue of fold1/fold2, its index in
            ``reference``, or ``None`` if it maps nowhere.
        diffs: ``replace`` blocks, i.e. positions where the two chains carry
            different amino acids. Fold1's residues are the ones kept.
    """

    reference: str
    map1: list[int | None]
    map2: list[int | None]
    diffs: list[dict]


def union_reference(obs1: str, obs2: str) -> UnionFrame:
    """Merge two near-identical observed sequences into one linear reference.

    The two chains of a fold-switching pair are ~99-100% identical but usually
    cover different residue ranges and have different unresolved loops. Taking
    either one as the frame would throw away the other's residues, so we walk
    the alignment and keep everything: shared runs once, each chain's private
    runs in place. The result is a real contiguous sequence containing both
    folds' observed residues.

    **Both mappings are derived from this single alignment and carried out**,
    rather than re-deriving each chain's mapping against the finished reference.
    difflib is greedy, not an optimal global aligner, so a second independent
    pass does not have to agree with this one — and when it did not, the
    disagreement looked exactly like a broken pair. It rejected four, KaiB
    (5jyt/2qke) among them.
    """
    sm = difflib.SequenceMatcher(a=obs1, b=obs2, autojunk=False)
    parts: list[str] = []
    diffs: list[dict] = []
    map1: list[int | None] = [None] * len(obs1)
    map2: list[int | None] = [None] * len(obs2)
    base = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            chunk = obs1[i1:i2]
            for t in range(i2 - i1):
                map1[i1 + t] = base + t
                map2[j1 + t] = base + t
        elif tag == "delete":  # fold1 only
            chunk = obs1[i1:i2]
            for t in range(i2 - i1):
                map1[i1 + t] = base + t
        elif tag == "insert":  # fold2 only
            chunk = obs2[j1:j2]
            for t in range(j2 - j1):
                map2[j1 + t] = base + t
        else:  # replace — a genuine sequence difference; fold1's residues win
            chunk = obs1[i1:i2]
            diffs.append({"ref_start": base, "fold1": obs1[i1:i2], "fold2": obs2[j1:j2]})
            for t in range(i2 - i1):
                map1[i1 + t] = base + t
            # Map fold2's positional overlap so its contacts still land; any
            # surplus fold2 residues map nowhere. Every position in this block
            # is excluded from the scored universe regardless.
            for t in range(min(i2 - i1, j2 - j1)):
                map2[j1 + t] = base + t
        parts.append(chunk)
        base += len(chunk)

    reference = "".join(parts)
    # Exact self-check: fold1 defines the reference, so every mapped fold1
    # residue must be identical to the reference residue it points at. A
    # failure here is an indexing bug, not biology.
    for k, pos in enumerate(map1):
        if pos is not None and reference[pos] != obs1[k]:
            raise ValueError(f"union reference mapping is inconsistent at fold1 residue {k}")
    return UnionFrame(reference=reference, map1=map1, map2=map2, diffs=diffs)


def remap_contacts(
    analyzed, mapping: list[int | None]
) -> tuple[set[tuple[int, int]], set[int]]:
    """Move one chain's pyconfind contacts into reference coordinates.

    Returns ``(contacts, resolved)`` where contacts are ``(i, j)`` pairs with
    ``i < j`` already filtered to contacts_v1's degree and separation cuts, and
    ``resolved`` is the set of reference positions this chain resolves.
    """
    resolved = {pos for pos in mapping if pos is not None}
    contacts: set[tuple[int, int]] = set()
    for contact in analyzed.contacts:
        if contact.degree < MIN_DEGREE:
            continue
        if contact.seq_i >= len(mapping) or contact.seq_j >= len(mapping):
            continue
        ci, cj = mapping[contact.seq_i], mapping[contact.seq_j]
        if ci is None or cj is None:
            continue
        lo, hi = (ci, cj) if ci < cj else (cj, ci)
        if hi - lo >= MIN_SEP:
            contacts.add((lo, hi))
    return contacts, resolved


#: A located region must recover at least this fraction of the table's
#: fold-switching-region sequence, and may not sprawl over more than this
#: multiple of its length. The first rejects a chance placement, the second
#: rejects one stitched together from short matches scattered down the chain.
FS_MIN_MATCHED = 0.6
FS_MAX_SPAN_RATIO = 1.5


def locate_fs_region(reference: str, fs_seq: str) -> tuple[int, int] | None:
    """Find the fold-switching region in the reference.

    TableS1 gives the region's sequence from the full construct, while the
    reference holds only residues the structures actually resolve — so the
    region routinely appears split across several blocks with unresolved loops
    between them (``1kcta/3t1pa``: 81 of 84 residues in four blocks). A plain
    substring search, or one requiring a single long block, misses those; it
    lost 8 otherwise-usable pairs.

    So: anchor on the longest matching block, keep the other blocks that agree
    with that placement, and accept the span they cover only if it recovers
    enough of the region without sprawling across the chain.
    """
    if not fs_seq:
        return None
    idx = reference.find(fs_seq)
    if idx >= 0:
        return idx, idx + len(fs_seq)

    sm = difflib.SequenceMatcher(a=fs_seq, b=reference, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    if not blocks:
        return None
    anchor = max(blocks, key=lambda b: b.size)
    # Unresolved residues shift later blocks earlier in the reference, so the
    # offset drifts; bound the drift by the region's own length rather than a
    # constant, and let the span check below catch anything that slips past.
    tolerance = max(len(fs_seq), 20)
    consistent = [b for b in blocks if abs((b.b - b.a) - (anchor.b - anchor.a)) <= tolerance]

    matched = sum(b.size for b in consistent)
    start = min(b.b for b in consistent)
    end = max(b.b + b.size for b in consistent)
    if matched < FS_MIN_MATCHED * len(fs_seq):
        return None
    if (end - start) > FS_MAX_SPAN_RATIO * len(fs_seq):
        return None
    return start, end


# --------------------------------------------------------------------------
# Per-pair build
# --------------------------------------------------------------------------
def build_pair(spec: PairSpec) -> dict:
    """Ground truth + premise numbers for one fold-switching pair."""
    path1, path2 = structure_path(spec.pdb1), structure_path(spec.pdb2)

    analyzed1, obs1 = analyze_chain(path1, spec.pdb1, spec.chain1)
    analyzed2, obs2 = analyze_chain(path2, spec.pdb2, spec.chain2)
    frame = union_reference(obs1, obs2)
    reference = frame.reference
    if len(reference) > MAX_RESIDUES:
        raise ValueError(f"reference length {len(reference)} exceeds the {MAX_RESIDUES}-residue token space")

    diffs = frame.diffs
    n_mismatch = sum(len(d["fold1"]) for d in diffs)
    pair_identity = 1.0 - n_mismatch / len(reference) if reference else 0.0

    raw1, resolved1 = remap_contacts(analyzed1, frame.map1)
    raw2, resolved2 = remap_contacts(analyzed2, frame.map2)

    # Positions where the two chains carry different amino acids are dropped
    # from the universe outright: a contact there is not the same contact in
    # the two folds, so it can neither be shared nor discriminative.
    mismatched = {
        pos
        for d in diffs
        for pos in range(d["ref_start"], d["ref_start"] + len(d["fold1"]))
    }
    common = (resolved1 & resolved2) - mismatched
    c1 = {(i, j) for i, j in raw1 if i in common and j in common}
    c2 = {(i, j) for i, j in raw2 if i in common and j in common}
    shared, only1, only2 = c1 & c2, c1 - c2, c2 - c1
    union = c1 | c2
    jaccard = len(shared) / len(union) if union else 1.0

    # The fold-switching region is where the biology is. A pair's two crystals
    # also differ for reasons that have nothing to do with fold switching —
    # ligands, domain motions, different constructs — and for a long protein
    # those dominate the raw discriminative set (1h38d/1qlna: L=882 with only
    # 2% of its discriminative pairs touching the switching region). So the
    # FS-restricted sets are carried alongside the global ones throughout, and
    # the analysis leads with whichever the gate shows is well-posed.
    fs = locate_fs_region(reference, spec.fs_region_seq)
    if fs is None:
        fs_share = None
        fs_only1: set[tuple[int, int]] = set()
        fs_only2: set[tuple[int, int]] = set()
    else:
        lo, hi = fs
        in_region = lambda i, j: lo <= i < hi or lo <= j < hi  # noqa: E731
        fs_only1 = {(i, j) for i, j in only1 if in_region(i, j)}
        fs_only2 = {(i, j) for i, j in only2 if in_region(i, j)}
        discriminative = only1 | only2
        n_in_fs = len(fs_only1) + len(fs_only2)
        fs_share = n_in_fs / len(discriminative) if discriminative else None

    method1, resolution1 = entry_metadata(path1)
    method2, resolution2 = entry_metadata(path2)
    same_entry = spec.pdb1 == spec.pdb2
    structural = {"X-RAY DIFFRACTION"}
    if same_entry:
        tier = "A"
    elif method1.upper() in structural and method2.upper() in structural:
        tier = "B"
    else:
        tier = "C"

    return {
        "pair_id": spec.pair_id,
        "fold1": f"{spec.pdb1}{spec.chain1}",
        "fold2": f"{spec.pdb2}{spec.chain2}",
        "tier": tier,
        "seq_class": "identical" if pair_identity >= IDENTICAL_SEQUENCE_THRESHOLD else "homolog",
        "same_entry": same_entry,
        "method1": method1,
        "method2": method2,
        "resolution1": resolution1,
        "resolution2": resolution2,
        "sequence": reference,
        "L": len(reference),
        "n_common": len(common),
        "coverage": len(common) / len(reference) if reference else 0.0,
        "common_positions": sorted(common),
        "seq_differences": diffs,
        "n_seq_mismatch": n_mismatch,
        "pair_identity": pair_identity,
        "fs_region_seq": spec.fs_region_seq,
        "fs_region": list(fs) if fs else None,
        "fs_share_of_discriminative": fs_share,
        "n_contacts_fold1": len(c1),
        "n_contacts_fold2": len(c2),
        "n_shared": len(shared),
        "n_only1": len(only1),
        "n_only2": len(only2),
        "n_only1_fs": len(fs_only1),
        "n_only2_fs": len(fs_only2),
        "jaccard": jaccard,
        "contacts_fold1": sorted(c1),
        "contacts_fold2": sorted(c2),
        "n_resolved_fold1": len(resolved1),
        "n_resolved_fold2": len(resolved2),
    }


GATE_COLUMNS = [
    "pair_id", "fold1", "fold2", "tier", "seq_class", "same_entry", "method1",
    "method2", "resolution1", "resolution2", "L", "n_common", "coverage",
    "pair_identity", "n_seq_mismatch", "n_contacts_fold1", "n_contacts_fold2",
    "n_shared", "n_only1", "n_only2", "n_only1_fs", "n_only2_fs", "jaccard",
    "fs_region_start", "fs_region_end", "fs_share_of_discriminative",
    "passes_gate", "status", "reason",
]


def gate_row(record: dict) -> dict:
    """Flatten one built pair into a premise-gate row.

    The gate asks one question — *can this pair discriminate between two folds
    at all?* — and answers it with three independent criteria, each of which
    kills the comparison on its own. Failures are recorded with their reason,
    never dropped, so the README can report what the 93 actually reduced to.
    """
    reasons: list[str] = []

    discriminative = min(record["n_only1"], record["n_only2"])
    if discriminative < MIN_DISCRIMINATIVE:
        reasons.append(f"min(|A|,|B|)={discriminative}<{MIN_DISCRIMINATIVE}")

    if record["coverage"] < MIN_COVERAGE:
        reasons.append(f"coverage={record['coverage']:.2f}<{MIN_COVERAGE}")

    fs = record["fs_region"]
    if fs is None:
        reasons.append("fold-switching region not located")

    return {
        **{k: record.get(k) for k in GATE_COLUMNS if k in record},
        "fs_region_start": fs[0] if fs else None,
        "fs_region_end": fs[1] if fs else None,
        "passes_gate": not reasons,
        "status": "ok",
        "reason": "; ".join(reasons),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="build only the first N pairs (smoke test)")
    parser.add_argument(
        "--max-failures", type=int, default=10,
        help="abort if more than this many pairs fail to build",
    )
    args = parser.parse_args()

    DATA.mkdir(exist_ok=True)
    pairs = load_pairs()
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"TableS1: {len(pairs)} fold-switching pairs")

    records: list[dict] = []
    rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    for n, spec in enumerate(pairs, 1):
        try:
            record = build_pair(spec)
        except Exception as exc:  # noqa: BLE001
            # A sweep over 93 heterogeneous PDB entries hits absent chains,
            # oversized assemblies and unreadable metadata. The failure reason
            # per pair IS the result here, so it is recorded and printed rather
            # than aborting the sweep — but a run with many failures aborts
            # below, because that means the pipeline is wrong, not the data.
            failures.append((spec.pair_id, f"{type(exc).__name__}: {exc}"))
            rows.append({
                **{c: None for c in GATE_COLUMNS},
                "pair_id": spec.pair_id,
                "fold1": f"{spec.pdb1}{spec.chain1}",
                "fold2": f"{spec.pdb2}{spec.chain2}",
                "passes_gate": False,
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            print(f"[{n:3d}/{len(pairs)}] {spec.pair_id:16s} FAILED  {type(exc).__name__}: {exc}")
            continue
        records.append(record)
        row = gate_row(record)
        rows.append(row)
        share = row["fs_share_of_discriminative"]
        share_str = " n/a" if share is None else f"{share:4.2f}"
        flag = "" if row["passes_gate"] else f"  [GATE: {row['reason']}]"
        print(
            f"[{n:3d}/{len(pairs)}] {record['pair_id']:16s} {record['tier']}/{record['seq_class'][:4]} "
            f"L={record['L']:4d} cov={record['coverage']:.2f} id={record['pair_identity']:.2f} "
            f"|A|={record['n_only1']:4d} |B|={record['n_only2']:4d} "
            f"J={record['jaccard']:.3f} fs={share_str} "
            f"|A_fs|={record['n_only1_fs']:3d} |B_fs|={record['n_only2_fs']:3d}{flag}"
        )

    if len(failures) > args.max_failures:
        print(f"\n{len(failures)} pairs failed to build (limit {args.max_failures}):", file=sys.stderr)
        for pair_id, reason in failures:
            print(f"  {pair_id}: {reason}", file=sys.stderr)
        raise SystemExit("too many build failures — fix the pipeline before trusting the gate")

    universe = DATA / "foldswitch_universe.jsonl"
    with universe.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")

    gate = DATA / "premise_gate.csv"
    with gate.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=GATE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    passing = [r for r in records if gate_row(r)["passes_gate"]]
    by_tier: dict[str, int] = {}
    by_class: dict[str, int] = {}
    for r in passing:
        key = f"{r['tier']}/{r['seq_class']}"
        by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
        by_class[key] = by_class.get(key, 0) + 1
    print(f"\nbuilt {len(records)}/{len(pairs)} pairs, {len(failures)} failed")
    print(f"passing the premise gate: {len(passing)}")
    print(f"  by tier:  {dict(sorted(by_tier.items()))}")
    print(f"  by class: {dict(sorted(by_class.items()))}")
    if passing:
        jac = sorted(r["jaccard"] for r in passing)
        disc = sorted(min(r["n_only1"], r["n_only2"]) for r in passing)
        print(f"  jaccard(fold1,fold2): median {jac[len(jac) // 2]:.3f}  range {jac[0]:.3f}-{jac[-1]:.3f}")
        print(f"  min(|A|,|B|):         median {disc[len(disc) // 2]}  range {disc[0]}-{disc[-1]}")
    print(f"wrote {universe.relative_to(HERE)} and {gate.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
