"""Fetch a curation sample and measure every within-cluster structural pair.

Coordinates are pinned AFDB v4 source objects. Network/parse failures propagate;
raw objects are cached atomically and hashed. This does not predict structures,
produce training documents, or certify new sequences as evaluation-decontaminated.
C-alpha contact overlap is a geometry diagnostic, distinct from pyconfind labels.
"""

import argparse
import csv
import hashlib
import itertools
import json
import multiprocessing
import socket
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path
from time import perf_counter

import gcsfs
import gemmi
import numpy as np
from tmtools import tm_align

STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def noncanonical_sequence(sequence: str) -> bool:
    """Flag source residues whose mapping to the training document needs an audit."""
    return bool(set(sequence) - STANDARD_AMINO_ACIDS)


@dataclass(frozen=True)
class Protein:
    """C-alpha geometry in the original residue order."""

    sequence: str
    coords: np.ndarray
    plddt: np.ndarray


def parse_structure(content: bytes, expected_length: int) -> Protein:
    """Parse one complete single-chain prediction, validating coordinates."""
    structure = gemmi.read_structure_string(content)
    if len(structure) != 1 or len(structure[0]) != 1:
        raise ValueError("Expected exactly one model and one chain")
    sequence, coordinates, confidence = [], [], []
    for residue in structure[0][0]:
        atom = residue.find_atom("CA", "*")
        if atom is None:
            continue
        sequence.append(
            gemmi.find_tabulated_residue(residue.name).one_letter_code.upper()
        )
        coordinates.append([atom.pos.x, atom.pos.y, atom.pos.z])
        confidence.append(atom.b_iso)
    coords = np.asarray(coordinates, dtype=np.float64)
    plddt = np.asarray(confidence, dtype=np.float64)
    if len(sequence) != expected_length or coords.shape != (expected_length, 3):
        raise ValueError(f"Expected {expected_length} residues, parsed {len(sequence)}")
    if not np.isfinite(coords).all() or not np.isfinite(plddt).all():
        raise ValueError("Non-finite coordinates or confidence")
    if np.any((plddt < 0) | (plddt > 100)):
        raise ValueError("AFDB confidence must use the 0-100 scale")
    return Protein("".join(sequence), coords, plddt)


@cache
def filesystem() -> gcsfs.GCSFileSystem:
    """Reuse the authenticated connection pool within a fetch worker."""
    return gcsfs.GCSFileSystem()


def read_csv(path: Path) -> list[dict]:
    """Read a small committed curation manifest."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a small table with a consistent schema, including no hidden index."""
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fetch(row: dict, cache_dir: Path, keep_raw: bool = True) -> dict:
    """Fetch one original object with a full GET and validate before caching."""
    started = perf_counter()
    entry_id = row["entry_id"]
    path = cache_dir / f"{entry_id}.cif"
    cached = path.exists()
    # cat_file reads the entire response, avoiding AFDB gzip-transcoding's
    # compressed Content-Length / decompressed seekable-read truncation trap.
    content = path.read_bytes() if cached else filesystem().cat_file(row["gcs_uri"])
    protein = parse_structure(content, int(row["seq_len"]))
    if not cached and keep_raw:
        temporary = path.with_suffix(".cif.part")
        temporary.write_bytes(content)
        temporary.replace(path)
    np.savez_compressed(
        cache_dir / f"{entry_id}.npz",
        sequence=protein.sequence,
        coords=protein.coords,
        plddt=protein.plddt,
    )
    return {
        "entry_id": entry_id,
        "struct_cluster_id": row["struct_cluster_id"],
        "source_uri": row["gcs_uri"],
        "n_residues": len(protein.sequence),
        "metadata_plddt": float(row["global_plddt"]),
        "mean_plddt": float(protein.plddt.mean()),
        "confident_fraction": float(np.mean(protein.plddt >= 70)),
        "noncanonical_residues": sum(
            aa not in STANDARD_AMINO_ACIDS for aa in protein.sequence
        ),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "cached": cached,
        "fetch_parse_seconds": perf_counter() - started,
        "hostname": socket.gethostname(),
    }


def load_protein(cache_dir: Path, entry_id: str) -> Protein:
    """Load validated cached arrays without Python object deserialization."""
    with np.load(cache_dir / f"{entry_id}.npz", allow_pickle=False) as data:
        return Protein(str(data["sequence"]), data["coords"], data["plddt"])


def aligned_indices(sequence_a: str, sequence_b: str) -> tuple[np.ndarray, np.ndarray]:
    """Map gapped alignment columns to original zero-based residue indices."""
    if len(sequence_a) != len(sequence_b):
        raise ValueError("Alignment strings have unequal lengths")
    ia, ib, indices_a, indices_b = 0, 0, [], []
    for aa, bb in zip(sequence_a, sequence_b, strict=True):
        if aa != "-" and bb != "-":
            indices_a.append(ia)
            indices_b.append(ib)
        ia += aa != "-"
        ib += bb != "-"
    return np.array(indices_a, dtype=int), np.array(indices_b, dtype=int)


def contacts(coords: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Return C-alpha contacts <8 angstrom, excluding short sequence separations."""
    distance_sq = np.sum((coords[:, None] - coords[None, :]) ** 2, axis=-1)
    return (distance_sq < 64) & (np.abs(positions[:, None] - positions[None, :]) >= 6)


def compare(a: Protein, b: Protein) -> dict:
    """Compute both TM normalizations and comparable-core contact differences."""
    start = perf_counter()
    result = tm_align(a.coords, b.coords, a.sequence, b.sequence)
    ia, ib = aligned_indices(result.seqxA, result.seqyA)
    ca = contacts(a.coords[ia], ia)
    cb = contacts(b.coords[ib], ib)
    # Compare only pairs that meet the sequence-separation criterion in BOTH
    # original chains; insertions must not manufacture contact disagreement.
    eligible = np.triu(
        (np.abs(ia[:, None] - ia[None, :]) >= 6)
        & (np.abs(ib[:, None] - ib[None, :]) >= 6),
        k=1,
    )
    union = int(np.sum((ca | cb) & eligible))
    intersection = int(np.sum(ca & cb & eligible))
    ma, mb = a.plddt >= 70, b.plddt >= 70
    core_tm = None
    if min(int(ma.sum()), int(mb.sum())) >= 20:
        core = tm_align(
            a.coords[ma],
            b.coords[mb],
            "".join(np.array(list(a.sequence))[ma]),
            "".join(np.array(list(b.sequence))[mb]),
        )
        core_tm = max(core.tm_norm_chain1, core.tm_norm_chain2)
    return {
        "tm_a": result.tm_norm_chain1,
        "tm_b": result.tm_norm_chain2,
        "tm_max": max(result.tm_norm_chain1, result.tm_norm_chain2),
        "rmsd": result.rmsd,
        "aligned_residues": len(ia),
        "coverage_a": len(ia) / len(a.sequence),
        "coverage_b": len(ib) / len(b.sequence),
        "aligned_sequence_identity": sum(
            a.sequence[i] == b.sequence[j] for i, j in zip(ia, ib, strict=True)
        )
        / len(ia)
        if len(ia)
        else 0.0,
        "ca_contact_jaccard": intersection / union if union else None,
        "ca_contact_union": union,
        "core_tm_max": core_tm,
        "core_residues_a": int(ma.sum()),
        "core_residues_b": int(mb.sum()),
        "elapsed_seconds": perf_counter() - start,
    }


def audit_cluster(job: tuple[list[dict], dict], cache_dir: Path) -> list[dict]:
    """Measure all sample pairs within one original cluster."""
    rows, previous = job
    proteins = {r["entry_id"]: load_protein(cache_dir, r["entry_id"]) for r in rows}
    return [
        {
            "struct_cluster_id": a["struct_cluster_id"],
            "entry_a": a["entry_id"],
            "entry_b": b["entry_id"],
            "anchor_a": a["is_anchor"],
            "anchor_b": b["is_anchor"],
            **(
                previous[(a["entry_id"], b["entry_id"])]
                if (a["entry_id"], b["entry_id"]) in previous
                else compare(proteins[a["entry_id"]], proteins[b["entry_id"]])
            ),
        }
        for a, b in itertools.combinations(rows, 2)
    ]


def select_candidates(
    rows: list[dict], pairs: list[dict], threshold: float = 0.8
) -> list[dict]:
    """Choose up to three candidates complementary to every retained anchor."""
    pair_map = {frozenset((p["entry_a"], p["entry_b"])): p for p in pairs}
    groups = {}
    for row in rows:
        groups.setdefault(row["struct_cluster_id"], []).append(row)
    results = []
    for cluster, members in groups.items():
        anchors = [r for r in members if r["is_anchor"].lower() == "true"]
        if not anchors:
            raise ValueError(f"{cluster}: no retained training anchors")
        selected = [r["entry_id"] for r in anchors]
        candidates = [r for r in members if r["is_anchor"].lower() != "true"]
        assessments = []
        for row in candidates:
            comparisons = [
                pair_map[frozenset((row["entry_id"], a["entry_id"]))] for a in anchors
            ]
            nearest = max(comparisons, key=lambda p: p["tm_max"])
            nearest_id = (
                nearest["entry_b"]
                if nearest["entry_a"] == row["entry_id"]
                else nearest["entry_a"]
            )
            minimum_coverage = min(
                min(p["coverage_a"], p["coverage_b"]) for p in comparisons
            )
            length_ratio = min(
                min(int(row["seq_len"]), int(a["seq_len"]))
                / max(int(row["seq_len"]), int(a["seq_len"]))
                for a in anchors
            )
            assessments.append(
                {
                    **row,
                    "nearest_anchor": nearest_id,
                    "nearest_anchor_tm": nearest["tm_max"],
                    "min_coverage": minimum_coverage,
                    "min_length_ratio": length_ratio,
                    "nearest_anchor_core_tm": nearest["core_tm_max"],
                    "max_anchor_core_tm": max(p["core_tm_max"] for p in comparisons)
                    if all(p["core_tm_max"] is not None for p in comparisons)
                    else None,
                    "nearest_anchor_ca_jaccard": nearest["ca_contact_jaccard"],
                    "selected_order": 0,
                    "selection_reason": "pending",
                }
            )
        pending = list(assessments)
        lengths = {r["entry_id"]: int(r["seq_len"]) for r in members}
        for rank in range(1, 4):
            admissible = [
                r
                for r in pending
                if r["min_coverage"] >= 0.8
                and r["min_length_ratio"] >= 0.8
                and all(
                    min(
                        pair_map[frozenset((r["entry_id"], s))]["coverage_a"],
                        pair_map[frozenset((r["entry_id"], s))]["coverage_b"],
                    )
                    >= 0.8
                    and min(lengths[r["entry_id"]], lengths[s])
                    / max(lengths[r["entry_id"]], lengths[s])
                    >= 0.8
                    for s in selected
                )
                and max(
                    pair_map[frozenset((r["entry_id"], s))]["tm_max"] for s in selected
                )
                <= threshold
            ]
            if not admissible:
                break
            chosen = min(
                admissible,
                key=lambda r: (
                    max(
                        pair_map[frozenset((r["entry_id"], s))]["tm_max"]
                        for s in selected
                    ),
                    r["entry_id"],
                ),
            )
            chosen["selected_order"] = rank
            chosen["selection_reason"] = (
                "provisional_diverse; visual/core audit required"
            )
            selected.append(chosen["entry_id"])
            pending.remove(chosen)
        for row in pending:
            if row["min_coverage"] < 0.8 or row["min_length_ratio"] < 0.8:
                row["selection_reason"] = "coverage_or_length"
            elif row["nearest_anchor_tm"] > threshold:
                row["selection_reason"] = "covered_by_training_anchor"
            else:
                row["selection_reason"] = "redundant_with_addition_or_cluster_cap"
        results.extend(assessments)
    return results


def main() -> None:
    """Fetch and score the manifest, retaining deterministic pair-level outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--skip-raw-cache",
        action="store_true",
        help="Keep validated C-alpha arrays without duplicating source mmCIFs locally",
    )
    parser.add_argument(
        "--reuse-audit",
        type=Path,
        help="Reuse measured pairs only when both source SHA256 hashes match",
    )
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.sample)
    source_sample_bytes = args.sample.read_bytes()
    started = perf_counter()
    filesystem()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fetched = list(
            pool.map(
                partial(fetch, cache_dir=args.cache, keep_raw=not args.skip_raw_cache),
                rows,
            )
        )
    write_csv(args.output / "fetch_timings.csv", fetched)
    noncanonical = {r["entry_id"] for r in fetched if r["noncanonical_residues"]}
    excluded_clusters = {
        r["struct_cluster_id"]
        for r in rows
        if r["entry_id"] in noncanonical and r["is_anchor"].lower() == "true"
    }
    rejected = [
        {
            "entry_id": r["entry_id"],
            "struct_cluster_id": r["struct_cluster_id"],
            "reason": "noncanonical_anchor_cluster"
            if r["struct_cluster_id"] in excluded_clusters
            else "noncanonical_sequence",
        }
        for r in rows
        if r["entry_id"] in noncanonical or r["struct_cluster_id"] in excluded_clusters
    ]
    if rejected:
        write_csv(args.output / "quality_rejections.csv", rejected)
        rows = [
            r
            for r in rows
            if r["entry_id"] not in noncanonical
            and r["struct_cluster_id"] not in excluded_clusters
        ]
    (args.output / "source_sample.csv").write_bytes(source_sample_bytes)
    write_csv(args.output / "sample.csv", rows)
    print(
        f"Fetched and validated {len(fetched)} structures in {perf_counter() - started:.1f}s",
        flush=True,
    )
    groups = {}
    for row in rows:
        groups.setdefault(row["struct_cluster_id"], []).append(row)
    reused = {}
    if args.reuse_audit:
        prior = json.loads((args.reuse_audit / "audit.json").read_text())
        if prior["tmtools_version"] != "0.3.0":
            raise ValueError(
                "Cannot reuse pairs measured with a different alignment version"
            )
        old_hashes = {
            r["entry_id"]: r["sha256"]
            for r in read_csv(args.reuse_audit / "fetch_timings.csv")
        }
        hashes = {r["entry_id"]: r["sha256"] for r in fetched}
        kept_ids = {r["entry_id"] for r in rows}
        valid = {
            entry
            for entry, sha in hashes.items()
            if old_hashes.get(entry) == sha and entry in kept_ids
        }
        identity_columns = {
            "struct_cluster_id",
            "entry_a",
            "entry_b",
            "anchor_a",
            "anchor_b",
        }
        for pair in read_csv(args.reuse_audit / "pairs.csv"):
            if pair["entry_a"] in valid and pair["entry_b"] in valid:
                reused.setdefault(pair["struct_cluster_id"], {})[
                    (pair["entry_a"], pair["entry_b"])
                ] = {
                    k: float(v) if v else None
                    for k, v in pair.items()
                    if k not in identity_columns
                }
        print(
            f"Reusing {sum(len(g) for g in reused.values())} source-hash-matched pairs",
            flush=True,
        )
    pairs = []
    jobs = [(members, reused.get(cluster, {})) for cluster, members in groups.items()]
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for i, batch in enumerate(
            pool.map(partial(audit_cluster, cache_dir=args.cache), jobs), 1
        ):
            pairs.extend(batch)
            print(
                f"Aligned {i}/{len(groups)} clusters ({len(pairs)} pairs)", flush=True
            )
    write_csv(args.output / "pairs.csv", pairs)
    selected = select_candidates(rows, pairs)
    write_csv(args.output / "candidates.csv", selected)
    summary = {
        "source": "afdb",
        "structures": len(rows),
        "fetched_structures": len(fetched),
        "clusters_excluded_noncanonical_anchor": len(excluded_clusters),
        "input_sample_sha256": hashlib.sha256(source_sample_bytes).hexdigest(),
        "audited_sample_sha256": hashlib.sha256(
            (args.output / "sample.csv").read_bytes()
        ).hexdigest(),
        "pairs": len(pairs),
        "candidates": len(selected),
        "provisional_diverse": sum(r["selected_order"] > 0 for r in selected),
        "clusters_with_diverse": len(
            {r["struct_cluster_id"] for r in selected if r["selected_order"] > 0}
        ),
        "elapsed_seconds": perf_counter() - started,
        "hostname": socket.gethostname(),
        "workers": args.workers,
        "tmtools_version": "0.3.0",
        "bytes_read_from_network": sum(r["bytes"] for r in fetched if not r["cached"]),
        "command": " ".join(sys.argv),
        "reuse_audit": str(args.reuse_audit) if args.reuse_audit else None,
        "reused_pairs": sum(len(g) for g in reused.values()),
        "status": "visual curation; no candidate is cleared for training",
    }
    (args.output / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
