"""Freeze eval-val votes and CASP coordinates without running a predictor.

Fetch only the current default checkpoint's 97 archived vote matrices. The
published bundle can subsequently be scored anonymously without S3 access.
"""

import argparse
import gzip
import hashlib
import io
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import s3fs
from huggingface_hub import HfFileSystem

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
DATA = HERE / "data"
CACHE = HERE / ".cache"
MODEL = "contacts-v1-exp277-m2-p06-full-epoch-1.5B"
LABEL = "exp277_full_epoch_m2_p06_step266344"
SOURCE = "marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot/evals/rollout-v2/2026-09-13/v2-01"
GT_SOURCE = "buckets/open-athena/MarinFold/data/contacts-v1-foldbench-monomers-exp245/gt_universe_scored.jsonl"
PUBLIC = "hf://buckets/open-athena/MarinFold/data/exp352-casp-precision-context/v1"
EXP277 = EXPERIMENTS / "exp277_models_single_mpnn_pilot/data/eval_rollout_v2"


def digest(raw: bytes) -> str:
    """Return the SHA256 of a source artifact."""
    return hashlib.sha256(raw).hexdigest()


def coordinates(path: Path, chain_name: str, sequence: str) -> tuple[list, dict]:
    """Extract first-model CB (CA for glycine), using verified label_seq indexing.

    Residues lacking the required atom are unavailable, never assigned a CA
    surrogate for non-glycine. Alternate atoms are selected by occupancy then
    altloc, independent of contacts or prediction confidence.
    """
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    chains = [c for c in structure[0] if c.name == chain_name]
    if len(chains) != 1:
        raise ValueError(
            f"{path}: expected exactly one chain {chain_name}, found {len(chains)}"
        )
    entity = structure.get_entity_of(chains[0].get_polymer())
    block = gemmi.cif.read(str(path)).sole_block()
    polymer = block.get_mmcif_category("_entity_poly.")
    entity_index = polymer["entity_id"].index(entity.name)
    canonical = "".join(polymer["pdbx_seq_one_letter_code_can"][entity_index].split())
    if canonical != sequence:
        raise ValueError(
            f"{path}: deposited canonical sequence differs from evaluation input"
        )
    positions: dict[int, list[float]] = {}
    mapped, missing, mismatches = [], [], []
    for residue in chains[0].get_polymer():
        if residue.label_seq is None:
            raise ValueError(f"{path}: missing label_seq at {residue.seqid}")
        index = residue.label_seq - 1
        if not 0 <= index < len(sequence):
            raise ValueError(f"{path}: label_seq {index + 1} outside sequence")
        # The deposited entity sequence identifies modified residues too (e.g.
        # 85L is cysteine), including compounds absent from gemmi's small table.
        if residue.name != entity.full_sequence[index]:
            mismatches.append([index, residue.name, entity.full_sequence[index]])
        aa = canonical[index]
        mapped.append(index)
        atom_name = "CA" if aa == "G" else "CB"
        atoms = [atom for atom in residue if atom.name == atom_name]
        if not atoms:
            missing.append([index, residue.name, atom_name])
            continue
        atom = min(atoms, key=lambda a: (-a.occ, a.altloc))
        if index in positions:
            raise ValueError(f"{path}: duplicate residue index {index}")
        positions[index] = [atom.pos.x, atom.pos.y, atom.pos.z]
    if mismatches:
        raise ValueError(f"{path}: sequence-index mismatch: {mismatches}")
    return [[i, *positions[i]] for i in sorted(positions)], {
        "n_polymer_residues": len(mapped),
        "missing_atoms": missing,
        "mapped_indices": sorted(mapped),
        "coordinate_residues": len(positions),
    }


def collect(cif_cache: Path) -> dict:
    """Fetch and validate only the permitted eval-val prediction cohort."""
    DATA.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    sets_path = (
        EXPERIMENTS / "exp245_evals_foldbench_held_out_monomers/data/eval_sets.csv"
    )
    selected = (
        pd.read_csv(sets_path).query("eval_set == 'eval-val'").sort_values("stem")
    )
    assert len(selected) == selected.stem.nunique() == 97
    assert selected.scorable.eq(1).all() and selected.designed.eq(0).all()
    gt_path = CACHE / "gt_universe_scored.jsonl"
    if not gt_path.exists():
        gt_path.write_bytes(HfFileSystem(token=False).read_bytes(GT_SOURCE))
    raw_gt = gt_path.read_bytes()
    allowed = set(selected.stem)
    truth = {
        r["stem"]: r
        for line in raw_gt.splitlines()
        if (r := json.loads(line))["stem"] in allowed
    }
    assert set(truth) == allowed
    timings = pd.read_csv(EXP277 / "timings.csv")
    timings = timings[
        timings.dataset.eq("foldbench_monomer") & timings.stem.isin(allowed)
    ]
    assert len(timings) == timings.stem.nunique() == 97
    assert timings.n_rollouts.eq(100).all() and timings.unfinished_rollouts.eq(0).all()
    timings.to_csv(DATA / "source_timings.csv", index=False)
    fs = s3fs.S3FileSystem(
        profile="cw",
        endpoint_url="https://cwobject.com",
        config_kwargs={"s3": {"addressing_style": "virtual"}},
    )

    def get_one(row: pd.Series) -> dict:
        stem = row.stem
        rec = truth[stem]
        assert rec["L"] == len(row.sequence)
        uri = f"{SOURCE}/dense_scores/{LABEL}/foldbench_monomer__{stem}.npz"
        score_path = CACHE / f"{stem}.npz"
        if not score_path.exists():
            score_path.write_bytes(fs.cat_file(uri))
        raw = score_path.read_bytes()
        with np.load(io.BytesIO(raw)) as archive:
            score = archive["score"]
        assert score.shape == (rec["L"], rec["L"])
        assert np.isfinite(score).all() and (score >= 0).all()
        assert np.array_equal(score, score.T)
        assert np.array_equal(score, np.round(score)) and score.max() <= 100
        cif = cif_cache / f"{row.pdb_id}-assembly1.cif.gz"
        url = f"https://files.rcsb.org/download/{row.pdb_id}-assembly1.cif.gz"
        if not cif.exists():
            cif = CACHE / f"{row.pdb_id}-assembly1.cif.gz"
            if not cif.exists():
                with urllib.request.urlopen(url, timeout=120) as handle:
                    cif.write_bytes(handle.read())
        xyz, coordinate_audit = coordinates(cif, rec["gt_chain"], row.sequence)
        # Frozen contacts-v1 resolves backbone-bearing residues. CASP additionally
        # needs CB/CA; report their overlap instead of silently changing indexing.
        coordinate_audit["polymer_not_frozen"] = sorted(
            set(coordinate_audit["mapped_indices"]) - set(rec["resolved"])
        )
        coordinate_audit["frozen_not_polymer"] = sorted(
            set(rec["resolved"]) - set(coordinate_audit["mapped_indices"])
        )
        # The old universe uses a sequence-only alignment, which can place a
        # repeated residue on the other side of an unresolved gap. CASP truth
        # uses the deposited, sequence-verified label_seq mapping. Preserve all
        # differences for a sensitivity analysis; do not relabel old contacts.
        i, j = np.where(np.triu(score, 1) > 0)
        return {
            "truth": rec,
            "votes": np.column_stack([i, j, score[i, j]]).astype(int).tolist(),
            "xyz": xyz,
            "coordinate_audit": coordinate_audit,
            "score_source": "s3://" + uri,
            "score_sha256": digest(raw),
            "cif_source": url,
            "cif_sha256": digest(cif.read_bytes()),
            "sequence": row.sequence,
            "is_viral": bool(row.is_viral),
            "exp199_best_identity": None
            if pd.isna(row.exp199_best_identity)
            else float(row.exp199_best_identity),
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        proteins = list(pool.map(get_one, [r for _, r in selected.iterrows()]))
    manifest = json.loads((EXP277 / "run_manifest.json").read_text())
    return {
        "schema": 1,
        "model": MODEL,
        "step": 266344,
        "gt_source": "hf://" + GT_SOURCE,
        "gt_sha256": digest(raw_gt),
        "eval_sets_sha256": digest(sets_path.read_bytes()),
        "sampling": manifest["sampling"],
        "n_rollouts": 100,
        "source_manifest_sha256": digest((EXP277 / "run_manifest.json").read_bytes()),
        "proteins": proteins,
    }


def main() -> None:
    """Write the frozen analysis input bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cif-cache",
        type=Path,
        default=Path.home()
        / ".cache/helico/data/benchmarks/FoldBench/examples/ground_truths",
    )
    args = parser.parse_args()
    result = collect(args.cif_cache)
    path = CACHE / "inputs.json.gz"
    path.write_bytes(
        gzip.compress(json.dumps(result, allow_nan=False).encode(), mtime=0)
    )
    print(
        f"Prepared {len(result['proteins'])} proteins; {path.stat().st_size:,} bytes; {digest(path.read_bytes())}"
    )


if __name__ == "__main__":
    main()
