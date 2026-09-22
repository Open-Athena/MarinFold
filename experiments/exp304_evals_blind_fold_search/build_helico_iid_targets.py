#!/usr/bin/env python
"""Prepare every primary-test iid rollout for individual Helico folding."""

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from build_helico_targets import prepare_structure, verify_contacts
from evaluate import score_candidate
from search_policy import canonical

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SOURCE = HERE.parent / "exp301_evals_fold_switching_proteins" / "data"
OLD = HERE / "_cache" / "iid500"
TAIL = HERE / "_cache" / "iid1000_tail_primary"
OUT = HERE / "_cache" / "helico_iid" / "data"


def index_raw(root: Path, expected: set[str]) -> dict[str, Path]:
    """Map one-protein rollout parquet files to pair IDs."""
    result = {}
    for path in sorted(root.glob("shard-*.parquet")):
        pair_id = str(pd.read_parquet(path, columns=["pair_id"]).pair_id.iloc[0])
        if pair_id in expected:
            result[pair_id] = path
    if set(result) != expected:
        raise ValueError(f"{root}: missing {sorted(expected - set(result))}")
    return result


def mapped_contacts(pairs: set[tuple[int, int]], positions: dict[int, int]) -> list[list[int]]:
    """Map union-sequence contacts into Helico's input-chain token indices."""
    return [list(pair) for pair in sorted({
        (min(positions[i], positions[j]), max(positions[i], positions[j]))
        for i, j in pairs if i in positions and j in positions
    })]


def main() -> None:
    """Write compact inputs plus one sequence-bearing CIF per protein."""
    cohort = pd.read_csv(DATA / "cohort.csv").set_index("pair_id")
    expected = set(cohort[(cohort.split == "test") & cohort.primary].index)
    frozen = set((DATA / "iid1000_primary_test_ids.txt").read_text().splitlines())
    if expected != frozen or len(expected) != 29:
        raise ValueError("primary-test IDs differ from the frozen iid1000 cohort")
    old, tail = index_raw(OLD, expected), index_raw(TAIL, expected)
    truth = pd.read_parquet(SOURCE / "eval_targets.parquet").set_index("pair_id")
    with (SOURCE / "foldswitch_universe.jsonl").open() as handle:
        mismatches = {row["pair_id"]: int(row["n_seq_mismatch"])
                      for row in (json.loads(line) for line in handle)}
    (OUT / "gt").mkdir(parents=True, exist_ok=True)
    rows, proteins = [], []
    for pair_id in sorted(expected):
        target = truth.loc[pair_id]
        sequence = str(target.sequence)
        cif, positions, coordinates, structure_meta = prepare_structure(str(target.fold1), sequence)
        true1, true2 = canonical(target.contacts_fold1), canonical(target.contacts_fold2)
        control_check = verify_contacts(str(target.fold1), true1, coordinates)
        (OUT / "gt" / f"{pair_id}.cif.gz").write_bytes(cif)
        proteins.append({
            "pair_id": pair_id, "L": int(target.L), "input_fold": str(target.fold1),
            "reference_fold1": str(target.fold1), "reference_fold2": str(target.fold2),
            "strict_exact": mismatches[pair_id] == 0,
            **structure_meta, **control_check,
        })
        first = pd.read_parquet(old[pair_id]).sort_values("rollout")
        second = pd.read_parquet(tail[pair_id]).sort_values("rollout")
        frame = pd.concat([first, second], ignore_index=True)
        if frame.rollout.tolist() != list(range(1000)):
            raise ValueError(f"{pair_id}: expected rollouts 0..999")
        target_dict = target.to_dict()
        target_dict["pair_id"] = pair_id
        for record in frame.to_dict("records"):
            score = score_candidate(record, target_dict, True, "test", "iid", 0)
            contacts = canonical(record["contacts"]) | canonical(record["given"])
            rows.append({
                "target_id": f"{pair_id}__r{int(record['rollout']):04d}",
                "pair_id": pair_id, "kind": "iid", "rollout": int(record["rollout"]),
                "candidate_id": str(record["candidate_id"]), "L": int(record["L"]),
                "marinfold_finished": bool(record["finished"]),
                "marinfold_n_tokens": int(record["n_tokens"]),
                "n_contacts_raw": len(contacts),
                "contacts": mapped_contacts(contacts, positions),
                "contact_recall_fold1_fs": float(score["recall_a_fs"]),
                "contact_recall_fold2_fs": float(score["recall_b_fs"]),
                "contact_phi_fs": float(score["phi_fs"]),
                "contact_precision_union": float(score["precision_union"]),
                "n_pred_fs": int(score["n_pred_fs"]),
            })
        for kind, rollout, pairs in (
            ("true_fold1", -1, true1), ("true_fold2", -2, true2), ("no_contacts", -3, set()),
        ):
            rows.append({
                "target_id": f"{pair_id}__{kind}", "pair_id": pair_id,
                "kind": kind, "rollout": rollout, "candidate_id": kind,
                "L": int(target.L), "marinfold_finished": True,
                "marinfold_n_tokens": 0, "n_contacts_raw": len(pairs),
                "contacts": mapped_contacts(pairs, positions),
                "contact_recall_fold1_fs": None, "contact_recall_fold2_fs": None,
                "contact_phi_fs": None, "contact_precision_union": None,
                "n_pred_fs": None,
            })
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, OUT / "inputs.parquet", compression="zstd")
    pd.DataFrame(rows)[["target_id", "pair_id", "kind", "rollout", "L"]].to_csv(
        OUT / "targets.csv", index=False
    )
    protein_frame = pd.DataFrame(proteins).sort_values("pair_id")
    protein_frame.to_csv(DATA / "helico_iid_proteins.csv", index=False)
    protocol = {
        "n_proteins": len(expected), "n_iid_rollouts_per_protein": 1000,
        "n_targets": len(rows), "input_structure": "Fold1 observed chain",
        "controls_per_protein": ["true_fold1", "true_fold2", "no_contacts"],
        "helico_checkpoint": "/ckpts/contacts-msafree-01/final.pt",
        "helico_checkpoint_step": 6000, "helico_source_revision": "b10385d736673c81b10e70d1099962af6f2573c0",
        "n_diffusion_samples": 1, "n_trunk_recycles": 6, "seed": 42,
        "msa": False, "single_sequence": True,
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"wrote {len(rows):,} targets ({len(table):,} rows, "
          f"{(OUT / 'inputs.parquet').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
