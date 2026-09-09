# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze eval-val contact interventions before new inference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from build_metrics import load_detail, true_matrix
from common import EXPECTED_UNITS, load_ground_truth, load_targets

ARMS = (
    "iid",
    "iid_repeat",
    "true_small",
    "false_small",
    "pred_small",
    "true_large",
    "false_large",
    "pred_large",
)


def digest(path: Path) -> str:
    """Return the SHA-256 of an input file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def range_bin(separation: int) -> int:
    """Return the conventional short, medium, or long separation bin."""
    return 0 if separation <= 11 else 1 if separation <= 23 else 2


def contexts_for_target(record: dict, votes: np.ndarray, replicate: int) -> dict:
    """Draw nested true/false controls and select predicted pairs without GT."""
    length = int(record["L"])
    truth_matrix = true_matrix(length, record["contacts"])
    resolved = set(record["resolved"])
    truth = [
        (int(i), int(j))
        for i, j in zip(*np.where(truth_matrix))
        if i in resolved and j in resolved
    ]
    count = length // 3
    if count < 10 or count >= len(truth):
        raise ValueError(f"Invalid conditioning dose for {record['stem']}")
    generator = np.random.default_rng(
        int.from_bytes(
            hashlib.sha256(
                f"exp254-context-v1:{record['stem']}:{replicate}".encode()
            ).digest()[:8],
            "little",
        )
    )
    chosen_true = [truth[i] for i in generator.permutation(len(truth))[:count]]
    negatives = {b: [] for b in range(3)}
    for i in sorted(resolved):
        for j in sorted(resolved):
            if j - i >= 6 and not truth_matrix[i, j]:
                negatives[range_bin(j - i)].append((i, j))
    for pairs in negatives.values():
        generator.shuffle(pairs)
    chosen_false = [negatives[range_bin(j - i)].pop() for i, j in chosen_true]
    # Predicted contexts use the full sequence candidate universe, without
    # reference labels or the experimentally resolved residue mask.
    ii, jj = np.triu_indices(length, k=6)
    order = np.argsort(-votes[ii, jj], kind="stable")[:count]
    predicted = list(zip(ii[order].tolist(), jj[order].tolist()))
    contexts = {"iid": [], "iid_repeat": []}
    for kind, pairs in (
        ("true", chosen_true),
        ("false", chosen_false),
        ("pred", predicted),
    ):
        contexts[f"{kind}_small"] = pairs[:10]
        contexts[f"{kind}_large"] = pairs
    union = {pair for pairs in contexts.values() for pair in pairs}
    if not set(truth) - union:
        raise ValueError(f"No common remaining true contacts for {record['stem']}")
    return contexts


def main() -> None:
    """Write the fixed protocol, contexts, first-pass votes, and input hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError(f"Refusing to overwrite a frozen plan: {args.out}")
    gt = load_ground_truth()
    targets = load_targets()
    if len(targets) != EXPECTED_UNITS:
        raise ValueError("Expected exactly 97 eval-val targets")
    detail = load_detail(args.source_run / "iid")
    grouped = dict(tuple(detail.groupby(["dataset", "stem"])))
    if set(grouped) != {(t.dataset, t.stem) for t in targets}:
        raise ValueError("Source rollout target universe differs from eval-val")
    plan_targets, matrices = [], {}
    for target in targets:
        frame = grouped[(target.dataset, target.stem)]
        if (
            not np.array_equal(np.sort(frame.rollout.unique()), np.arange(100))
            or frame.is_seed.any()
        ):
            raise ValueError(
                f"{target.stem}: source must have exactly 100 unseeded rollouts"
            )
        if frame.duplicated(["rollout", "i", "j"]).any():
            raise ValueError(f"{target.stem}: duplicate source contact")
        matrix = np.zeros((target.L, target.L), dtype=np.int16)
        np.add.at(matrix, (frame.i.to_numpy(), frame.j.to_numpy()), 1)
        matrix += matrix.T.copy()
        matrices[target.stem] = matrix
        record = gt[(target.dataset, target.stem)]
        tmat = true_matrix(target.L, record["contacts"])
        plan_targets.append(
            {
                "dataset": target.dataset,
                "stem": target.stem,
                "L": target.L,
                "input_seq": target.input_seq,
                "resolved": record["resolved"],
                "truth": [
                    [int(i), int(j)]
                    for i, j in zip(*np.where(tmat))
                    if i in set(record["resolved"]) and j in set(record["resolved"])
                ],
                "contexts": [
                    contexts_for_target(record, matrix, rep) for rep in range(2)
                ],
            }
        )
    args.out.mkdir(parents=True)
    np.savez_compressed(args.out / "source_votes.npz", **matrices)
    plan = {
        "protocol": "exp254-matched-contact-intervention-v1",
        "eval_set": "eval-val",
        "model_run": "prot-exp232-cw-cv1-decontam-s02-m2-p06-aug",
        "step": 145199,
        "n_rollouts": 100,
        "source_n_rollouts": 100,
        "n_repeats": 2,
        "arms": ARMS,
        "doses": {"small": 10, "large": "floor(L/3)"},
        "practical_margin": 0.03,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": -1,
        "completion_budget": "6L+128 after the complete prompt",
        "max_model_len": 8192,
        "prediction_source": "archived_iid_100_consensus",
        "source_votes_file": "source_votes.npz",
        "source_votes_sha256": digest(args.out / "source_votes.npz"),
        "primary": "pred_large full-universe R-precision vs source first pass + fresh iid (200 total)",
        "diagnostic": "common union-excluded remaining-pair scoring; true/false are oracle controls",
        "source_files": {
            str(p.relative_to(args.source_run)): digest(p)
            for p in sorted((args.source_run / "iid").glob("detail-part-*.parquet"))
        },
        "targets": plan_targets,
    }
    dest = args.out / "plan.json"
    dest.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Frozen {len(targets)} targets, 2 replicates, 8 arms: {digest(dest)}")


if __name__ == "__main__":
    main()
