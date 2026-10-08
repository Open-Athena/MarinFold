"""Score intra/inter contacts on one fixed set of full-complex predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import gemmi
import numpy as np
import pyarrow.parquet as pq
from freeze_foldbench_eval import atom_site_maps, selected_structure
from marinfold.document_structures.contacts_v1 import analyze_structure
from score_foldbench_contacts import bootstrap_groups, load_votes, write_csv

HERE = Path(__file__).resolve().parent
MODELS = ("MarinFold multichain", "MarinFold default + 10G", "ESMFold2", "AlphaFold3")
REGIONS = ("intra", "inter", "chain_A", "chain_B", "intra_long")


def candidate_pairs(target: dict, region: str) -> list[tuple[int, int]]:
    """Use experimental resolved positions and original chain boundaries."""
    a, b = target["resolved_positions_by_chain"]
    if region == "inter":
        return [(i, j) for i in a for j in b]
    chains = [a] if region == "chain_A" else [b] if region == "chain_B" else [a, b]
    separation = 24 if region == "intra_long" else 6
    return [
        (i, j)
        for chain in chains
        for index, i in enumerate(chain)
        for j in chain[index + 1 :]
        if j - i >= separation
    ]


def precision_at_r(target: dict, scores: dict, region: str) -> dict:
    """Rank all eligible pairs, including zero scores; preserve row order on ties."""
    pairs = candidate_pairs(target, region)
    gt = {(i, j) for i, j, _ in target["all_contacts"]}
    truth = np.array([pair in gt for pair in pairs], dtype=bool)
    r = int(truth.sum())
    values = np.array([scores.get(pair, 0.0) for pair in pairs])
    order = np.argsort(-values, kind="stable")
    correct = int(truth[order[:r]].sum())
    # Also record the expectation for a random ordering within the cutoff tie.
    expected = float("nan")
    if r:
        threshold = values[order[r - 1]]
        above = values > threshold
        tied = values == threshold
        expected = float(
            (truth[above].sum() + (r - above.sum()) * truth[tied].mean()) / r
        )
    return {
        "region": region,
        "n_candidate": len(pairs),
        "n_true": r,
        "n_correct": correct,
        "r_precision": correct / r if r else float("nan"),
        "tie_expected_r_precision": expected,
        "random_r_precision": r / len(pairs) if pairs else float("nan"),
        "n_positive_predictions": int((values > 0).sum()),
    }


def structure_scores(target: dict, path: Path) -> dict[tuple[int, int], float]:
    """Map model input A/B to canonical coordinates and score pyconfind degrees."""
    auth, residue_map = atom_site_maps(path, ["A", "B"])
    analysis = analyze_structure(
        selected_structure(path, auth), entry_id=target["stem"], max_chains=2
    )
    offsets = dict(zip(auth, target["chain_offsets"], strict=True))
    chain_index = {cid: i for i, cid in enumerate(auth)}
    mapping = {}
    for residue in analysis.residues:
        local = residue_map[(residue.chain, residue.resnum)]
        seq = target["chain_sequences"][chain_index[residue.chain]]
        observed = gemmi.find_tabulated_residue(residue.resname).one_letter_code
        if not 0 <= local < len(seq) or observed != seq[local]:
            raise ValueError(
                f"{path}: predicted residue does not match input: {residue} at {local}"
            )
        mapping[residue.seq_index] = offsets[residue.chain] + local
    if not set(target["resolved_positions"]).issubset(mapping.values()):
        raise ValueError(f"{path}: prediction misses experimental resolved residues")
    return {
        tuple(sorted((mapping[c.seq_i], mapping[c.seq_j]))): c.degree
        for c in analysis.contacts
    }


def main() -> None:
    """Require all four predictors on every complex before writing comparison scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multichain", type=Path, required=True)
    parser.add_argument("--default", type=Path, required=True)
    parser.add_argument("--esm", type=Path, required=True)
    parser.add_argument("--af3", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "data/four_model_v1")
    parser.add_argument("--marinfold-only", action="store_true")
    parser.add_argument("--split", choices=("all", "test", "dev"), default="all")
    args = parser.parse_args()
    targets = json.loads((HERE / "data/four_model_v1/targets.json").read_text())
    if args.split != "all":
        targets = [target for target in targets if target["split"] == args.split]
    predictions = {
        MODELS[0]: load_votes([args.multichain / "scores"]),
        MODELS[1]: load_votes([args.default / "scores"]),
    }
    provenance = []
    if not args.marinfold_only:
        for model, directory in ((MODELS[2], args.esm), (MODELS[3], args.af3)):
            predictions[model] = {}
            for t in targets:
                path = (
                    directory
                    / t["stem"]
                    / (
                        "structure.cif"
                        if model == MODELS[2]
                        else t["stem"] + "_model.cif"
                    )
                )
                predictions[model][t["stem"]] = structure_scores(t, path)
                provenance.append(
                    {
                        "model": model,
                        "stem": t["stem"],
                        "file": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
    rows = []
    for t in targets:
        for model, scores in predictions.items():
            if t["stem"] not in scores:
                raise ValueError(f"{model}: missing {t['stem']}")
            for region in REGIONS:
                rows.append(
                    dict(
                        stem=t["stem"],
                        split=t["split"],
                        group_id=t["group_id"],
                        complex_type=t["complex_type"],
                        n_residues=t["L"],
                        model=model,
                        **precision_at_r(t, scores[t["stem"]], region),
                    )
                )
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "per_target.csv", rows)
    summary = []
    for split in ("test", "dev", "all"):
        for model in predictions:
            for region in REGIONS:
                selected = [
                    r
                    for r in rows
                    if (split == "all" or r["split"] == split)
                    and r["model"] == model
                    and r["region"] == region
                    and np.isfinite(r["r_precision"])
                ]
                if not selected:
                    continue
                low, high = bootstrap_groups(selected, 10000, 350)
                summary.append(
                    {
                        "split": split,
                        "model": model,
                        "region": region,
                        "n_targets": len(selected),
                        "n_groups": len({r["group_id"] for r in selected}),
                        "mean_r_precision": float(
                            np.mean([r["r_precision"] for r in selected])
                        ),
                        "ci_low": low,
                        "ci_high": high,
                        "mean_tie_expected_r_precision": float(
                            np.mean([r["tie_expected_r_precision"] for r in selected])
                        ),
                        "mean_random_r_precision": float(
                            np.mean([r["random_r_precision"] for r in selected])
                        ),
                    }
                )
    write_csv(args.output / "summary.csv", summary)
    if provenance:
        write_csv(args.output / "structure_provenance.csv", provenance)
    timings = []
    for model, path in ((MODELS[0], args.multichain), (MODELS[1], args.default)):
        for p in sorted((path / "timings").glob("*.parquet")):
            timings.extend(
                {**r, "comparison_model": model} for r in pq.read_table(p).to_pylist()
            )
    (args.output / "marinfold_timings.json").write_text(
        json.dumps(timings, indent=2) + "\n"
    )
    for r in summary:
        if r["split"] == "test" and r["region"] in ("intra", "inter"):
            print(r)


if __name__ == "__main__":
    main()
