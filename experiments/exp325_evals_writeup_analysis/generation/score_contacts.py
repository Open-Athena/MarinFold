"""Reduce archived test rollouts into contact metrics and paired sampling diagnostics.

Uses exp89's metric functions and exp321's ordered-rollout diagnostic unchanged.
No model loads or predictor calls occur here. Main contact votes exclude unfinished
rollouts (exp277 recipe); the separate exp321-compatible diagnostic consensus uses
all parsed maps and gates invalid individual maps to zero. Both are saved explicitly.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXPERIMENTS = ROOT.parent
sys.path.insert(0, str(EXPERIMENTS / "exp89_evals_contacts_v1_model_on_eval_set"))
sys.path.insert(0, str(EXPERIMENTS / "exp321_evals_null_sequence_contrastive_guidance_for_contacts"))
from compute_metrics import metric_rows, resolved_pairs, true_matrix
from analyze_results import mean_pairwise_jaccard, ordered_true_pairs, rollout_r_precision, vote_matrix

CONTACT = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")


def parse_rollout(row: dict) -> tuple[list[tuple[int, int]], int]:
    """Recover emission order and the invalid-position/statement count from raw text."""
    mapping = {int(k): int(v) for k, v in row["position_map"].items()}
    matches = CONTACT.findall(row["text"])
    malformed = row["text"].count("<contact>") - len(matches)
    pairs = []
    seen = set()
    for left, right in matches:
        i, j = mapping.get(int(left)), mapping.get(int(right))
        if i is None or j is None:
            malformed += 1
            continue
        pair = (min(i, j), max(i, j))
        if pair[1] - pair[0] >= 6 and pair not in seen:
            pairs.append(pair)
            seen.add(pair)
    return pairs, malformed


def main() -> None:
    """Require all 217 targets and check sparse votes against their original samples."""
    scratch = ROOT / "scratch/contacts"
    truth = {r["stem"]: r for r in map(json.loads, (scratch / "inputs/gt_universe_scored.jsonl").read_text().splitlines())}
    wanted = set(pd.read_csv(ROOT / "data/missing_eval_test_targets.csv").stem)
    markers = [json.loads(p.read_text()) for p in sorted((scratch / "results/complete").glob("*.json"))]
    units = [r["stem"] for m in markers for r in m["units"]]
    if len(units) != 217 or set(units) != wanted:
        raise ValueError(f"Need 217 unique completed test targets; got {len(units)}")
    scores = pd.concat([pd.read_parquet(p) for p in sorted((scratch / "results/scores").glob("*.parquet"))])
    metrics, diagnostics, pins = [], [], {}
    for stem in sorted(wanted):
        path = scratch / "results/rollouts" / f"{stem}.json"
        archive = json.loads(path.read_text())
        pins[stem] = hashlib.sha256(path.read_bytes()).hexdigest()
        record = truth[stem]
        length = record["L"]
        rollouts = archive["rollouts"]
        if len(rollouts) != 100 or archive["L"] != length:
            raise ValueError(f"{stem}: inconsistent sample count or length")
        parsed = [parse_rollout(r) for r in rollouts]
        maps = [p for p, _ in parsed]
        stopped = [r["finish_reason"] == "stop" for r in rollouts]
        valid = [ok and errors == 0 for ok, (_, errors) in zip(stopped, parsed, strict=True)]
        votes = vote_matrix([m for m, ok in zip(maps, stopped, strict=True) if ok], length)
        sparse = scores[scores.stem == stem]
        saved = np.zeros((length, length), dtype=np.float32)
        saved[sparse.i.to_numpy(), sparse.j.to_numpy()] = sparse.votes
        if not np.array_equal(np.triu(votes, 1), saved):
            raise ValueError(f"{stem}: raw rollouts do not reproduce saved votes")
        tmat = true_matrix(length, record["contacts"])
        pairs = resolved_pairs(np.asarray(record["resolved"], dtype=int))
        metrics.extend({"stem": stem, "eval_set": "eval-test", "model": "exp277-step266344", **r}
                       for r in metric_rows(votes, tmat, *pairs, length, with_precision=True))
        diagnostic_metrics = pd.DataFrame(metric_rows(vote_matrix(maps, length), tmat, *pairs, length, with_precision=True))
        resolved = set(record["resolved"])
        for region, minimum in (("all", 6), ("long", 24)):
            selected = [[(i, j) for i, j in m if i in resolved and j in resolved and j - i >= minimum] for m in maps]
            true = ordered_true_pairs(record, region)
            individual = [rollout_r_precision(m, true) if ok else 0.0 for m, ok in zip(selected, valid, strict=True)]
            union = set().union(*map(set, selected))
            diagnostics.append({
                "stem": stem, "eval_set": "eval-test", "mode": "full_iid_single", "N": 100, "range": region,
                "consensus_r_precision": float(diagnostic_metrics.loc[(diagnostic_metrics["range"] == region) & (diagnostic_metrics.cut == "R"), "precision"].iloc[0]),
                "mean_validity_gated_rollout_r_precision": float(np.mean(individual)),
                "validity_gated_oracle_r_precision": float(np.max(individual)),
                "true_union_recall": len(union & true) / len(true) if true else np.nan,
                "mean_pairwise_jaccard": mean_pairwise_jaccard(selected),
                "unique_maps": len({frozenset(m) for m in selected}),
                "finished": sum(stopped), "invalid_rollouts": 100 - sum(valid),
            })
    data = ROOT / "data"
    pd.DataFrame(metrics).to_csv(data / "test_contact_metrics.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(data / "test_sampling_metrics.csv", index=False)
    pd.concat([pd.read_parquet(p) for p in sorted((scratch / "results/timings").glob("*.parquet"))]).sort_values("stem").to_csv(data / "contact_timings.csv", index=False)
    baseline = pd.read_csv(scratch / "inputs/per_protein.csv.gz")
    baseline = baseline[baseline.predictor.isin(["ESMFold", "ESMFold2", "Protenix-v2 single-seq", "Protenix-v2 + MSA", "seq-KNN (decontaminated corpus)"]) & (baseline.cut == "R")]
    baseline.to_csv(data / "inputs/baseline_contact_metrics.csv", index=False)
    (data / "test_contact_run.json").write_text(json.dumps({
        "checkpoint": "contacts-v1-exp277-m2-p06-full-epoch-1.5B-step-266344",
        "n_targets": 217, "n_rollouts": 21700, "runner": "Modal H100, us-east, 8 shards",
        "temperature": 1.0, "top_p": 0.95, "top_k": -1, "seed": 0,
        "budget": "min(8192-prompt_tokens, 6L+128)", "chunk": 1,
        "raw_rollout_sha256": pins,
        "main_votes": "Only finished maps, same as exp277",
        "sampling_votes": "All parsed maps, same as exp321; unfinished/malformed individual scores zero",
    }, indent=2) + "\n")
    print(pd.DataFrame(metrics).query("cut == 'R'").groupby("range").precision.mean())


if __name__ == "__main__":
    main()
