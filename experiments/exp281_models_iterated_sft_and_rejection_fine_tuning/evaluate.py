"""Score saved, unselected trajectories; never evaluate rejection-selected winners.

Final-answer R-precision/AUC rank residue pairs by frequency across independent
final answers at a common budget. Draft voting uses each section once. Both use
the library's contacts-v1 metric implementation and its existing tie convention.
Single-answer F1 and invalid-output rates are reported separately.
"""

import argparse
import csv
import itertools
import math
from collections import Counter, defaultdict

import fsspec
import numpy as np
from marinfold.document_structures.contacts_v1.inference import _metric_rows
from marinfold.document_structures.contacts_v1_multi import END, FINAL, parse_history

from common import files, rows, write_json
from generate import decode_pairs


def score_pool(pool: list[dict]) -> dict:
    """Score all candidates including invalid answers, which contribute zero votes."""
    first = pool[0]
    if any(c["bootstrap"] for c in pool):
        raise ValueError("bootstrap examples contain reference closures, not sampled answers")
    candidate_ids = [c["candidate_id"] for c in pool]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("duplicate candidate ids in evaluation pool")
    length = first["n_residues"]
    truth_pairs = decode_pairs(parse_history([FINAL, *first["reference"], END]).final, first["positions"])
    truth = np.zeros((length, length), dtype=bool)
    for i, j in truth_pairs:
        truth[i, j] = truth[j, i] = True
    final_votes, draft_votes = Counter(), Counter()
    jaccard = []
    sections = 0
    f1s = []
    for candidate in pool:
        if not candidate["valid"]:
            f1s.append(0.0)
            continue
        parsed = parse_history(candidate["generated"])
        final_votes.update(set(decode_pairs(parsed.final, first["positions"])))
        hypotheses = [set(decode_pairs(h, first["positions"])) for h in parsed.hypotheses]
        for hypothesis in hypotheses:
            draft_votes.update(hypothesis)
        sections += len(hypotheses)
        jaccard.extend(len(a & b) / max(1, len(a | b)) for a, b in itertools.combinations(hypotheses, 2))
        f1s.append(candidate["score"]["f1"])
    result = {"target_id": first["target_id"], "forced": first["forced"], "budget": first["budget"],
              "candidates": len(pool), "valid_fraction": sum(c["valid"] for c in pool) / len(pool),
              "final_f1": float(np.mean(f1s)), "sections_per_trajectory": sections / len(pool),
              "hypothesis_jaccard": float(np.mean(jaccard)) if jaccard else None,
              "hypothesis_union_recall": len(set(draft_votes) & set(truth_pairs)) / max(1, len(truth_pairs)),
              "generated_tokens": sum(len(c["generated"]) for c in pool)}
    for name, votes in (("final", final_votes), ("draft_vote", draft_votes)):
        matrix = np.zeros((length, length), dtype=np.float32)
        for (i, j), count in votes.items():
            matrix[i, j] = matrix[j, i] = count
        for span, metrics in _metric_rows(matrix, truth, length, 6).items():
            for metric in ("r_precision", "auc"):
                value = metrics.get(metric, float("nan"))
                result[f"{name}_{metric}_{span}"] = value if math.isfinite(value) else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = []
    seen = set()
    for path in files(args.candidates):
        groups = defaultdict(list)
        for candidate in rows(path):
            groups[(candidate["target_id"], candidate["forced"], candidate["budget"])].append(candidate)
        for key, pool in groups.items():
            if key in seen:
                raise ValueError("evaluation pool split or duplicated across shards")
            seen.add(key)
            records.append(score_pool(pool))
    with fsspec.open(args.output + ".csv", "w", auto_mkdir=True) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(args.output + ".json", {"groups": len(records), "protocol": "all unselected candidates; pair-frequency final ranking",
               "primary": "final_f1", "macro_final_f1": float(np.mean([r["final_f1"] for r in records]))})


if __name__ == "__main__":
    main()
