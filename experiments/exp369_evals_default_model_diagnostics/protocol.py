"""Frozen oracle-contact sampling and leakage-free remaining-contact metrics."""

import hashlib

import numpy as np

COUNTS = (0, 1, 2, 4, 8, 16, 32, 64, 128)
FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80)
REPEATS = 2
ROLLOUTS = 100


def stable_seed(key: str) -> int:
    """Return a portable deterministic RNG seed."""
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % (2**31)


def contact_order(pairs: list[list[int]], elements: list[int], policy: str, key: str) -> list[list[int]]:
    """Order distinct contacts uniformly or by coverage of helix/strand segments.

    The balanced policy chooses an eligible element with the fewest incident
    supplied contacts, then samples one of its unused incident contacts. Both
    endpoint elements receive credit. Exhausted elements leave the competition;
    coil-only contacts fill the tail after all SSE-incident contacts. This gives
    a nested path whose first pass covers every coverable element, then two
    contacts per element, and so on, with unavoidable shared-edge overshoot.
    """
    if len({tuple(pair) for pair in pairs}) != len(pairs):
        raise ValueError("Contacts must be unique")
    rng = np.random.default_rng(stable_seed(key))
    order = rng.permutation(len(pairs)).tolist()
    if policy == "random":
        return [pairs[i] for i in order]
    if policy != "sse_balanced":
        raise ValueError(policy)
    touched = [{elements[i], elements[j]} - {-1} for i, j in pairs]
    incidence = {element: {index for index, group in enumerate(touched) if element in group}
                 for element in sorted(set(elements) - {-1})}
    coverage = {element: 0 for element in incidence}
    remaining = set(range(len(pairs)))
    result = []
    while remaining:
        active = [element for element, indices in incidence.items() if indices & remaining]
        if not active:
            result.extend(pairs[i] for i in order if i in remaining)
            break
        minimum = min(coverage[element] for element in active)
        element = int(rng.choice([element for element in active if coverage[element] == minimum]))
        index = int(rng.choice(sorted(incidence[element] & remaining)))
        result.append(pairs[index])
        remaining.remove(index)
        for endpoint in touched[index]:
            coverage[endpoint] += 1
    return result


def doses(n_true: int) -> list[dict]:
    """Map predeclared count/fraction doses to integer counts with truth remaining."""
    values = [{"axis": "count", "dose": k, "k": k} for k in COUNTS if k < n_true]
    values += [{"axis": "fraction", "dose": fraction, "k": max(1, int(np.floor(fraction * n_true)))}
               for fraction in FRACTIONS if max(1, int(np.floor(fraction * n_true))) < n_true]
    return values


def coverage_milestones(order: list[list[int]], elements: list[int]) -> list[dict]:
    """Find doses covering every coverable element with one, two, or four edges.

    Elements with fewer available incident true contacts are saturated at their
    available count. The same integer dose can then be tested under random
    sampling, so coverage-based comparisons do not have unequal contact budgets.
    """
    touched = [{elements[i], elements[j]} - {-1} for i, j in order]
    total = {element: sum(element in pair for pair in touched) for element in set(elements) - {-1}}
    total = {element: value for element, value in total.items() if value}
    if not total:
        return []
    count = dict.fromkeys(total, 0)
    pending = {1, 2, 4}
    result = []
    for k, group in enumerate(touched, start=1):
        for element in group:
            count[element] += 1
        for quota in sorted(pending):
            if all(count[element] >= min(quota, value) for element, value in total.items()):
                if k < len(order):
                    result.append({"axis": "sse_round", "dose": quota, "k": k})
                pending.remove(quota)
    return result


def remaining_metrics(scores: np.ndarray, truth: list[list[int]], resolved: list[int],
                      supplied: list[list[int]], min_separation: int = 6) -> dict:
    """Score exp89's stable top-R ranking after removing supplied candidate pairs.

    R is recomputed from the remaining truth, independently for each range.
    R=0 is undefined, not perfect accuracy. Resolved upper-triangle candidates
    and stable descending score order match the established evaluator.
    """
    length = len(scores)
    if scores.shape != (length, length) or not np.isfinite(scores).all():
        raise ValueError("Expected a finite square vote matrix")
    resolved_mask = np.zeros(length, dtype=bool)
    resolved_mask[resolved] = True
    ii, jj = np.triu_indices(length, k=min_separation)
    keep = resolved_mask[ii] & resolved_mask[jj]
    excluded = np.zeros((length, length), dtype=bool)
    supplied_indices = np.asarray(supplied, dtype=int).reshape(-1, 2)
    excluded[supplied_indices[:, 0], supplied_indices[:, 1]] = True
    keep &= ~excluded[ii, jj]
    ii, jj = ii[keep], jj[keep]
    positives = np.zeros((length, length), dtype=bool)
    truth_indices = np.asarray(truth, dtype=int).reshape(-1, 2)
    positives[truth_indices[:, 0], truth_indices[:, 1]] = True
    labels = positives[ii, jj]
    n_true = int(labels.sum())
    if not n_true:
        return {"r_precision": None, "n_remaining_true": 0, "n_candidates": len(ii)}
    top = np.argsort(-scores[ii, jj].astype(np.float64), kind="stable")[:n_true]
    return {"r_precision": float(labels[top].mean()), "n_remaining_true": n_true,
            "n_candidates": len(ii), "random_expectation": n_true / len(ii)}
