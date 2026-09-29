# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze the frozen multi-contact conditioning experiment without inference.

The practical primary compares predicted large-context decoding against the
same archived 100-rollout source plus 100 fresh iid rollouts. Supplied context
pairs are present in every complete conditioned document, so full-pipeline
votes for those pairs equal N, rather than N plus repeated continuation votes.

Mechanistic scoring excludes the union of all supplied context sets for a
protein/replicate from every arm and from its true-contact denominator. This
removes both prefix credit and copied context from continuation measurements.
Two replicates are averaged within each protein before bootstrapping proteins.
Intervals are pointwise, conditional on the archived first pass and saved runs.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

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
CONDITIONED_ARMS = ARMS[2:]
DERIVED_ARMS = ("iid200", "source_plus_iid200")
PRIMARY_REFERENCE = "source_plus_iid200"


def read_matrix(
    archive: np.lib.npyio.NpzFile, key: str, length: int, maximum: float, counts: bool
) -> np.ndarray:
    """Validate a saved symmetric score matrix and return an independent array."""
    if key not in archive:
        raise ValueError(f"missing matrix {key}")
    matrix = np.asarray(archive[key], dtype=float)
    if matrix.shape != (length, length):
        raise ValueError(
            f"{key}: expected shape {(length, length)}, got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{key}: nonfinite scores")
    if (matrix < 0).any() or (matrix > maximum).any():
        raise ValueError(f"{key}: scores outside [0, {maximum}]")
    if not np.array_equal(matrix, matrix.T):
        raise ValueError(f"{key}: matrix is not symmetric")
    if counts and not np.equal(matrix, np.floor(matrix)).all():
        raise ValueError(f"{key}: vote counts must be integers")
    return matrix


def validate_pairs(pairs: list, length: int, label: str) -> list[tuple[int, int]]:
    """Check pair identity and bounds without using any reference information."""
    result = []
    for pair in pairs:
        if len(pair) != 2 or any(not isinstance(value, int) for value in pair):
            raise ValueError(f"{label}: expected integer pairs")
        i, j = pair
        if not 0 <= i < j < length:
            raise ValueError(f"{label}: invalid pair {pair}")
        result.append((i, j))
    if len(set(result)) != len(result):
        raise ValueError(f"{label}: duplicate pairs")
    return result


def candidate_universes(
    target: dict, contexts: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build canonical resolved candidates, truth, and a shared context-free mask."""
    length = target["L"]
    resolved = np.asarray(target["resolved"], dtype=int)
    if (
        len(np.unique(resolved)) != len(resolved)
        or (resolved < 0).any()
        or (resolved >= length).any()
    ):
        raise ValueError(f"{target['stem']}: invalid resolved residues")
    resolved = np.sort(resolved)
    ai, aj = np.triu_indices(len(resolved), k=1)
    pi, pj = resolved[ai], resolved[aj]
    keep = pj - pi >= 6
    pi, pj = pi[keep], pj[keep]
    truth = np.zeros((length, length), dtype=bool)
    for i, j in validate_pairs(target["truth"], length, "truth"):
        if j - i >= 6:
            truth[i, j] = True
    excluded = np.zeros((length, length), dtype=bool)
    if set(contexts) != set(ARMS):
        raise ValueError(f"{target['stem']}: context arms must be {ARMS}")
    for arm in ARMS:
        pairs = validate_pairs(contexts[arm], length, arm)
        if arm in ("iid", "iid_repeat") and pairs:
            raise ValueError(f"{arm}: unconditioned arm contains context")
        for i, j in pairs:
            if j - i < 6:
                raise ValueError(f"{arm}: context pair violates minimum separation")
            if arm.startswith("true_") and not truth[i, j]:
                raise ValueError(f"{arm}: context is not a true contact")
            if arm.startswith("false_") and truth[i, j]:
                raise ValueError(f"{arm}: context contains a true contact")
            excluded[i, j] = True
    return pi, pj, truth, ~excluded[pi, pj]


def complete_votes(votes: np.ndarray, context: list, n_rollouts: int) -> np.ndarray:
    """Count each supplied context pair once per complete rollout, without mutation."""
    complete = votes.copy()
    for i, j in context:
        complete[i, j] = complete[j, i] = n_rollouts
    return complete


def precision_at_r(
    matrix: np.ndarray, truth: np.ndarray, pi: np.ndarray, pj: np.ndarray
) -> tuple[float, np.ndarray, int]:
    """Use canonical stable score ordering and a fixed true-contact denominator."""
    labels = truth[pi, pj]
    n_true = int(labels.sum())
    if n_true <= 0:
        raise ValueError("scoring universe has no true contacts")
    selected = np.argsort(-matrix[pi, pj], kind="stable")[:n_true]
    return float(labels[selected].sum() / n_true), selected, n_true


def prediction_changes(
    votes: np.ndarray,
    baseline: np.ndarray,
    truth: np.ndarray,
    pi: np.ndarray,
    pj: np.ndarray,
    n_rollouts: int,
    probability: np.ndarray | None,
    baseline_probability: np.ndarray,
) -> dict:
    """Measure changed predictions separately from correct predictions."""
    _, selected, n_true = precision_at_r(votes, truth, pi, pj)
    _, original, _ = precision_at_r(baseline, truth, pi, pj)
    frequency_change = np.abs(votes[pi, pj] - baseline[pi, pj]) / n_rollouts
    probability_change = (
        None
        if probability is None
        else np.abs(probability[pi, pj] - baseline_probability[pi, pj])
    )
    return {
        "top_R_turnover": 1.0 - len(np.intersect1d(selected, original)) / n_true,
        "vote_frequency_MAD": float(frequency_change.mean()),
        "vote_frequency_L1": float(frequency_change.sum()),
        "probability_MAD": None
        if probability_change is None
        else float(probability_change.mean()),
        "probability_L1": None
        if probability_change is None
        else float(probability_change.sum()),
    }


def score_target(
    target: dict,
    archive: np.lib.npyio.NpzFile,
    source_votes: np.ndarray,
    n_rollouts: int,
    n_repeats: int,
    source_n_rollouts: int,
) -> tuple[list, list]:
    """Score all arms of one protein on full and shared withheld universes."""
    if len(target["contexts"]) != n_repeats:
        raise ValueError(f"{target['stem']}: wrong number of context replicates")
    score_rows, change_rows = [], []
    for repeat, contexts in enumerate(target["contexts"]):
        pi, pj, truth, withheld = candidate_universes(target, contexts)
        votes = {
            arm: read_matrix(
                archive, f"r{repeat}__{arm}__votes", target["L"], n_rollouts, True
            )
            for arm in ARMS
        }
        probability = {
            arm: read_matrix(
                archive, f"r{repeat}__{arm}__prob", target["L"], 1.0, False
            )
            for arm in ARMS
            if arm != "iid_repeat"
        }
        matrices = dict(votes)
        matrices["iid200"] = votes["iid"] + votes["iid_repeat"]
        matrices[PRIMARY_REFERENCE] = source_votes + votes["iid"]
        for scope, ii, jj in (
            ("full_pipeline", pi, pj),
            ("withheld_continuation", pi[withheld], pj[withheld]),
        ):
            for arm, matrix in matrices.items():
                context = contexts.get(arm, [])
                scored = (
                    complete_votes(matrix, context, n_rollouts)
                    if scope == "full_pipeline"
                    else matrix
                )
                precision, _, n_true = precision_at_r(scored, truth, ii, jj)
                count = (
                    2 * n_rollouts
                    if arm == "iid200"
                    else source_n_rollouts + n_rollouts
                    if arm == PRIMARY_REFERENCE
                    else n_rollouts
                )
                score_rows.append(
                    {
                        "dataset": target["dataset"],
                        "stem": target["stem"],
                        "repeat": repeat,
                        "scope": scope,
                        "arm": arm,
                        "L": target["L"],
                        "precision": precision,
                        "n_true": n_true,
                        "n_candidates": len(ii),
                        "n_context": len(context),
                        "n_rollouts": count,
                    }
                )
            # Changes use generated continuations before inserting given pairs,
            # and the withheld view additionally removes all copied context.
            for arm in ARMS[1:]:
                change_rows.append(
                    dict(
                        dataset=target["dataset"],
                        stem=target["stem"],
                        repeat=repeat,
                        scope="full_continuation"
                        if scope == "full_pipeline"
                        else scope,
                        arm=arm,
                        reference="iid",
                        n_true=n_true,
                        n_candidates=len(ii),
                        **prediction_changes(
                            votes[arm],
                            votes["iid"],
                            truth,
                            ii,
                            jj,
                            n_rollouts,
                            probability.get(arm),
                            probability["iid"],
                        ),
                    )
                )
    return score_rows, change_rows


def paired_interval(values: np.ndarray) -> dict:
    """Bootstrap independent protein effects with 20,000 deterministic draws."""
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("paired interval requires nonempty finite protein effects")
    draw = np.random.default_rng(254).integers(0, len(values), (20_000, len(values)))
    lower, upper = np.quantile(values[draw].mean(axis=1), [0.025, 0.975])
    return {
        "n_proteins": len(values),
        "mean_delta": float(values.mean()),
        "ci_low": float(lower),
        "ci_high": float(upper),
        "proteins_improved": int((values > 0).sum()),
        "proteins_tied": int((values == 0).sum()),
    }


def margin_status(lower: float, upper: float, margin: float) -> str:
    """Classify evidence against the prespecified practical effect threshold."""
    if lower >= margin:
        return "gain_at_least_margin_supported"
    if upper < margin:
        return "target_gain_ruled_out"
    return "inconclusive_at_margin"


def aggregate_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Average replicates within each protein before any population inference."""
    identity = ["dataset", "stem", "repeat", "scope", "arm"]
    if scores.duplicated(identity).any():
        raise ValueError("duplicate protein/replicate/arm score")
    return scores.groupby(["dataset", "stem", "scope", "arm"], as_index=False).agg(
        precision=("precision", "mean"),
        n_repeats=("repeat", "nunique"),
        L=("L", "first"),
        n_true_mean=("n_true", "mean"),
        n_true_min=("n_true", "min"),
        n_true_max=("n_true", "max"),
        n_candidates_mean=("n_candidates", "mean"),
        n_context=("n_context", "mean"),
        n_rollouts=("n_rollouts", "first"),
    )


def summarize_scores(
    per_protein: pd.DataFrame, practical_margin: float
) -> pd.DataFrame:
    """Report paired contrasts with the practical and diagnostic roles explicit."""
    rows = []
    for scope, group in per_protein.groupby("scope", sort=True):
        values = group.pivot(
            index=["dataset", "stem"], columns="arm", values="precision"
        )
        contrasts = [(arm, "iid") for arm in ARMS[1:] + DERIVED_ARMS]
        contrasts += [
            (arm, reference)
            for arm in ("pred_small", "pred_large")
            for reference in ("iid200", PRIMARY_REFERENCE)
        ]
        if scope == "withheld_continuation":
            contrasts += [
                (f"true_{dose}", f"false_{dose}") for dose in ("small", "large")
            ]
        for arm, reference in contrasts:
            interval = paired_interval((values[arm] - values[reference]).to_numpy())
            primary = (
                scope == "full_pipeline"
                and arm == "pred_large"
                and reference == PRIMARY_REFERENCE
            )
            practical = scope == "full_pipeline" and arm.startswith("pred_")
            role = (
                "practical_primary"
                if primary
                else "practical_secondary"
                if practical
                else "diagnostic"
            )
            rows.append(
                dict(
                    scope=scope,
                    arm=arm,
                    reference=reference,
                    role=role,
                    arm_mean=float(values[arm].mean()),
                    reference_mean=float(values[reference].mean()),
                    **interval,
                    practical_margin=practical_margin if practical else None,
                    margin_status=margin_status(
                        interval["ci_low"], interval["ci_high"], practical_margin
                    )
                    if practical
                    else "not_a_practical_test",
                )
            )
    return pd.DataFrame(rows)


def analyze(plan_path: Path, run: Path, out: Path) -> dict:
    """Validate completed units, write score tables, and summarize the primary test."""
    plan = json.loads(plan_path.read_text())
    targets = plan["targets"]
    if not targets or len({target["stem"] for target in targets}) != len(targets):
        raise ValueError("plan targets must have unique nonempty stems")
    n_rollouts, n_repeats = int(plan["n_rollouts"]), int(plan["n_repeats"])
    source_n_rollouts = int(plan["source_n_rollouts"])
    if min(n_rollouts, n_repeats, source_n_rollouts) <= 0:
        raise ValueError("rollout and replicate counts must be positive")
    practical_margin = float(plan["practical_margin"])
    if not 0 < practical_margin < 1:
        raise ValueError("practical margin must be in (0, 1)")
    source_path = plan_path.parent / plan["source_votes_file"]
    score_rows, change_rows = [], []
    with np.load(source_path, allow_pickle=False) as source:
        for target in targets:
            unit = run / "units" / f"{target['stem']}.npz"
            if not unit.is_file():
                raise FileNotFoundError(f"missing completed unit: {unit}")
            source_votes = read_matrix(
                source, target["stem"], target["L"], source_n_rollouts, True
            )
            with np.load(unit, allow_pickle=False) as archive:
                scored, changed = score_target(
                    target,
                    archive,
                    source_votes,
                    n_rollouts,
                    n_repeats,
                    source_n_rollouts,
                )
            score_rows.extend(scored)
            change_rows.extend(changed)
    scores = pd.DataFrame(score_rows)
    per_protein = aggregate_scores(scores)
    if not (per_protein.n_repeats == n_repeats).all():
        raise ValueError("incomplete replicate coverage")
    summary = summarize_scores(per_protein, practical_margin)
    primary = summary[summary.role == "practical_primary"]
    if len(primary) != 1:
        raise ValueError("expected exactly one practical primary contrast")
    conclusion = {
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "prediction_source": plan["prediction_source"],
        "source_votes_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "n_targets": len(targets),
        "n_repeats": n_repeats,
        "primary": primary.iloc[0].to_dict(),
        "interval_unit": "protein, after averaging context/sampling replicates",
        "limitation": "Conditional on archived first-pass votes and saved runs. Pointwise intervals; "
        "secondary comparisons are exploratory. A 100+100 rollout budget is not a "
        "claim of identical measured compute or latency. Oracle-conditioned arms "
        "and withheld-universe effects are mechanistic diagnostics.",
    }
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "conditioning_per_repeat.csv", index=False)
    per_protein.to_csv(out / "conditioning_per_protein.csv", index=False)
    summary.to_csv(out / "conditioning_summary.csv", index=False)
    pd.DataFrame(change_rows).to_csv(out / "conditioning_changes.csv", index=False)
    (out / "conditioning_conclusion.json").write_text(
        json.dumps(conclusion, indent=2) + "\n"
    )
    return conclusion


def main() -> int:
    """Analyze a completed CPU-readable conditioning run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.plan, args.run, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
