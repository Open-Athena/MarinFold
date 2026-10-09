"""Apply exp89's unchanged metric to full and oracle-reduced universes."""

import importlib.util
import os
from pathlib import Path

import numpy as np

from protocol import oracle_pairs, vote_matrix

METRIC_PATH = Path(
    os.environ.get(
        "EXP89_METRICS_PATH",
        str(
            Path(__file__).resolve().parents[1]
            / "exp89_evals_contacts_v1_model_on_eval_set/compute_metrics.py"
        ),
    )
)
SPEC = importlib.util.spec_from_file_location("exp89_metrics", METRIC_PATH)
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


def score_votes(record: dict, rollouts: list[dict], *, reduced: bool) -> list[dict]:
    """Evaluate votes with optional given contacts removed from candidates."""
    length = record["L"]
    truth = METRICS.true_matrix(length, record["contacts"])
    pi, pj, separation = METRICS.resolved_pairs(
        np.asarray(record["resolved"], dtype=np.int64)
    )
    if reduced:
        supplied = {tuple(pair) for pair in oracle_pairs(record)}
        keep = np.asarray(
            [(int(i), int(j)) not in supplied for i, j in zip(pi, pj, strict=True)]
        )
        pi, pj, separation = pi[keep], pj[keep], separation[keep]
    return METRICS.metric_rows(
        vote_matrix(rollouts, length),
        truth,
        pi,
        pj,
        separation,
        length,
        with_precision=True,
    )


def summarize_samples(record: dict, rollouts: list[dict]) -> list[dict]:
    """Report single-sample set accuracy independently of tied vote rankings."""
    resolved = set(record["resolved"])
    truth = {
        (int(i), int(j))
        for i, j, d in record["contacts"]
        if d >= 0.001 and i in resolved and j in resolved
    }
    rows = []
    for band, separation in (("all", 6), ("long", 24)):
        actual = {pair for pair in truth if pair[1] - pair[0] >= separation}
        if not actual:
            continue
        values = []
        for rollout in rollouts:
            if rollout["finish_reason"] != "stop":
                continue
            predicted = {
                (i, j)
                for i, j in rollout["contacts"]
                if j - i >= separation and i in resolved and j in resolved
            }
            correct = len(predicted & actual)
            values.append(
                (
                    correct / len(predicted) if predicted else 0.0,
                    correct / len(actual),
                    2 * correct / (len(predicted) + len(actual)),
                    len(predicted),
                )
            )
        if not values:
            raise ValueError("No terminated samples to summarize")
        array = np.asarray(values)
        rows.append(
            dict(
                range=band,
                mean_precision=float(array[:, 0].mean()),
                mean_recall=float(array[:, 1].mean()),
                mean_f1=float(array[:, 2].mean()),
                oracle_best_f1=float(array[:, 2].max()),
                mean_predicted_contacts=float(array[:, 3].mean()),
                n_true=len(actual),
            )
        )
    return rows


def analyze_record(
    record: dict, unconditioned: list[dict], oracle: list[dict]
) -> list[dict]:
    """Score nested sample budgets and matched remaining-contact diagnostics."""
    rows = []
    for budget in (1, 10, 30, 100, 300, 1000):
        if budget > len(unconditioned):
            continue
        selected = unconditioned[:budget]
        if not any(r["finish_reason"] == "stop" for r in selected):
            continue
        for metric in score_votes(record, selected, reduced=False):
            rows.append({**metric, "mode": "unconditioned", "budget": budget})
    if oracle:
        for mode, selected in (
            ("unconditioned_remaining", unconditioned[:100]),
            ("unconditioned_remaining", unconditioned),
            ("oracle_half_remaining", oracle),
        ):
            for metric in score_votes(record, selected, reduced=True):
                rows.append({**metric, "mode": mode, "budget": len(selected)})
    return rows
