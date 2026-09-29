# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only sensitivity audit of the saved eval-val samples and cluster scores.

The historical single-rollout metric divides true positives by min(R, emitted
contacts), so short predictions can score perfectly despite missing contacts.
Keep it for comparison with the original experiment, and separately report
TP@R / R, where unfilled slots count as misses. Neither oracle is deployable.
The continuation analysis removes externally supplied seed rows before scoring.

Bootstrap intervals resample proteins, condition on this one saved set of
rollouts, and are exploratory pointwise intervals, not simultaneous intervals
or estimates of variation between fresh inference runs.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_metrics import ARMS, load_detail, resolved_pairs, true_matrix
from common import EXPECTED_UNITS, load_ground_truth, load_targets


def bootstrap(values: np.ndarray, replicates: int = 20_000) -> dict:
    """Return a protein-level mean and deterministic percentile interval."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap requires nonempty finite protein values")
    draws = np.random.default_rng(254).integers(0, len(values), (replicates, len(values)))
    low, high = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
    return dict(n_proteins=len(values), mean=float(values.mean()),
                ci_low=float(low), ci_high=float(high))


def rollout_scores(contacts: pd.DataFrame, truth: np.ndarray, n_true: int,
                   rollout_ids: np.ndarray, exclude_seed: bool) -> pd.DataFrame:
    """Score every supplied rollout, including empty continuations, at two cuts.

    ``contacts`` must already be restricted to resolved residues and the range.
    Stable rank sorting preserves the recorded emission order. Removing the seed
    makes room for the next predicted contact; we still cut at the original R.
    """
    if n_true <= 0:
        raise ValueError("R must be positive")
    current = contacts.loc[~contacts.is_seed] if exclude_seed else contacts
    current = current.sort_values(["rollout", "rank"], kind="stable")
    groups = {int(key): group for key, group in current.groupby("rollout", sort=False)}
    rows = []
    for rollout in rollout_ids:
        group = groups.get(int(rollout))
        n_emitted = 0 if group is None else len(group)
        n_top = min(n_true, n_emitted)
        true_positives = 0 if group is None else int(truth[
            group.i.to_numpy()[:n_top], group.j.to_numpy()[:n_top]].sum())
        rows.append(dict(rollout=int(rollout), n_emitted=n_emitted, n_top=n_top,
                         true_positives=true_positives,
                         legacy_precision=true_positives / n_top if n_top else 0.0,
                         fixed_R=true_positives / n_true))
    return pd.DataFrame(rows)


def audit_rollouts(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read existing samples and report oracle sensitivity and union coverage."""
    targets = load_targets()
    if len(targets) != EXPECTED_UNITS:
        raise ValueError(f"expected {EXPECTED_UNITS} eval-val proteins")
    keys = {(target.dataset, target.stem) for target in targets}
    ground_truth = load_ground_truth()
    oracle_rows, coverage_rows = [], []
    for arm in ARMS:
        detail = load_detail(run / arm.directory)
        if set(zip(detail.dataset, detail.stem)) != keys:
            raise ValueError(f"{arm.directory}: rollout targets differ from eval-val")
        for (dataset, stem), group in detail.groupby(["dataset", "stem"], sort=True):
            record = ground_truth[(dataset, stem)]
            length = record["L"]
            truth = true_matrix(length, record["contacts"])
            resolved = np.zeros(length, dtype=bool)
            resolved[record["resolved"]] = True
            pi, pj, psep = resolved_pairs(np.asarray(record["resolved"], dtype=np.int64))
            rollout_ids = np.sort(group.rollout.unique())
            if not np.array_equal(rollout_ids, np.arange(100)):
                raise ValueError(f"{arm.directory}/{stem}: expected rollouts 0..99")
            if group.duplicated(["rollout", "i", "j"]).any():
                raise ValueError(f"{arm.directory}/{stem}: repeated contact within rollout")
            seed_counts = group.groupby("rollout").is_seed.sum()
            if not (seed_counts == (0 if arm.seeds is None else 1)).all():
                raise ValueError(f"{arm.directory}/{stem}: unexpected seed count")
            resolved_rows = group[resolved[group.i.to_numpy()] & resolved[group.j.to_numpy()]]
            for range_name, min_sep in (("all", 6), ("long", 24)):
                keep = psep >= min_sep
                ii, jj = pi[keep], pj[keep]
                n_true = int(truth[ii, jj].sum())
                if n_true <= 0:
                    raise ValueError(f"{stem}/{range_name}: no true contacts")
                contacts = resolved_rows[(resolved_rows.j - resolved_rows.i) >= min_sep]
                union = contacts.drop_duplicates(["i", "j"])
                union_true = int(truth[union.i.to_numpy(), union.j.to_numpy()].sum())
                n_unvoted = len(ii) - len(union)
                coverage_rows.append(dict(
                    dataset=dataset, stem=stem, arm=arm.directory, range=range_name,
                    L=length, n_rollouts=len(rollout_ids), n_true=n_true,
                    n_candidates=len(ii), n_voted=len(union), n_unvoted=n_unvoted,
                    n_unvoted_negative=n_unvoted - (n_true - union_true),
                    n_union_true=union_true, union_recall=union_true / n_true,
                ))
                for mode, exclude_seed in (("full", False), ("continuation", True)):
                    scores = rollout_scores(contacts, truth, n_true, rollout_ids, exclude_seed)
                    oracle_rows.append(dict(
                        dataset=dataset, stem=stem, arm=arm.directory, range=range_name,
                        mode=mode, L=length, n_true=n_true, n_rollouts=len(rollout_ids),
                        n_short_rollouts=int((scores.n_emitted < n_true).sum()),
                        n_empty_rollouts=int((scores.n_emitted == 0).sum()),
                        oracle_legacy=float(scores.legacy_precision.max()),
                        oracle_fixed_R=float(scores.fixed_R.max()),
                        mean_legacy=float(scores.legacy_precision.mean()),
                        mean_fixed_R=float(scores.fixed_R.mean()),
                    ))
        print(f"[audit] finished {arm.directory}", flush=True)
    return pd.DataFrame(oracle_rows), pd.DataFrame(coverage_rows)


def oracle_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare seeded and unseeded oracles on exactly matching proteins."""
    rows = []
    for (range_name, mode), group in frame.groupby(["range", "mode"]):
        for metric in ("oracle_legacy", "oracle_fixed_R"):
            values = group.pivot(index=["dataset", "stem"], columns="arm", values=metric)
            for arm in values.columns:
                if arm == "iid":
                    continue
                delta = (values[arm] - values.iid).to_numpy()
                rows.append(dict(range=range_name, mode=mode, metric=metric, arm=arm,
                                 reference="iid", arm_mean=float(values[arm].mean()),
                                 reference_mean=float(values.iid.mean()), **bootstrap(delta)))
    return pd.DataFrame(rows)


def cluster_comparisons(data: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit saved cluster candidates, including the pooled map as a fallback.

    The augmented oracle adds an already-available candidate to each pool; this
    is an upper bound for that particular candidate pool, not for downstream
    structure quality. The augmented geometric selector remains reference-free
    except that all saved input sets were cut using ground-truth R.
    """
    clusters = pd.read_csv(data / "exp254_cluster_per_protein.csv")
    consistency = pd.read_csv(data / "exp254_cluster_consistency.csv.gz")
    rows = []
    for record in clusters.to_dict("records"):
        for k in (2, 3, 5, 10, 20):
            base = record["single"]
            oracle = record[f"kmeans_oracle@{k}"]
            for name, value, reference in (
                ("cluster_oracle_vs_pooled", oracle, base),
                ("cluster_oracle_with_pooled_vs_pooled", max(oracle, base), base),
                ("cluster_oracle_vs_random_oracle", oracle, record[f"random_oracle@{k}"]),
            ):
                rows.append(dict(dataset=record["dataset"], stem=record["stem"], K=k,
                                 comparison=name, value=value, reference=reference,
                                 delta=value - reference))
    for (dataset, stem), group in consistency.groupby(["dataset", "stem"], sort=True):
        singles = group[group.kind == "single"]
        if len(singles) != 1:
            raise ValueError(f"{stem}: expected one pooled candidate")
        single = singles.iloc[0]
        for k in (5, 10):
            candidates = group[group.kind == f"kmeans{k}"]
            if len(candidates) != k:
                raise ValueError(f"{stem}: expected {k} saved clusters")
            selected = candidates.loc[candidates.contact_excess.idxmin(), "precision"]
            # Put the pooled map first so an exact tie keeps the baseline.
            augmented = pd.concat([singles, candidates])
            with_pooled = augmented.loc[augmented.contact_excess.idxmin(), "precision"]
            for name, value, reference in (
                ("geometric_vs_blind", selected, candidates.precision.mean()),
                ("geometric_vs_pooled", selected, single.precision),
                ("geometric_with_pooled_vs_pooled", with_pooled, single.precision),
            ):
                rows.append(dict(dataset=dataset, stem=stem, K=k, comparison=name,
                                 value=value, reference=reference, delta=value - reference))
    frame = pd.DataFrame(rows)
    summary = [dict(K=k, comparison=name, value_mean=float(group.value.mean()),
                    reference_mean=float(group.reference.mean()),
                    **bootstrap(group.delta.to_numpy()))
               for (k, name), group in frame.groupby(["K", "comparison"])]
    return frame, pd.DataFrame(summary)


def coverage_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Report pair-weighted counts and protein-weighted union recall separately."""
    rows = []
    count_columns = ["n_candidates", "n_voted", "n_unvoted", "n_unvoted_negative",
                     "n_union_true", "n_true"]
    for (arm, range_name), group in frame.groupby(["arm", "range"]):
        totals = {column: int(group[column].sum()) for column in count_columns}
        rows.append(dict(arm=arm, range=range_name, **totals,
                         unvoted_negative_fraction=totals["n_unvoted_negative"] / totals["n_candidates"],
                         **bootstrap(group.union_recall.to_numpy())))
    return pd.DataFrame(rows)


def main() -> int:
    """Write new audit CSVs without changing the historical metric tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    oracle, coverage = audit_rollouts(args.run)
    cluster, cluster_summary = cluster_comparisons(args.data)
    outputs = {
        "oracle_per_protein": oracle,
        "oracle_summary": oracle_summary(oracle),
        "coverage_per_protein": coverage,
        "coverage_summary": coverage_summary(coverage),
        "cluster_per_protein": cluster,
        "cluster_summary": cluster_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.out / f"exp254_audit_{name}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
