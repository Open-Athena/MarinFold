# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit selective recovery after a complete conditioning run, without new inference.

Canonical outputs are never modified. Original saved failure groups replace only
corresponding false-large groups in a diagnostic replay. All 97 proteins and the
same withheld universes remain in that comparison. A separate practical contrast
excludes every protein produced by an explicitly identified resumed worker. This
is a selected-cohort diagnostic, not a replacement for the full-cohort primary.

Original failure files reveal cap incidence only inside observed failing groups;
other first-attempt groups were not all saved. They cannot supply a population
first-attempt cap rate. Same seeds do not guarantee identical numerical draws.
"""

import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_conditioning import (
    ARMS,
    DERIVED_ARMS,
    PRIMARY_REFERENCE,
    candidate_universes,
    margin_status,
    paired_interval,
    precision_at_r,
)
from verify_conditioning import checked_payloads, position_frames, reparse_group, sha256


def validate_analysis(
    plan_path: Path, analysis: Path
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Require completed, matching canonical analysis before computing sensitivity."""
    plan = json.loads(plan_path.read_text())
    conclusion = json.loads((analysis / "conditioning_conclusion.json").read_text())
    if conclusion["plan_sha256"] != sha256(plan_path.read_bytes()):
        raise ValueError("canonical analysis belongs to another input plan")
    if (
        conclusion["n_targets"] != len(plan["targets"])
        or conclusion["n_repeats"] != plan["n_repeats"]
    ):
        raise ValueError("canonical analysis is not complete for the frozen population")
    repeats = pd.read_csv(analysis / "conditioning_per_repeat.csv")
    proteins = pd.read_csv(analysis / "conditioning_per_protein.csv")
    identity = ["dataset", "stem", "repeat", "scope", "arm"]
    expected = {
        (target["dataset"], target["stem"], repeat, scope, arm)
        for target in plan["targets"]
        for repeat in range(plan["n_repeats"])
        for scope in ("full_pipeline", "withheld_continuation")
        for arm in ARMS + DERIVED_ARMS
    }
    if (
        repeats.duplicated(identity).any()
        or set(map(tuple, repeats[identity].to_numpy())) != expected
    ):
        raise ValueError(
            "canonical per-repeat table has missing/extra/duplicate samples"
        )
    if not np.isfinite(repeats.precision).all():
        raise ValueError("canonical per-repeat precision contains nonfinite values")
    expected_proteins = {
        (target["dataset"], target["stem"]) for target in plan["targets"]
    }
    if set(map(tuple, proteins[["dataset", "stem"]].to_numpy())) != expected_proteins:
        raise ValueError("canonical protein table has a different target population")
    return plan, repeats, proteins


def recovery_cohort(
    plan: dict,
    plan_sha: str,
    run: Path,
    original_worker: str,
    resumed_workers: set[str],
) -> pd.DataFrame:
    """Identify all regenerated units from checksum-verified timing provenance."""
    if original_worker in resumed_workers or not resumed_workers:
        raise ValueError("original and resumed worker identities must be distinct")
    allowed = {original_worker} | resumed_workers
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in allowed):
        raise ValueError("worker identities must be complete SHA-256 digests")
    rows = []
    for target in plan["targets"]:
        stem = target["stem"]
        marker = json.loads((run / "units" / f"{stem}.complete.json").read_text())
        if marker["stem"] != stem or marker["plan_sha256"] != plan_sha:
            raise ValueError(f"{stem}: completion marker differs from frozen plan")
        payload = (run / "units" / f"{stem}.timings.csv").read_bytes()
        expected = marker["files"]["timings.csv"]
        if len(payload) != expected["bytes"] or sha256(payload) != expected["sha256"]:
            raise ValueError(f"{stem}: timing provenance checksum mismatch")
        # Reading via the filename is safe after its immutable downloaded payload
        # has been checked against the completed-unit manifest.
        timing = pd.read_csv(
            run / "units" / f"{stem}.timings.csv", dtype={"worker_sha256": str}
        )
        if (
            timing.worker_sha256.nunique() != 1
            or not (timing.plan_sha256 == plan_sha).all()
        ):
            raise ValueError(f"{stem}: inconsistent worker/plan identities")
        worker = timing.worker_sha256.iloc[0]
        if worker not in allowed:
            raise ValueError(
                f"{stem}: worker is not among the explicitly audited versions"
            )
        rows.append(
            {
                "dataset": target["dataset"],
                "stem": stem,
                "L": target["L"],
                "worker_sha256": worker,
                "resumed_worker": worker in resumed_workers,
            }
        )
    return pd.DataFrame(rows)


def replay_failures(
    plan: dict, run: Path, failures: Path, repeats: pd.DataFrame, plan_sha: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replace failed false-large group scores using original raw completions."""
    selected = repeats[repeats.scope == "withheld_continuation"].copy()
    targets = {target["stem"]: target for target in plan["targets"]}
    reports = []
    rollout_reports = []
    paths = sorted(failures.glob("*.json.gz"))
    if not paths:
        raise ValueError("no preserved original failure groups were found")
    for path in paths:
        match = re.fullmatch(r"(.+)-r(\d+)__(false_large)\.json\.gz", path.name)
        if match is None:
            raise ValueError(
                f"unexpected failure group; this replay is scoped to false_large: {path.name}"
            )
        stem, repeat_text, arm = match.groups()
        repeat = int(repeat_text)
        if stem not in targets or not 0 <= repeat < plan["n_repeats"]:
            raise ValueError(f"failure group is outside the frozen plan: {path.name}")
        target = targets[stem]
        _, payloads = checked_payloads(run, stem, plan_sha)
        current_raw = json.loads(gzip.decompress(payloads["raw.json.gz"]))
        key = f"r{repeat}__{arm}"
        original_bytes = path.read_bytes()
        original = json.loads(gzip.decompress(original_bytes))
        current = current_raw[key]
        frames = position_frames(target, plan["n_repeats"], plan["n_rollouts"])[repeat]
        context = target["contexts"][repeat][arm]
        budget = 6 * target["L"] + 128
        old_votes, old_counts = reparse_group(
            original,
            frames,
            target["L"],
            context,
            budget,
            f"original/{stem}/{key}",
            accept_budget_termination=True,
        )
        new_votes, new_counts = reparse_group(
            current,
            frames,
            target["L"],
            context,
            budget,
            f"resumed/{stem}/{key}",
            accept_budget_termination=True,
        )
        if old_counts["unfinished_rollouts"] == 0:
            raise ValueError(
                f"{path.name}: preserved group contains no original budget stops"
            )
        with np.load(run / "units" / f"{stem}.npz", allow_pickle=False) as archive:
            if not np.array_equal(archive[f"{key}__votes"], new_votes):
                raise ValueError(
                    f"{stem}/{key}: resumed raw votes do not match canonical NPZ"
                )
        pi, pj, truth, withheld = candidate_universes(
            target, target["contexts"][repeat]
        )
        old_precision, _, n_true = precision_at_r(
            old_votes, truth, pi[withheld], pj[withheld]
        )
        new_precision, _, _ = precision_at_r(
            new_votes, truth, pi[withheld], pj[withheld]
        )
        where = (
            (selected.dataset == target["dataset"])
            & (selected.stem == stem)
            & (selected["repeat"] == repeat)
            & (selected.arm == arm)
        )
        canonical = selected.loc[where]
        if len(canonical) != 1 or not np.isclose(
            canonical.precision.iloc[0], new_precision, rtol=0, atol=1e-12
        ):
            raise ValueError(
                f"{stem}/{key}: canonical accuracy differs from independently scored resumed votes"
            )
        if canonical.n_true.iloc[0] != n_true:
            raise ValueError(f"{stem}/{key}: canonical withheld denominator differs")
        selected.loc[where, "precision"] = old_precision
        for index, (before, after) in enumerate(zip(original, current, strict=True)):
            rollout_reports.append(
                {
                    "stem": stem,
                    "repeat": repeat,
                    "arm": arm,
                    "rollout": index,
                    "original_file": path.name,
                    "original_text_sha256": sha256(before["text"].encode()),
                    "resumed_text_sha256": sha256(after["text"].encode()),
                    "original_finish_reason": before["finish_reason"],
                    "resumed_finish_reason": after["finish_reason"],
                    "original_tokens": before["tokens"],
                    "resumed_tokens": after["tokens"],
                    "identical_text": before["text"] == after["text"],
                    "identical_contacts": before["contacts"] == after["contacts"],
                }
            )
        reports.append(
            {
                "dataset": target["dataset"],
                "stem": stem,
                "repeat": repeat,
                "arm": arm,
                "n_rollouts": len(original),
                "original_budget_stops": old_counts["unfinished_rollouts"],
                "resumed_budget_stops": new_counts["unfinished_rollouts"],
                "identical_texts": sum(
                    a["text"] == b["text"]
                    for a, b in zip(original, current, strict=True)
                ),
                "identical_contact_lists": sum(
                    a["contacts"] == b["contacts"]
                    for a, b in zip(original, current, strict=True)
                ),
                "identical_records": sum(
                    a == b for a, b in zip(original, current, strict=True)
                ),
                "original_withheld_precision": old_precision,
                "resumed_withheld_precision": new_precision,
                "replay_minus_resumed": old_precision - new_precision,
                "n_remaining_true": n_true,
                "original_failure_sha256": sha256(original_bytes),
            }
        )
    return selected, pd.DataFrame(reports), pd.DataFrame(rollout_reports)


def mechanistic_summary(
    canonical: pd.DataFrame, replay: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare original-failure replay with canonical all-protein paired effects."""
    keys = ["dataset", "stem"]
    original = (
        canonical[canonical.scope == "withheld_continuation"]
        .groupby(keys + ["arm"])
        .precision.mean()
        .unstack("arm")
    )
    alternative = replay.groupby(keys + ["arm"]).precision.mean().unstack("arm")
    if not original.index.equals(alternative.index):
        raise ValueError("replay changed the matched protein population")
    stable_arms = [arm for arm in original.columns if arm != "false_large"]
    if not original[stable_arms].equals(alternative[stable_arms]):
        raise ValueError("failure replay unexpectedly changed another arm")
    per_protein = original[["iid", "true_large", "false_large"]].rename(
        columns={"false_large": "canonical_false_large"}
    )
    per_protein["replay_false_large"] = alternative.false_large
    per_protein["replay_minus_canonical_false_large"] = (
        alternative.false_large - original.false_large
    )
    rows = []
    for version, values in (
        ("canonical", original),
        ("original_failure_replay", alternative),
    ):
        for arm, reference in (("false_large", "iid"), ("true_large", "false_large")):
            rows.append(
                dict(
                    analysis=version,
                    scope="withheld_continuation",
                    arm=arm,
                    reference=reference,
                    **paired_interval((values[arm] - values[reference]).to_numpy()),
                )
            )
    for arm, change in (
        ("false_large_vs_iid", alternative.false_large - original.false_large),
        ("true_large_vs_false_large", original.false_large - alternative.false_large),
    ):
        rows.append(
            dict(
                analysis="replay_minus_canonical_effect",
                scope="withheld_continuation",
                arm=arm,
                reference="canonical_effect",
                **paired_interval(change.to_numpy()),
            )
        )
    return per_protein.reset_index(), pd.DataFrame(rows)


def practical_cohort_summary(
    proteins: pd.DataFrame, cohort: pd.DataFrame, margin: float
) -> pd.DataFrame:
    """Report the original primary and a clearly selected-cohort diagnostic."""
    values = proteins[proteins.scope == "full_pipeline"].pivot(
        index=["dataset", "stem"], columns="arm", values="precision"
    )
    membership = cohort.set_index(["dataset", "stem"]).reindex(values.index)
    if membership.resumed_worker.isna().any():
        raise ValueError("missing worker provenance for a primary-score protein")
    rows = []
    for scope, keep in (
        ("canonical_all_proteins", np.ones(len(values), dtype=bool)),
        (
            "diagnostic_excluding_resumed_workers",
            ~membership.resumed_worker.to_numpy(dtype=bool),
        ),
    ):
        subset = values.loc[keep]
        if subset.empty:
            rows.append(
                {
                    "cohort": scope,
                    "arm": "pred_large",
                    "reference": PRIMARY_REFERENCE,
                    "n_proteins": 0,
                    "mean_delta": None,
                    "ci_low": None,
                    "ci_high": None,
                    "margin_status": "not_estimable_empty_cohort",
                    "practical_margin": margin,
                }
            )
            continue
        interval = paired_interval(
            (subset.pred_large - subset[PRIMARY_REFERENCE]).to_numpy()
        )
        rows.append(
            dict(
                cohort=scope,
                arm="pred_large",
                reference=PRIMARY_REFERENCE,
                **interval,
                practical_margin=margin,
                margin_status=margin_status(
                    interval["ci_low"], interval["ci_high"], margin
                ),
                median_L=float(membership.loc[keep, "L"].median()),
            )
        )
    return pd.DataFrame(rows)


def analyze_recovery(
    plan_path: Path,
    run: Path,
    analysis: Path,
    failures_path: Path,
    out: Path,
    original_worker: str,
    resumed_workers: set[str],
) -> dict:
    """Write bounded recovery diagnostics without changing canonical artifacts."""
    plan, repeats, proteins = validate_analysis(plan_path, analysis)
    plan_sha = sha256(plan_path.read_bytes())
    cohort = recovery_cohort(plan, plan_sha, run, original_worker, resumed_workers)
    replay, failures, failure_rollouts = replay_failures(
        plan, run, failures_path, repeats, plan_sha
    )
    per_protein, mechanism = mechanistic_summary(repeats, replay)
    practical = practical_cohort_summary(proteins, cohort, plan["practical_margin"])
    report = {
        "plan_sha256": plan_sha,
        "scope": "recovery_sensitivity_only; canonical all-protein primary remains unchanged",
        "n_planned_proteins": len(plan["targets"]),
        "n_original_failure_groups": len(failures),
        "observed_original_failure_group_rollouts": int(failures.n_rollouts.sum()),
        "original_caps_in_observed_failure_groups": int(
            failures.original_budget_stops.sum()
        ),
        "resumed_caps_in_same_groups": int(failures.resumed_budget_stops.sum()),
        "n_resumed_worker_proteins": int(cohort.resumed_worker.sum()),
        "original_worker_sha256": original_worker,
        "resumed_worker_sha256": sorted(resumed_workers),
        "mechanism": mechanism.to_dict("records"),
        "practical_cohort_diagnostic": practical.astype(object)
        .where(practical.notna(), None)
        .to_dict("records"),
        "limitations": [
            "Initial failure incidence is observed only in preserved trigger groups, not all first-attempt rollouts.",
            "Original failure records lack embedded worker/plan/timing provenance; filenames, completion metadata, frozen-frame parsing, and recorded file/text hashes are checked, but historical execution identity cannot be independently recovered from those records.",
            "Per-rollout text hashes identify every original trigger-group completion; full original texts remain in the referenced compressed input files.",
            "Replay changes only false_large on the original shared withheld universe; all other arm scores remain canonical.",
            "Other pre-abort original groups were not all retained, so replay is not a reconstruction of every initial attempt.",
            "Excluding resumed-worker proteins selects a cohort through completion/recovery and may change its length/difficulty mix; it is not a replacement primary or an estimate of a worker-version effect.",
            "Intervals average replicates within protein before bootstrapping proteins; secondary sensitivity comparisons are pointwise and conditional on saved draws.",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("failures", failures),
        ("failure_rollouts", failure_rollouts),
        ("cohort", cohort),
        ("mechanistic_per_protein", per_protein),
        ("mechanistic_summary", mechanism),
        ("practical_cohort_summary", practical),
    ):
        frame.to_csv(out / f"conditioning_recovery_{name}.csv", index=False)
    (out / "conditioning_recovery_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> int:
    """Run only after complete canonical analysis is available."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--original-worker-sha256", required=True)
    parser.add_argument("--resumed-worker-sha256", required=True, action="append")
    args = parser.parse_args()
    report = analyze_recovery(
        args.plan,
        args.run,
        args.analysis,
        args.failures,
        args.out,
        args.original_worker_sha256,
        set(args.resumed_worker_sha256),
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
