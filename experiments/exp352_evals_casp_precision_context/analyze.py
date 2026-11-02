"""Rescore frozen eval-val votes with pyconfind and CASP distance contacts.

Run ``uv run python analyze.py``. Downloads only this experiment's public input
bundle when absent; no model inference, historical CASP targets or eval-test.
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfFileSystem
from prepare_inputs import CACHE, DATA, EXP277, EXPERIMENTS, PUBLIC, digest

sys.path.insert(0, str(EXPERIMENTS / "exp89_evals_contacts_v1_model_on_eval_set"))
from compute_metrics import (
    RANGES,
    metric_rows,
    resolved_pairs,
    true_matrix,
)

CUTS = ("L/5", "L/2", "L")


def tie_statistics(scores: np.ndarray, labels: np.ndarray, top: int) -> dict:
    """Return expected precision and exact extremes over a boundary vote tie."""
    threshold = np.sort(scores)[-top]
    above = scores > threshold
    tied = scores == threshold
    remaining = top - int(above.sum())
    true_above = int(labels[above].sum())
    true_tied = int(labels[tied].sum())
    n_tied = int(tied.sum())
    return {
        "precision_tie_expected": (true_above + remaining * true_tied / n_tied) / top,
        "precision_tie_min": (true_above + max(0, remaining - (n_tied - true_tied)))
        / top,
        "precision_tie_max": (true_above + min(remaining, true_tied)) / top,
        "boundary_votes": float(threshold),
        "boundary_tie_size": n_tied,
        "zero_vote_fraction": max(0, top - int((scores > 0).sum())) / top,
    }


def protein_rows(item: dict) -> list[dict]:
    """Score a protein with both ground-truth definitions and audited universes."""
    rec = item["truth"]
    length = rec["L"]
    score = np.zeros((length, length), dtype=float)
    votes = np.asarray(item["votes"], dtype=int)
    score[votes[:, 0], votes[:, 1]] = votes[:, 2]
    score += score.T
    pytruth = true_matrix(length, rec["contacts"])
    coords = np.full((length, 3), np.nan)
    xyz = np.asarray(item["xyz"])
    coords[xyz[:, 0].astype(int)] = xyz[:, 1:]
    distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    cb_resolved = np.flatnonzero(np.isfinite(coords).all(axis=1))
    common = np.intersect1d(rec["resolved"], cb_resolved)
    definitions = (
        ("pyconfind", pytruth, np.asarray(rec["resolved"])),
        ("cb8", distances < 8.0, cb_resolved),
        ("cb8_common", distances < 8.0, common),
    )
    rows = []
    for definition, truth, resolved in definitions:
        pi, pj, sep = resolved_pairs(resolved)
        metrics = metric_rows(score, truth, pi, pj, sep, length, with_precision=True)
        for row in metrics:
            lo, hi = RANGES[row["range"]]
            mask = (sep >= lo) & ((sep <= hi) if hi is not None else True)
            s, g = score[pi[mask], pj[mask]], truth[pi[mask], pj[mask]]
            result = dict(
                stem=rec["stem"],
                definition=definition,
                L=length,
                n_resolved=len(resolved),
                is_viral=item["is_viral"],
                **row,
            )
            if row["cut"] in CUTS:
                assert row["n_top"] > 0
                result.update(tie_statistics(s, g, row["n_top"]))
                result["random_precision"] = row["n_true"] / row["n_candidate"]
                result["oracle_precision"] = (
                    min(row["n_true"], row["n_top"]) / row["n_top"]
                )
                # <= vs < is explicitly checked because CASP papers vary.
                result["pairs_at_exactly_8"] = int(
                    (distances[pi[mask], pj[mask]] == 8.0).sum()
                )
            rows.append(result)
    return rows


def bootstrap(values: np.ndarray, draws: int, seed: int = 352) -> tuple[float, float]:
    """Percentile interval for a protein-macro mean, resampling proteins."""
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def summarize(
    frame: pd.DataFrame, cohorts: dict[str, set[str]], draws: int
) -> pd.DataFrame:
    """Produce macro means and diagnostics for prespecified cohorts and cuts."""
    rows = []
    for cohort, stems in cohorts.items():
        subset = frame[frame.stem.isin(stems) & frame.cut.isin(CUTS)]
        for (definition, span, cut), group in subset.groupby(
            ["definition", "range", "cut"], sort=True
        ):
            values = group.precision.to_numpy()
            assert np.isfinite(values).all() and group.stem.nunique() == len(stems)
            low, high = bootstrap(values, draws)
            rows.append(
                {
                    "cohort": cohort,
                    "definition": definition,
                    "range": span,
                    "cut": cut,
                    "n": len(values),
                    "mean": values.mean(),
                    "ci_low": low,
                    "ci_high": high,
                    "median": np.median(values),
                    "q25": np.quantile(values, 0.25),
                    "q75": np.quantile(values, 0.75),
                    "random_precision": group.random_precision.mean(),
                    "oracle_precision": group.oracle_precision.mean(),
                    "tie_expected": group.precision_tie_expected.mean(),
                    "tie_min": group.precision_tie_min.mean(),
                    "tie_max": group.precision_tie_max.mean(),
                    "zero_vote_fraction": group.zero_vote_fraction.mean(),
                    "n_below_20pct": int((values < 0.2).sum()),
                    "n_above_50pct": int((values >= 0.5).sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Validate the archived control and save all metrics and cohort summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=CACHE / "inputs.json.gz")
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    if not args.inputs.exists():
        args.inputs.parent.mkdir(parents=True, exist_ok=True)
        args.inputs.write_bytes(
            HfFileSystem(token=False).read_bytes(
                PUBLIC.removeprefix("hf://") + "/inputs.json.gz"
            )
        )
    raw = args.inputs.read_bytes()
    bundle = json.loads(gzip.decompress(raw))
    assert (
        bundle["model"] == "contacts-v1-exp277-m2-p06-full-epoch-1.5B"
        and bundle["step"] == 266344
    )
    items = bundle["proteins"]
    stems = {p["truth"]["stem"] for p in items}
    assert len(items) == len(stems) == 97
    allowed = pd.read_csv(
        EXPERIMENTS / "exp245_evals_foldbench_held_out_monomers/data/eval_sets.csv"
    )
    assert stems == set(allowed.loc[allowed.eval_set.eq("eval-val"), "stem"])
    frame = pd.DataFrame(row for item in items for row in protein_rows(item))
    expected = pd.read_csv(EXP277 / "contact_precision_all.csv")
    expected = expected[
        expected.dataset.eq("foldbench_monomer") & expected.stem.isin(stems)
    ]
    control = frame[frame.definition.eq("pyconfind")].merge(
        expected,
        on=["stem", "range", "cut"],
        validate="one_to_one",
        suffixes=("", "_source"),
    )
    assert len(control) == 97 * 20
    assert np.allclose(
        control.precision, control.precision_source, atol=1e-12, rtol=0, equal_nan=True
    )
    for column in ("n_candidate", "n_true", "n_top"):
        assert (control[column] == control[column + "_source"]).all()
    audit_rows = []
    ambiguous = set()
    for item in items:
        audit = item["coordinate_audit"]
        stem = item["truth"]["stem"]
        mismatch = bool(audit["polymer_not_frozen"] or audit["frozen_not_polymer"])
        if mismatch:
            ambiguous.add(stem)
        audit_rows.append(
            {
                "stem": stem,
                "L": item["truth"]["L"],
                "n_frozen": len(item["truth"]["resolved"]),
                "n_cb": len(item["xyz"]),
                "missing_cb_ca": json.dumps(audit["missing_atoms"]),
                "polymer_not_frozen": json.dumps(audit["polymer_not_frozen"]),
                "frozen_not_polymer": json.dumps(audit["frozen_not_polymer"]),
                "index_universe_differs": mismatch,
                "score_sha256": item["score_sha256"],
                "cif_sha256": item["cif_sha256"],
            }
        )
    pd.DataFrame(audit_rows).to_csv(DATA / "coordinate_audit.csv", index=False)
    depths = pd.read_csv(
        EXPERIMENTS / "exp260_evals_msa_depth_stratified/data/msa_depth.csv"
    )
    depths = depths[depths.stem.isin(stems)].drop_duplicates("stem")
    assert depths.stem.nunique() == 97
    cohorts = {
        "eval-val": stems,
        "viral": {p["truth"]["stem"] for p in items if p["is_viral"]},
        "nonviral": {p["truth"]["stem"] for p in items if not p["is_viral"]},
        "unambiguous_index_universe": stems - ambiguous,
    }
    for low, high, name in (
        (0, 10, "MSA_lt10"),
        (10, 100, "MSA_10_99"),
        (100, 1000, "MSA_100_999"),
        (1000, np.inf, "MSA_ge1000"),
    ):
        members = set(
            depths.loc[depths.n_seqs.ge(low) & depths.n_seqs.lt(high), "stem"]
        )
        if members:
            cohorts[name] = members
    frame.to_csv(
        DATA / "per_protein.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    summary = summarize(frame, cohorts, args.bootstrap_draws)
    summary.to_csv(DATA / "summary.csv", index=False)
    metadata = {key: value for key, value in bundle.items() if key != "proteins"}
    metadata.update(
        inputs_sha256=digest(raw),
        n_proteins=97,
        n_rollouts=9700,
        unfinished_rollouts=0,
        control_rows_checked=len(control),
        maximum_control_error=float(
            np.nanmax(abs(control.precision - control.precision_source))
        ),
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=352,
        altered_index_universes=sorted(ambiguous),
        low_msa_eval_val_n=int((depths.n_seqs < 10).sum()),
        cohorts={k: sorted(v) for k, v in cohorts.items()},
        contact_definition="CB distance < 8 Angstrom; CA for glycine only; first deposited model; highest occupancy altloc",
        rank="descending rollout votes, ties ascending (i,j), zero-vote pairs included",
        length="full input sequence, floor(L/divisor), candidates require both atoms observed",
        inference_rerun=False,
        eval_test_read=False,
        exp89_metric_sha256=digest(
            (
                EXPERIMENTS
                / "exp89_evals_contacts_v1_model_on_eval_set/compute_metrics.py"
            ).read_bytes()
        ),
    )
    (DATA / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        summary.query("cohort == 'eval-val' and definition != 'cb8_common'")[
            [
                "definition",
                "range",
                "cut",
                "n",
                "mean",
                "ci_low",
                "ci_high",
                "random_precision",
                "tie_expected",
            ]
        ].to_string(index=False)
    )
    print("Index-universe sensitivity:", sorted(ambiguous))
    print("Low MSA eval-val:", metadata["low_msa_eval_val_n"])


if __name__ == "__main__":
    main()
