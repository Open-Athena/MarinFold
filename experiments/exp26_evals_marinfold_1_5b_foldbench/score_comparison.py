# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Merge MarinFold + Protenix scores into a single 3-way comparison CSV.

Inputs:
  - ``data/marinfold_scores.csv`` (produced by ``score_marinfold.py``):
    one row per protein, ``method == "marinfold_1b"``.
  - ``protenix_data/.../scores.csv`` (downloaded by ``fetch_protenix_data.py``):
    two rows per protein, ``mode ∈ {single_seq, msa}`` — exp12's full
    schema, but for the merge we keep only the columns that overlap
    with MarinFold (the distogram-side metrics).

Output:
  - ``data/scores.csv`` — 300 rows when all 100 proteins are scored:
    100 × ``marinfold_1b`` + 100 × ``protenix_single_seq`` + 100 ×
    ``protenix_msa``. Column schema matches ``score_marinfold.py``.
  - ``data/scores_summary.csv`` — per-method aggregates
    (mean / median / min / max) for each headline metric.
  - ``data/hypothesis_verdict.json`` — machine-readable verdict details.

Hypothesis verdict (from issue #20 success criteria):

    The hypothesis is supported iff ``marinfold_1b`` sits strictly
    between ``protenix_single_seq`` and ``protenix_msa`` on the
    aggregate (mean) of at least 2 of the 3 headline metrics:

      - ``lddt_distogram_cb`` (higher is better)
      - ``mae_distogram_cb_angstrom`` (lower is better)
      - ``drmsd_distogram_cb_angstrom`` (lower is better)
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


_OUR_COLUMNS = [
    "pdb_id", "chain_id", "method", "n_residues",
    "mae_distogram_cb_angstrom", "drmsd_distogram_cb_angstrom",
    "n_mae_distogram_pairs",
    "mae_distogram_cb_contact_angstrom", "drmsd_distogram_cb_contact_angstrom",
    "n_mae_distogram_contact_pairs",
    "prec_short_L", "prec_short_L_2", "prec_short_L_5",
    "prec_medium_L", "prec_medium_L_2", "prec_medium_L_5",
    "prec_long_L", "prec_long_L_2", "prec_long_L_5",
    "n_short_contacts", "n_medium_contacts", "n_long_contacts",
    "lddt_distogram_cb", "lddt_distogram_cb_soft",
]

# Headline metrics for the hypothesis check (issue #20 success criteria).
# Direction is "lower is better" for MAE/dRMSD, "higher is better" for LDDT.
_HEADLINE = (
    ("lddt_distogram_cb", "higher"),
    ("mae_distogram_cb_angstrom", "lower"),
    ("drmsd_distogram_cb_angstrom", "lower"),
)


def _parse_float(x: str) -> float:
    if x in ("", "nan", "NaN"):
        return float("nan")
    return float(x)


def _parse_int(x: str) -> int:
    if x in ("", "nan", "NaN"):
        return 0
    return int(x)


def _format_summary_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def _load_marinfold_rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    if rows[0].get("method") != "marinfold_1b":
        # Cheap sanity check on the input.
        raise ValueError(f"{csv_path}: expected method=marinfold_1b, got {rows[0].get('method')!r}")
    return rows


def _load_protenix_rows(csv_path: Path) -> list[dict]:
    """Read exp12's scores.csv and project it down to our column set.

    exp12's CSV has ``mode ∈ {single_seq, msa}``; we map that to
    ``method ∈ {protenix_single_seq, protenix_msa}`` here. Columns that
    only exist on the Protenix side (structure metrics, ranking_score,
    seed, sample_idx, selected_as_best, the structure LDDT variants)
    are dropped — they don't apply to MarinFold so they have no place
    in the 3-way comparison CSV.
    """
    with csv_path.open() as f:
        protenix_rows = list(csv.DictReader(f))
    out: list[dict] = []
    for r in protenix_rows:
        mode = r["mode"]
        if mode not in ("single_seq", "msa"):
            continue
        out.append({
            "pdb_id": r["pdb_id"],
            "chain_id": r["chain_id"],
            "method": f"protenix_{mode}",
            "n_residues": r["n_residues"],
            "mae_distogram_cb_angstrom": r["mae_distogram_cb_angstrom"],
            "drmsd_distogram_cb_angstrom": r["drmsd_distogram_cb_angstrom"],
            "n_mae_distogram_pairs": r["n_mae_distogram_pairs"],
            "mae_distogram_cb_contact_angstrom": r["mae_distogram_cb_contact_angstrom"],
            "drmsd_distogram_cb_contact_angstrom": r["drmsd_distogram_cb_contact_angstrom"],
            "n_mae_distogram_contact_pairs": r["n_mae_distogram_contact_pairs"],
            "prec_short_L": r["prec_short_L"],
            "prec_short_L_2": r["prec_short_L_2"],
            "prec_short_L_5": r["prec_short_L_5"],
            "prec_medium_L": r["prec_medium_L"],
            "prec_medium_L_2": r["prec_medium_L_2"],
            "prec_medium_L_5": r["prec_medium_L_5"],
            "prec_long_L": r["prec_long_L"],
            "prec_long_L_2": r["prec_long_L_2"],
            "prec_long_L_5": r["prec_long_L_5"],
            "n_short_contacts": r["n_short_contacts"],
            "n_medium_contacts": r["n_medium_contacts"],
            "n_long_contacts": r["n_long_contacts"],
            "lddt_distogram_cb": r["lddt_distogram_cb"],
            "lddt_distogram_cb_soft": r["lddt_distogram_cb_soft"],
        })
    return out


def _aggregate_for_method(rows: list[dict], method: str) -> dict[str, object]:
    """Mean / median / min / max for each headline metric, for one method.

    Only rows whose ``method`` matches are aggregated. NaN values are
    skipped — the smoke runs may have fewer than 100 proteins, and
    one or two might have all-zero contact classes (very short
    chains). Returns a flat dict keyed ``{metric}_{stat}``.
    """
    method_rows = [r for r in rows if r["method"] == method]
    out: dict[str, object] = {"method": method, "n_proteins": len(method_rows)}
    for col, _direction in _HEADLINE:
        values = [_parse_float(r[col]) for r in method_rows]
        clean = [v for v in values if v == v]  # drop NaN
        if not clean:
            out[f"{col}_mean"] = float("nan")
            out[f"{col}_median"] = float("nan")
            out[f"{col}_min"] = float("nan")
            out[f"{col}_max"] = float("nan")
            continue
        out[f"{col}_mean"] = mean(clean)
        out[f"{col}_median"] = median(clean)
        out[f"{col}_min"] = min(clean)
        out[f"{col}_max"] = max(clean)
    return out


def _hypothesis_verdict(summary_rows: list[dict]) -> dict[str, object]:
    """Decide whether 1B lands between protenix_single_seq and protenix_msa.

    Per issue #20: hypothesis is supported iff the aggregate mean of
    ``marinfold_1b`` is strictly between the two Protenix means on at
    least 2 of the 3 headline metrics, in the direction that says
    "1B beats single-seq but loses to MSA".
    """
    by_method = {r["method"]: r for r in summary_rows}
    if "marinfold_1b" not in by_method or "protenix_single_seq" not in by_method or "protenix_msa" not in by_method:
        return {
            "verdict": "incomplete",
            "n_metrics_supporting": 0,
            "metrics_supporting": [],
            "metrics_refuting": [],
            "details": ["missing one or more methods"],
        }

    supporting: list[str] = []
    refuting: list[str] = []
    details: list[str] = []
    for col, direction in _HEADLINE:
        m1b = by_method["marinfold_1b"][f"{col}_mean"]
        mss = by_method["protenix_single_seq"][f"{col}_mean"]
        mms = by_method["protenix_msa"][f"{col}_mean"]
        if any(v != v for v in (m1b, mss, mms)):  # NaN check
            details.append(f"{col}: NaN in means; skipped.")
            continue
        if direction == "higher":
            # higher = better. 1B beats SS iff m1b > mss; loses to MSA iff m1b < mms.
            ok = mss < m1b < mms
        else:
            # lower = better. 1B beats SS iff m1b < mss; loses to MSA iff m1b > mms.
            ok = mms < m1b < mss
        actual_order = ", ".join(
            f"{name}={value:.4f}"
            for name, value in sorted(
                (
                    ("marinfold_1b", m1b),
                    ("protenix_single_seq", mss),
                    ("protenix_msa", mms),
                ),
                key=lambda item: item[1],
                reverse=(direction == "higher"),
            )
        )
        details.append(
            f"{col}: {actual_order} -> {'support' if ok else 'refute'}"
        )
        (supporting if ok else refuting).append(col)
    verdict = "supported" if len(supporting) >= 2 else "not_supported"
    return {
        "verdict": verdict,
        "n_metrics_supporting": len(supporting),
        "metrics_supporting": supporting,
        "metrics_refuting": refuting,
        "details": details,
    }


def merge_and_summarize(
    *,
    marinfold_csv: Path,
    protenix_csv: Path,
    out_scores_csv: Path,
    out_summary_csv: Path,
    out_verdict_json: Path,
) -> dict[str, object]:
    """Top-level entry point — produce the combined CSV + summary CSV.

    Returns the hypothesis verdict dict (also written to JSON).
    """
    mf_rows = _load_marinfold_rows(marinfold_csv)
    px_rows = _load_protenix_rows(protenix_csv)

    all_rows = mf_rows + px_rows

    # Sort: protein-major, then method, deterministic.
    method_order = {"marinfold_1b": 0, "protenix_single_seq": 1, "protenix_msa": 2}
    all_rows.sort(key=lambda r: (r["pdb_id"], r["chain_id"], method_order.get(r["method"], 99)))

    out_scores_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_scores_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_OUR_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {out_scores_csv} ({len(all_rows)} rows).")

    summary = [
        _aggregate_for_method(all_rows, "marinfold_1b"),
        _aggregate_for_method(all_rows, "protenix_single_seq"),
        _aggregate_for_method(all_rows, "protenix_msa"),
    ]
    verdict = _hypothesis_verdict(summary)

    summary_cols = ["method", "n_proteins"] + [
        f"{col}_{stat}" for col, _ in _HEADLINE for stat in ("mean", "median", "min", "max")
    ]
    with out_summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_cols, extrasaction="ignore")
        writer.writeheader()
        for row in summary:
            row_out = {k: _format_summary_value(v) for k, v in row.items()}
            writer.writerow(row_out)
    print(f"Wrote {out_summary_csv}.")
    out_verdict_json.parent.mkdir(parents=True, exist_ok=True)
    out_verdict_json.write_text(json.dumps(verdict, indent=2) + "\n")
    print(f"Wrote {out_verdict_json}.")
    print(f"Hypothesis verdict: {verdict}")
    return verdict


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--marinfold-csv", type=Path, default=here / "data" / "marinfold_scores.csv",
    )
    parser.add_argument(
        "--protenix-csv",
        type=Path,
        default=here / "protenix_data" / "data" / "protenix-foldbench-monomers" / "scores.csv",
    )
    parser.add_argument("--scores", type=Path, default=here / "data" / "scores.csv")
    parser.add_argument("--summary", type=Path, default=here / "data" / "scores_summary.csv")
    parser.add_argument(
        "--verdict-json",
        type=Path,
        default=here / "data" / "hypothesis_verdict.json",
    )
    args = parser.parse_args()
    merge_and_summarize(
        marinfold_csv=args.marinfold_csv,
        protenix_csv=args.protenix_csv,
        out_scores_csv=args.scores,
        out_summary_csv=args.summary,
        out_verdict_json=args.verdict_json,
    )


if __name__ == "__main__":
    main()
