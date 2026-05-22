# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Merge MarinFold 1.5B + 1B + Protenix scores into a 4-way comparison CSV.

Inputs:
  - ``data/marinfold_scores.csv`` (produced by ``score_marinfold.py``
    on this experiment's 1.5B outputs): one row per protein,
    ``method == "marinfold_1_5b"``.
  - ``../exp20_evals_marinfold_1b_foldbench/data/scores.csv`` (already
    a 3-way exp20 CSV); we pull the ``marinfold_1b`` rows directly
    from it so the 1B numbers stay byte-identical to PR #21's
    published baseline — no re-scoring.
  - ``protenix_data/.../scores.csv`` (downloaded by ``fetch_protenix_data.py``):
    two rows per protein, ``mode ∈ {single_seq, msa}`` — exp12's full
    schema; we project to our column set.

Output:
  - ``data/scores.csv`` — 400 rows when all 100 proteins are scored:
    100 × each of ``marinfold_1_5b``, ``marinfold_1b``,
    ``protenix_single_seq``, ``protenix_msa``.
  - ``data/scores_summary.csv`` — per-method aggregates.
  - ``data/hypothesis_verdict.json`` — machine-readable verdict.

Hypothesis verdict (from issue #26 success criteria):

    The hypothesis is supported iff ``marinfold_1_5b`` beats
    ``marinfold_1b`` on the aggregate (mean) of at least 3 of the
    4 headline metrics:

      - ``lddt_distogram_cb`` (higher is better)
      - ``mae_distogram_cb_angstrom`` (lower is better)
      - ``drmsd_distogram_cb_angstrom`` (lower is better)
      - ``prec_long_L`` (higher is better)
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

# Headline metrics for the hypothesis check (issue #26 success criteria).
# Direction is "lower is better" for MAE/dRMSD, "higher is better" for
# LDDT and the long-range contact precision @ L.
_HEADLINE = (
    ("lddt_distogram_cb", "higher"),
    ("mae_distogram_cb_angstrom", "lower"),
    ("drmsd_distogram_cb_angstrom", "lower"),
    ("prec_long_L", "higher"),
)

_METHODS = (
    "marinfold_1_5b",
    "marinfold_1b",
    "protenix_single_seq",
    "protenix_msa",
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


def _load_marinfold_rows(csv_path: Path, expected_method: str) -> list[dict]:
    """Load this experiment's MarinFold scores CSV.

    ``expected_method`` is enforced to catch the easy mistake of
    pointing at the wrong file (e.g. 1B's CSV when running the 1.5B
    pipeline).
    """
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    if rows[0].get("method") != expected_method:
        raise ValueError(
            f"{csv_path}: expected method={expected_method!r}, "
            f"got {rows[0].get('method')!r}"
        )
    return rows


def _load_exp20_1b_rows(csv_path: Path) -> list[dict]:
    """Pull ``marinfold_1b`` rows out of exp20's 3-way scores.csv.

    exp20's scores.csv is itself a merged CSV with three methods
    interleaved. We just filter to the 1B rows so the numbers come
    in byte-identical to PR #21 (no re-scoring of 1B).
    """
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    one_b = [r for r in rows if r.get("method") == "marinfold_1b"]
    if not one_b:
        raise ValueError(
            f"{csv_path}: no rows with method=marinfold_1b "
            f"(found methods: {sorted({r['method'] for r in rows})})"
        )
    return one_b


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
    """Decide whether 1.5B beats 1B on >=3 of 4 headline metrics.

    Per issue #26: hypothesis is supported iff the aggregate (mean)
    of ``marinfold_1_5b`` is strictly better than ``marinfold_1b``
    on at least 3 of the 4 headline metrics (in each metric's
    "better" direction).
    """
    by_method = {r["method"]: r for r in summary_rows}
    if "marinfold_1_5b" not in by_method or "marinfold_1b" not in by_method:
        return {
            "verdict": "incomplete",
            "n_metrics_supporting": 0,
            "metrics_supporting": [],
            "metrics_refuting": [],
            "details": ["missing 1.5B or 1B summary row"],
        }

    supporting: list[str] = []
    refuting: list[str] = []
    details: list[str] = []
    for col, direction in _HEADLINE:
        m15 = by_method["marinfold_1_5b"][f"{col}_mean"]
        m1b = by_method["marinfold_1b"][f"{col}_mean"]
        if any(v != v for v in (m15, m1b)):  # NaN check
            details.append(f"{col}: NaN in means; skipped.")
            continue
        ok = (m15 > m1b) if direction == "higher" else (m15 < m1b)
        delta = m15 - m1b
        details.append(
            f"{col}: 1.5B={m15:.4f} vs 1B={m1b:.4f} (delta={delta:+.4f}, "
            f"direction={direction}) -> {'support' if ok else 'refute'}"
        )
        (supporting if ok else refuting).append(col)
    threshold = 3
    verdict = "supported" if len(supporting) >= threshold else "not_supported"
    return {
        "verdict": verdict,
        "threshold": f"{threshold}/{len(_HEADLINE)}",
        "n_metrics_supporting": len(supporting),
        "metrics_supporting": supporting,
        "metrics_refuting": refuting,
        "details": details,
    }


def merge_and_summarize(
    *,
    marinfold_1_5b_csv: Path,
    marinfold_1b_csv: Path,
    protenix_csv: Path,
    out_scores_csv: Path,
    out_summary_csv: Path,
    out_verdict_json: Path,
) -> dict[str, object]:
    """Produce the 4-way combined CSV + summary CSV + verdict JSON."""
    m15_rows = _load_marinfold_rows(marinfold_1_5b_csv, "marinfold_1_5b")
    m1b_rows = _load_exp20_1b_rows(marinfold_1b_csv)
    px_rows = _load_protenix_rows(protenix_csv)

    all_rows = m15_rows + m1b_rows + px_rows

    method_order = {name: i for i, name in enumerate(_METHODS)}
    all_rows.sort(
        key=lambda r: (r["pdb_id"], r["chain_id"], method_order.get(r["method"], 99))
    )

    out_scores_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_scores_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_OUR_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {out_scores_csv} ({len(all_rows)} rows).")

    summary = [_aggregate_for_method(all_rows, m) for m in _METHODS]
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
    print(f"Hypothesis verdict: {verdict['verdict']} "
          f"({verdict['n_metrics_supporting']}/{len(_HEADLINE)} headline metrics)")
    return verdict


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--marinfold-1-5b-csv", type=Path,
        default=here / "data" / "marinfold_scores.csv",
        help="This experiment's 1.5B scores (from score_marinfold.py).",
    )
    parser.add_argument(
        "--marinfold-1b-csv", type=Path,
        default=here.parent / "exp20_evals_marinfold_1b_foldbench" / "data" / "scores.csv",
        help="exp20's published 3-way scores.csv; we filter to its 1B rows.",
    )
    parser.add_argument(
        "--protenix-csv", type=Path,
        default=here / "protenix_data" / "data" / "protenix-foldbench-monomers" / "scores.csv",
    )
    parser.add_argument("--scores", type=Path, default=here / "data" / "scores.csv")
    parser.add_argument("--summary", type=Path, default=here / "data" / "scores_summary.csv")
    parser.add_argument(
        "--verdict-json", type=Path,
        default=here / "data" / "hypothesis_verdict.json",
    )
    args = parser.parse_args()
    merge_and_summarize(
        marinfold_1_5b_csv=args.marinfold_1_5b_csv,
        marinfold_1b_csv=args.marinfold_1b_csv,
        protenix_csv=args.protenix_csv,
        out_scores_csv=args.scores,
        out_summary_csv=args.summary,
        out_verdict_json=args.verdict_json,
    )


if __name__ == "__main__":
    main()
