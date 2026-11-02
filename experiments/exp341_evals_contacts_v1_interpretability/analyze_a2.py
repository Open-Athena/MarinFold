# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""A2 analysis: how local the contact Jacobian is, and what tracks it.

Reads ``a2_contact_jacobian.py``'s per-protein outputs (``--a2-dir``) and
the covariates in ``data/a2_proteins.csv``. Per predicted contact (i, j)
(the top-L pairs by clean P(contact) at separation >= 6), and per mutated
residue k, the effect is the summed size of the change over the four
substitutions. Two effect scales:

- ``raw``: |ΔP|, the plan's definition (section 5.2).
- ``log``: |log(P'/P) - m|, where m is that mutation's median log ratio over
  the protein's L predicted contacts. P(contact) is the probability that
  {i, j} is the next contact statement, so it sums to about one over pairs:
  a mutation that strengthens some contacts lowers every other one through
  renormalization. Subtracting m removes that common shift, and the log
  ratio puts high- and low-P contacts on one scale. P' is floored at
  ``P_FLOOR``: a substitution at a contact's own residues can knock it out,
  and the fp32 difference then overshoots to P' slightly below zero (down to
  about -1e-5 on the A4 panel), where an unfloored log would turn the
  strongest effects into NaNs that the sums drop.

From those effects, per contact:

- ``locality``: share of the effect from residues within 3 of i or j;
  ``null`` is that neighbourhood's share of the chain.
- ``structural``: share from residues outside that neighbourhood that are
  true contacts (separation >= 6) of i or j in the experimental structure;
  ``structural_null`` is their share of the chain.
- an effect-per-residue profile over sequence distance from the nearer of
  i and j, each bin divided by the contact's mean effect per residue.

Contacts are averaged within protein and proteins within set; CIs bootstrap
proteins (``N_BOOT``, seeded). Outputs:

- ``data/a2_per_protein.csv``: per protein, both scales, true and false
  predicted contacts separately, precision at L, and covariates.
- ``data/a2_distance_profile.csv``: the enrichment profile per scale.
- ``data/a2_summary.json``: set means, true-vs-false paired differences,
  and an OLS of logit(locality) on log length, log(1 + MSA N_eff),
  log(1 + training hits) and a novel-fold indicator, with bootstrap CIs.
- ``data/a2_timings.csv``: the per-protein predictor timings.
- with ``--compare-dir`` (the same proteins rerun at another precision or
  batch size): ``data/a2_precision_check.json``, how far ΔP and locality
  move between the two runs.

    uv run python analyze_a2.py --a2-dir out/a2 [--compare-dir out/a2_fp32]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import harness
from marinfold.document_structures.contacts_v1.inference import gt_contact_matrix

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
NEIGHBOURHOOD = 3
DISTANCE_EDGES = [0, 1, 2, 4, 8, 16, 32, 64, np.inf]
DISTANCE_LABELS = ["0", "1", "2-3", "4-7", "8-15", "16-31", "32-63", "64+"]
SCALES = ("raw", "log")
N_BOOT = 10_000
SEED = 341
P_FLOOR = 1e-5


def effects(arrays: dict) -> dict[str, np.ndarray]:
    """Per scale, the ``[L residues, contacts]`` effect summed over substitutions."""
    dp = arrays["dp_top"].astype(np.float64)
    p = arrays["p_clean_top"].astype(np.float64)
    log_ratio = np.log(np.maximum(p[None, None, :] + dp, P_FLOOR) / p[None, None, :])
    centered = log_ratio - np.nanmedian(log_ratio, axis=2, keepdims=True)
    return {
        "raw": np.nansum(np.abs(dp), axis=1),
        "log": np.nansum(np.abs(centered), axis=1),
    }


def contact_masks(length: int, top_pairs: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sequence distance to the nearer endpoint, the ±3 neighbourhood, and structural neighbours."""
    positions = np.arange(length)[:, None]
    i, j = top_pairs[:, 0], top_pairs[:, 1]
    distance = np.minimum(np.abs(positions - i[None, :]), np.abs(positions - j[None, :]))
    near = distance <= NEIGHBOURHOOD
    structural = (gt[:, i] | gt[:, j]) & ~near
    return distance, near, structural


def protein_rows(key: str, arrays: dict, gt: np.ndarray, is_true: np.ndarray) -> tuple[dict, list[dict]]:
    length = arrays["dp_top"].shape[0]
    distance, near, structural = contact_masks(length, arrays["top_pairs"], gt)
    row = {"key": key, "precision_at_l": is_true.mean(), "null": near.mean(axis=0).mean(),
           "structural_null": structural.mean(axis=0).mean()}
    profile = []
    for scale, effect in effects(arrays).items():
        total = effect.sum(axis=0)
        moved = total > 0
        locality = (effect * near).sum(axis=0)[moved] / total[moved]
        struct_share = (effect * structural).sum(axis=0)[moved] / total[moved]
        true = is_true[moved]
        row |= {
            f"locality_{scale}": locality.mean(),
            f"locality_{scale}_true": locality[true].mean() if true.any() else np.nan,
            f"locality_{scale}_false": locality[~true].mean() if (~true).any() else np.nan,
            f"structural_{scale}": struct_share.mean(),
        }
        per_residue = effect[:, moved] / effect[:, moved].mean(axis=0, keepdims=True)
        bins = np.digitize(distance[:, moved], DISTANCE_EDGES[1:-1])
        for b, label in enumerate(DISTANCE_LABELS):
            in_bin = bins == b
            counts = in_bin.sum(axis=0)
            if not counts.any():
                continue
            sums = (per_residue * in_bin).sum(axis=0)
            profile.append({"key": key, "scale": scale, "distance": label,
                            "enrichment": (sums[counts > 0] / counts[counts > 0]).mean()})
    return row, profile


def boot_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    values = values[np.isfinite(values)]
    draws = rng.choice(values, size=(N_BOOT, len(values))).mean(axis=1)
    return [float(values.mean()), *map(float, np.quantile(draws, [0.025, 0.975]))]


def ols(frame: pd.DataFrame, response: str, predictors: list[str], rng: np.random.Generator) -> dict:
    """Coefficients with a bootstrap-over-proteins 95% CI."""
    data = frame[[response, *predictors]].dropna()
    x = np.column_stack([np.ones(len(data)), data[predictors].to_numpy(float)])
    y = data[response].to_numpy(float)
    point = np.linalg.lstsq(x, y, rcond=None)[0]
    draws = np.empty((N_BOOT, len(point)))
    for b in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        draws[b] = np.linalg.lstsq(x[idx], y[idx], rcond=None)[0]
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    names = ["intercept", *predictors]
    return {"n": len(y), "coefficients": {n: [float(c), float(lo), float(hi)]
                                          for n, c, lo, hi in zip(names, point, low, high)}}


def precision_check(a2_dir: Path, compare_dir: Path) -> dict:
    """How much ΔP and locality move when the same proteins are rerun differently.

    The rerun can reorder P(contact) enough to change a few of the top-L
    pairs, so everything is compared on the pairs both runs kept. Effects
    are computed on each run's full top-L set (the log scale's median shift
    uses all of them) and then restricted.
    """
    rows = []
    for path in sorted((compare_dir / "proteins").glob("*.npz")):
        base_path = a2_dir / "proteins" / path.name
        if not base_path.exists():
            continue
        base, other = dict(np.load(base_path)), dict(np.load(path))
        index_b = {tuple(pair): n for n, pair in enumerate(other["top_pairs"])}
        keep_a = np.array([n for n, pair in enumerate(base["top_pairs"]) if tuple(pair) in index_b])
        keep_b = np.array([index_b[tuple(base["top_pairs"][n])] for n in keep_a])
        pairs = base["top_pairs"][keep_a]
        dp_a = base["dp_top"].astype(np.float64)[:, :, keep_a]
        dp_b = other["dp_top"].astype(np.float64)[:, :, keep_b]
        finite = np.isfinite(dp_a) & np.isfinite(dp_b)
        difference = np.abs(dp_a - dp_b)[finite]
        length = dp_a.shape[0]
        distance, near, _ = contact_masks(length, pairs, np.zeros((length, length), bool))
        far = np.broadcast_to((distance > 16)[:, None, :], dp_a.shape)[finite]
        size = np.abs(dp_b)[finite]
        p_a, p_b = base["p_clean_top"][keep_a], other["p_clean_top"][keep_b]
        row = {"key": path.stem, "L": length, "top_pairs_shared": len(keep_a) / len(base["top_pairs"]),
               "p_clean_median_rel_diff": float(np.median(np.abs(p_a - p_b) / p_b)),
               "dp_corr": float(np.corrcoef(dp_a[finite], dp_b[finite])[0, 1]),
               "far_abs_dp_median": float(np.median(size[far])),
               "far_abs_diff_median": float(np.median(difference[far])),
               "far_diff_over_dp": float(np.median(difference[far]) / np.median(size[far]))}
        effects_a, effects_b = effects(base), effects(other)
        for scale in SCALES:
            e_a, e_b = effects_a[scale][:, keep_a], effects_b[scale][:, keep_b]
            row[f"locality_{scale}_base"] = float(np.nanmean((e_a * near).sum(0) / e_a.sum(0)))
            row[f"locality_{scale}_compare"] = float(np.nanmean((e_b * near).sum(0) / e_b.sum(0)))
        rows.append(row)
    return {"base": str(a2_dir), "compare": str(compare_dir), "proteins": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a2-dir", type=Path, required=True)
    parser.add_argument("--compare-dir", type=Path, default=None)
    args = parser.parse_args()

    table = pd.read_csv(DATA / "a2_proteins.csv")
    proteins = {p.stem: p for p in harness.load_eval_proteins() + harness.load_legacy_proteins(DATA / "a2_legacy_gt.jsonl.gz")}
    rows, profile = [], []
    for key in table["key"]:
        path = args.a2_dir / "proteins" / f"{key}.npz"
        if not path.exists():
            continue
        arrays = dict(np.load(path))
        protein = proteins[key]
        gt = gt_contact_matrix(protein.structure.gt_contacts, protein.length, harness.MIN_SEQ_SEPARATION)
        gt = gt | gt.T
        is_true = gt[arrays["top_pairs"][:, 0], arrays["top_pairs"][:, 1]]
        row, prof = protein_rows(key, arrays, gt, is_true)
        rows.append(row)
        profile.extend(prof)
    missing = sorted(set(table["key"]) - {row["key"] for row in rows})
    per_protein = table.merge(pd.DataFrame(rows), on="key", how="inner", validate="one_to_one")
    per_protein["log_length"] = np.log(per_protein["L"])
    per_protein["log1p_msa_neff"] = np.log1p(per_protein["msa_neff"])
    per_protein["log1p_train_hits"] = np.log1p(per_protein["n_hits_significant"])
    per_protein["novel_fold"] = (per_protein["fold_verdict"] == "novel_fold").astype(float)
    for scale in SCALES:
        loc = per_protein[f"locality_{scale}"]
        per_protein[f"logit_locality_{scale}"] = np.log(loc / (1 - loc))
        per_protein[f"enrichment_{scale}"] = loc / per_protein["null"]

    rng = np.random.default_rng(SEED)
    summary = {"n_proteins": len(per_protein), "missing": missing, "scales": {}}
    groups = {"all": per_protein, **dict(tuple(per_protein.groupby("set"))),
              "novel_no_homolog": per_protein[per_protein["novel_no_homolog"]]}
    predictors = ["log_length", "log1p_msa_neff", "log1p_train_hits", "novel_fold"]
    for scale in SCALES:
        block = {"by_group": {}}
        for name, frame in groups.items():
            block["by_group"][name] = {
                "n": len(frame),
                "locality": boot_ci(frame[f"locality_{scale}"].to_numpy(), rng),
                "null": float(frame["null"].mean()),
                "enrichment_median": float(frame[f"enrichment_{scale}"].median()),
                "structural": boot_ci(frame[f"structural_{scale}"].to_numpy(), rng),
                "structural_null": float(frame["structural_null"].mean()),
            }
        paired = (per_protein[f"locality_{scale}_true"] - per_protein[f"locality_{scale}_false"]).to_numpy()
        block["true_minus_false"] = boot_ci(paired, rng)
        block["ols_logit_locality"] = ols(per_protein, f"logit_locality_{scale}", predictors, rng)
        block["ols_logit_locality_with_precision"] = ols(
            per_protein, f"logit_locality_{scale}", [*predictors, "precision_at_l"], rng)
        summary["scales"][scale] = block

    DATA.mkdir(exist_ok=True)
    per_protein.to_csv(DATA / "a2_per_protein.csv", index=False, float_format="%.6g")
    profile = pd.DataFrame(profile)
    profile["distance"] = pd.Categorical(profile["distance"], categories=DISTANCE_LABELS, ordered=True)
    (profile.groupby(["scale", "distance"], observed=True)["enrichment"]
     .agg(mean="mean", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75), n="size")
     .reset_index().to_csv(DATA / "a2_distance_profile.csv", index=False, float_format="%.6g"))
    (DATA / "a2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    timings = pd.concat([pd.read_csv(path) for path in sorted((args.a2_dir / "timings").glob("*.csv"))])
    timings.to_csv(DATA / "a2_timings.csv", index=False)
    if args.compare_dir is not None:
        check = precision_check(args.a2_dir, args.compare_dir)
        (DATA / "a2_precision_check.json").write_text(json.dumps(check, indent=2) + "\n")

    print(f"{len(per_protein)} proteins; missing {len(missing)}")
    for scale in SCALES:
        block = summary["scales"][scale]
        print(f"[{scale}] true-false {np.round(block['true_minus_false'], 3)}")
        for name, stats in block["by_group"].items():
            print(f"  {name:20s} n={stats['n']:3d} locality {np.round(stats['locality'], 3)} null {stats['null']:.3f} "
                  f"enrich {stats['enrichment_median']:.2f} structural {np.round(stats['structural'], 3)} "
                  f"(null {stats['structural_null']:.3f})")
        for name, coef in block["ols_logit_locality"]["coefficients"].items():
            print(f"  ols {name:18s} {np.round(coef, 3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
