"""Audit saved exp277 eval-val contact maps; never runs a predictor.

Run ``uv run python analyze.py`` from this directory. On first use, read the
published exp345 input bundle, or pass --fetch-source to collect the original
exp245 truth and exp277 score matrices using the CoreWeave ``cw`` profile.
"""

import argparse
import gzip
import hashlib
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
from huggingface_hub import HfFileSystem
from scipy.ndimage import distance_transform_cdt, maximum_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS / "exp89_evals_contacts_v1_model_on_eval_set"))
from compute_metrics import metric_rows, resolved_pairs, true_matrix

MODEL = "marinfold-exp277-full-epoch2-from213072-step479417"
LABEL = "exp277_full_epoch2_from213072_step479417"
RUN = "contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B"
SOURCE = "marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot/evals/rollout-v2/2026-09-13/v2-02"
PUBLIC = "buckets/open-athena/MarinFold/data/exp345-qualitative-contact-map-errors/v1"
GT_SOURCE = "buckets/open-athena/MarinFold/data/contacts-v1-foldbench-monomers-exp245/gt_universe_scored.jsonl"
EXP277_DATA = EXPERIMENTS / "exp277_models_single_mpnn_pilot/data/eval_rollout_v2_epochs"
INPUTS = HERE / "data/report_inputs.json.gz"


def digest(data: bytes) -> str:
    """Return a reproducibility digest."""
    return hashlib.sha256(data).hexdigest()


def collect_inputs() -> dict:
    """Fetch only eval-val score matrices and filter the frozen truth universe."""
    sets_path = EXPERIMENTS / "exp245_evals_foldbench_held_out_monomers/data/eval_sets.csv"
    sets = pd.read_csv(sets_path)
    selected = sets[sets.eval_set.eq("eval-val")].sort_values("stem")
    assert len(selected) == selected.stem.nunique() == 97
    cache = HERE / ".cache"
    cache.mkdir(exist_ok=True)
    hf = HfFileSystem(token=False)
    gt_path = cache / "gt_universe_scored.jsonl"
    if not gt_path.exists():
        gt_path.write_bytes(hf.read_bytes(GT_SOURCE))
    raw_gt = gt_path.read_bytes()
    allowed = set(selected.stem)
    truth = {}
    for line in raw_gt.splitlines():
        record = json.loads(line)
        if record["stem"] in allowed:
            assert record["stem"] not in truth
            truth[record["stem"]] = record
    assert set(truth) == allowed
    fs = s3fs.S3FileSystem(profile="cw", endpoint_url="https://cwobject.com",
                         config_kwargs={"s3": {"addressing_style": "virtual"}})

    def get_one(stem: str) -> dict:
        path = cache / f"{stem}.npz"
        uri = f"{SOURCE}/dense_scores/{LABEL}/foldbench_monomer__{stem}.npz"
        if not path.exists():
            path.write_bytes(fs.cat_file(uri))
        raw = path.read_bytes()
        with np.load(io.BytesIO(raw)) as loaded:
            score = loaded["score"]
        rec = truth[stem]
        length = rec["L"]
        assert score.shape == (length, length)
        assert np.isfinite(score).all() and (score >= 0).all()
        assert np.array_equal(score, score.T)
        # Saved matrices are vote counts, not probabilities. All eval-val
        # samples terminated, so exactly 100 votes were available per protein.
        assert np.allclose(score, np.round(score)) and score.max() <= 100
        i, j = np.where(np.triu(score, 1) > 0)
        return {"truth": rec, "votes": np.column_stack((i, j, score[i, j])).astype(int).tolist(),
                "source": "s3://" + uri, "sha256": digest(raw),
                "is_viral": bool(selected.set_index("stem").loc[stem, "is_viral"])}

    with ThreadPoolExecutor(max_workers=8) as pool:
        proteins = list(pool.map(get_one, sorted(allowed)))
    manifest = json.loads((EXP277_DATA / "run_manifest.json").read_text())
    capped = pd.read_csv(EXP277_DATA / "capped_rollouts.csv")
    assert not capped[capped.stem.isin(allowed)].shape[0]
    return {"schema": 1, "model": MODEL, "run": RUN, "step": 479417,
            "gt_source": "hf://" + GT_SOURCE, "gt_sha256": digest(raw_gt),
            "eval_sets_sha256": digest(sets_path.read_bytes()),
            "sampling": manifest["sampling"], "n_rollouts": 100,
            "proteins": proteins}


def unpack(item: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recover the canonical resolved pair universe and matrix."""
    rec = item["truth"]
    score = np.zeros((rec["L"], rec["L"]), dtype=float)
    votes = np.asarray(item["votes"], dtype=int)
    score[votes[:, 0], votes[:, 1]] = votes[:, 2]
    score += score.T
    truth = true_matrix(rec["L"], rec["contacts"])
    i, j, sep = resolved_pairs(np.asarray(rec["resolved"]))
    keep = sep >= 6
    return score, truth, i[keep], j[keep]


def analyze_one(item: dict) -> tuple[dict, list[dict], list[dict]]:
    """Measure exact ranking, range bias, near misses and missed-contact support."""
    rec = item["truth"]
    length = rec["L"]
    score, truth, i, j = unpack(item)
    g, s = truth[i, j], score[i, j]
    r = int(g.sum())
    order = np.argsort(-s, kind="mergesort")
    chosen = order[:r]
    predicted = np.zeros_like(truth)
    predicted[i[chosen], j[chosen]] = True
    candidate = np.zeros_like(truth)
    candidate[i, j] = True
    truth &= candidate
    tp, fp, fn = predicted & truth, predicted & ~truth, truth & ~predicted
    # Chessboard distance matches ±d residues at each endpoint. This is a
    # many-to-one visual diagnostic, not an alternative contact-accuracy score.
    distances = distance_transform_cdt(~truth, metric="chessboard")
    metrics = metric_rows(score, truth, i, j, j-i, length, with_precision=True)
    rows = [{"stem": rec["stem"], **row} for row in metrics]
    values = {(row["range"], row["cut"]): row["precision"] for row in metrics}
    rng_rows = []
    for name, lo, hi in (("short", 6, 11), ("medium", 12, 23), ("long", 24, length), ("distant", 96, length)):
        mask = (j-i >= lo) & (j-i <= hi)
        n_true = int(g[mask].sum())
        n_pred = int(predicted[i[mask], j[mask]].sum())
        n_tp = int(tp[i[mask], j[mask]].sum())
        rng_rows.append({"stem": rec["stem"], "range": name, "n_true": n_true,
                         "n_predicted_global_top_r": n_pred, "n_tp": n_tp,
                         "recall": n_tp / n_true if n_true else None,
                         "precision": n_tp / n_pred if n_pred else None})
    degrees_gt = truth.sum(0) + truth.sum(1)
    degrees_pred = predicted.sum(0) + predicted.sum(1)
    row = {"stem": rec["stem"], "length": length, "resolved": len(rec["resolved"]),
           "is_viral": item["is_viral"], "n_candidates": len(i), "R": r,
           "r_precision": values["all", "R"], "long_r_precision": values["long", "R"],
           "p_at_l": values["all", "L"], "p_at_l5": values["all", "L/5"],
           "n_tp": int(tp.sum()), "n_fp": int(fp.sum()), "n_fn": int(fn.sum()),
           "gt_long_fraction": float(g[j-i >= 24].sum()/r),
           "pred_long_fraction": float(np.mean(j[chosen]-i[chosen] >= 24)),
           "gt_mean_separation": float((j-i)[g].mean()),
           "pred_mean_separation": float((j-i)[chosen].mean()),
           "true_seen_fraction": float(np.mean(s[g] > 0)),
           "missed_seen_fraction": float(np.mean(score[fn] > 0)) if fn.any() else 1.,
           "true_votes_median": float(np.median(s[g])),
           "tp_votes_median": float(np.median(score[tp])) if tp.any() else 0.,
           "fp_votes_median": float(np.median(score[fp])) if fp.any() else 0.,
           "fn_votes_median": float(np.median(score[fn])) if fn.any() else 0.,
           "cutoff_votes": float(s[chosen[-1]]),
           "cutoff_tie_size": int(np.sum(s == s[chosen[-1]])),
           "degree_correlation": float(spearmanr(degrees_gt[rec["resolved"]], degrees_pred[rec["resolved"]]).statistic)}
    for tolerance in (1, 2, 5):
        near = maximum_filter(truth, size=2*tolerance+1, mode="constant")
        row[f"near_{tolerance}_precision"] = float(near[predicted].mean())
        row[f"near_{tolerance}_random_expectation"] = float(near[i, j].mean())
        row[f"fp_within_{tolerance}_fraction"] = float(np.mean(distances[fp] <= tolerance)) if fp.any() else 0.
        # Fix exact true positives, then permit each remaining predicted pair
        # and each missed true pair to be used at most once. This oracle upper
        # bound prevents dense neighborhoods from rescuing many predictions
        # against a single already-correct contact.
        if fp.any() and fn.any():
            neighbors = cKDTree(np.argwhere(fp)).query_ball_tree(
                cKDTree(np.argwhere(fn)), r=tolerance, p=np.inf)
            left = np.repeat(np.arange(len(neighbors)), [len(v) for v in neighbors])
            right = np.asarray([v for group in neighbors for v in group], dtype=int)
            graph = csr_matrix((np.ones(len(left)), (left, right)), shape=(int(fp.sum()), int(fn.sum())))
            matched = int(np.sum(maximum_bipartite_matching(graph, perm_type="column") >= 0))
        else:
            matched = 0
        row[f"one_to_one_{tolerance}_precision"] = float((tp.sum() + matched) / r)
    for cutoff in (1, 5, 10, 25, 50):
        row[f"true_below_{cutoff}_votes_fraction"] = float(np.mean(s[g] < cutoff))
    return row, rows, rng_rows


def main() -> None:
    """Fetch inputs, reproduce source metrics, and persist diagnostic tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-source", action="store_true")
    args = parser.parse_args()
    (HERE / "data").mkdir(exist_ok=True)
    if args.fetch_source:
        bundle = collect_inputs()
        INPUTS.write_bytes(gzip.compress(json.dumps(bundle, separators=(",", ":")).encode(), mtime=0))
    elif not INPUTS.exists():
        INPUTS.write_bytes(HfFileSystem(token=False).read_bytes(PUBLIC + "/report_inputs.json.gz"))
    bundle = json.loads(gzip.decompress(INPUTS.read_bytes()))
    assert bundle["model"] == MODEL and len(bundle["proteins"]) == 97
    diagnostics, metrics, ranges = [], [], []
    for item in bundle["proteins"]:
        row, metric, range_rows = analyze_one(item)
        diagnostics.append(row)
        metrics.extend(metric)
        ranges.extend(range_rows)
    df, measured = pd.DataFrame(diagnostics), pd.DataFrame(metrics)
    reference_path = HERE / "data/source_reference.csv"
    if not reference_path.exists():
        fs = s3fs.S3FileSystem(profile="cw", endpoint_url="https://cwobject.com",
                             config_kwargs={"s3": {"addressing_style": "virtual"}})
        reference = pd.read_csv(io.BytesIO(fs.cat_file(SOURCE + "/results/marinfold_precision.csv")))
        reference = reference[reference.model.eq(MODEL) & reference.dataset.eq("foldbench_monomer") & reference.stem.isin(df.stem)]
        reference.to_csv(reference_path, index=False)
    reference = pd.read_csv(reference_path)
    reference = reference[reference.model.eq(MODEL) & reference.dataset.eq("foldbench_monomer") & reference.stem.isin(df.stem)]
    checked = measured.merge(reference, on=["stem", "range", "cut"], suffixes=("", "_source"), validate="one_to_one")
    assert len(checked) == 97 * 20
    for key in ("precision", "n_true", "n_top", "n_candidate"):
        np.testing.assert_allclose(checked[key], checked[key + "_source"], rtol=0, atol=1e-12, equal_nan=True)
    df.to_csv(HERE / "data/per_protein.csv", index=False)
    measured.to_csv(HERE / "data/canonical_metrics.csv", index=False)
    pd.DataFrame(ranges).to_csv(HERE / "data/range_diagnostics.csv", index=False)
    stats = {key: float(df[key].mean()) for key in df.select_dtypes("number").columns}
    summary = {"model": MODEL, "run": RUN, "step": 479417, "n_proteins": len(df),
               "canonical_rows_verified": len(checked), "mean_per_protein": stats,
               "viral": df.groupby("is_viral").r_precision.agg(["count", "mean"]).reset_index().to_dict("records"),
               "low_msa_depth_natural": "0 eval-val members in frozen exp260 low-depth natural set",
               "input_bundle_sha256": digest(INPUTS.read_bytes()),
               "reference_source": "s3://" + SOURCE + "/results/marinfold_precision.csv",
               "eval_val_reference_sha256": digest(reference_path.read_bytes()),
               "canonical_scorer_sha256": digest((EXPERIMENTS / "exp89_evals_contacts_v1_model_on_eval_set/compute_metrics.py").read_bytes()),
               "sampling": bundle["sampling"], "gt_source": bundle["gt_source"],
               "gt_sha256": bundle["gt_sha256"], "eval_sets_sha256": bundle["eval_sets_sha256"],
               "source_matrices": [{k: item[k] for k in ("source", "sha256")} for item in bundle["proteins"]]}
    (HERE / "data/provenance.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("source_matrices", "sampling")}, indent=2))
    print(df.sort_values("r_precision")[["stem", "length", "R", "r_precision", "near_2_precision", "true_seen_fraction", "fp_votes_median"]].to_string(index=False))


if __name__ == "__main__":
    main()
