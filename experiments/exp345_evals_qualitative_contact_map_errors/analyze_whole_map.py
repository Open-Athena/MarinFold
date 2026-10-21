"""Separate sample quality, sample selection and marginal-vote mixing on eval-val.

Only post-generation scoring sees truth. Clustering and blind selectors consume
contact maps or model likelihoods. Full-map F1 handles variable sample sizes;
emission-order fixed-R precision is a separately labelled secondary diagnostic.
"""

import gzip
import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering

from analyze import HERE, INPUTS, metric_rows, resolved_pairs, true_matrix

CACHE = HERE / ".cache/whole_maps/production"


def contact_sets(frame: pd.DataFrame, record: dict) -> list[list[tuple[int, int]]]:
    """Keep emission order while restricting maps to the fixed resolved universe."""
    resolved = set(record["resolved"])
    return [[(int(i), int(j)) for i, j in row.contacts if int(i) in resolved and int(j) in resolved and int(j)-int(i) >= 6]
            for row in frame.itertuples()]


def jaccard_matrix(maps: list[list[tuple[int, int]]]) -> np.ndarray:
    """Compute exact unordered-map overlap for all pairs of samples."""
    universe = sorted({pair for contacts in maps for pair in contacts})
    index = {pair: n for n, pair in enumerate(universe)}
    rows = [r for r, contacts in enumerate(maps) for _ in contacts]
    cols = [index[pair] for contacts in maps for pair in contacts]
    matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(maps), len(universe)))
    overlap = (matrix @ matrix.T).toarray()
    sizes = np.asarray([len(m) for m in maps])
    union = sizes[:, None] + sizes[None, :] - overlap
    return np.divide(overlap, union, out=np.ones_like(overlap), where=union > 0)


def map_metrics(contacts: list[tuple[int, int]], truth: set[tuple[int, int]]) -> dict:
    """Score a whole unordered map and, separately, its emission-order prefix."""
    predicted = set(contacts)
    assert len(predicted) == len(contacts)
    n, r, tp = len(predicted), len(truth), len(predicted & truth)
    assert r > 0
    return {"n_contacts": n, "n_true": r, "tp": tp,
            "precision": tp/n if n else 0., "recall": tp/r,
            "f1": 2*tp/(n+r), "ordered_fixed_r": len(set(contacts[:r]) & truth)/r}


def pooled_map(maps: list[list[tuple[int, int]]], indices: np.ndarray, record: dict, minimum: int) -> list[tuple[int, int]]:
    """Rank occurrence counts over the canonical universe, retaining stable ties."""
    truth = true_matrix(record["L"], record["contacts"])
    i, j, _ = resolved_pairs(np.asarray(record["resolved"]))
    keep = j-i >= minimum
    i, j = i[keep], j[keep]
    r = int(truth[i, j].sum())
    votes = np.zeros((record["L"], record["L"]), dtype=np.int32)
    for k in indices:
        for a, b in maps[int(k)]:
            if b-a >= minimum:
                votes[a, b] += 1
    chosen = np.argsort(-votes[i, j], kind="mergesort")[:r]
    return [(int(i[k]), int(j[k])) for k in chosen]


def clusters_from_pool(similarity: np.ndarray, training: np.ndarray, test: np.ndarray, k: int) -> dict:
    """Learn clusters without truth and assign independent samples to their medoids."""
    distance = np.clip(1-similarity[np.ix_(training, training)], 0, 1)
    np.fill_diagonal(distance, 0)
    labels = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(distance)
    groups = [training[labels == label] for label in range(k)]
    medoids = [int(group[np.argmax(similarity[np.ix_(group, group)].mean(axis=1))]) for group in groups]
    assigned = np.argmax(similarity[np.ix_(test, medoids)], axis=1)
    return {"groups": groups, "medoids": medoids, "assigned": assigned,
            "test_groups": [test[assigned == label] for label in range(k)],
            "dominant": int(np.argmax([len(group) for group in groups]))}


def analyze_protein(item: dict, frame: pd.DataFrame) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Score pooled, individual, cross-pool-selected and clustered contact maps."""
    record = item["truth"]
    stem = record["stem"]
    maps = contact_sets(frame, record)
    similarity = jaccard_matrix(maps)
    pool_a, pool_b = np.arange(100), np.arange(100, 200)
    folds = [("A", pool_a, pool_b), ("B", pool_b, pool_a)]
    clustering = {(name, k): clusters_from_pool(similarity, train, test, k)
                  for name, train, test in folds for k in (2, 4)}
    canonical = {}
    pi, pj, psep = resolved_pairs(np.asarray(record["resolved"]))
    for label, indices in (("pooled_200", np.arange(200)), ("pooled_100_A", pool_a), ("pooled_100_B", pool_b)):
        votes = np.zeros((record["L"], record["L"]), dtype=float)
        for n in indices:
            for i, j in maps[n]:
                votes[i, j] += 1
        canonical[label] = metric_rows(votes, true_matrix(record["L"], record["contacts"]),
                                        pi, pj, psep, record["L"], with_precision=True)
    sample_rows, results, cluster_rows = [], [], []
    selections = {}
    for region, minimum in (("all", 6), ("long", 24)):
        truth = {(int(i), int(j)) for i, j, d in record["contacts"]
                 if d >= .001 and j-i >= minimum and i in record["resolved"] and j in record["resolved"]}
        region_maps = [[pair for pair in pairs if pair[1]-pair[0] >= minimum] for pairs in maps]
        individual = [map_metrics(contacts, truth) for contacts in region_maps]
        scores = np.asarray([m["f1"] for m in individual])
        oracle = int(np.argmax(scores))
        for n, metrics in enumerate(individual):
            sample_rows.append({"stem": stem, "range": region, "rollout": n, "pool": "A" if n<100 else "B",
                                "mean_logprob": float(frame.iloc[n].mean_logprob), **metrics})
        def add(method: str, metrics: dict, fold: str = "all", sample: int = -1, scoring_range: str = region, **extra) -> None:
            results.append({"stem": stem, "range": scoring_range, "method": method, "fold": fold, "sample": sample, **metrics, **extra})
        for label, indices in (("pooled_200", np.arange(200)), ("pooled_100_A", pool_a), ("pooled_100_B", pool_b)):
            values = map_metrics(pooled_map(maps, indices, record, minimum), truth)
            reference = next(r for r in canonical[label] if r["range"] == region and r["cut"] == "R")
            assert abs(values["f1"] - reference["precision"]) < 1e-12
            add(label, values)
        add("sample_mean_200", {key: float(np.mean([m[key] for m in individual])) for key in individual[0]})
        add("sample_oracle_200", individual[oracle], sample=oracle)
        add("sample_oracle_recall_p50", {"recall": max([m["recall"] for m in individual if m["precision"] >= .5], default=0.)})
        # Consensus chooses from one pool using similarity to the other pool.
        # Whole-map similarity (including both short and long contacts) is the
        # same in each scoring range, so these are actual single selected maps.
        for name, query, support in folds:
            medoid = int(query[np.argmax(similarity[np.ix_(query, support)].mean(axis=1))])
            likelihood = int(query[np.argmax(frame.iloc[query].mean_logprob.to_numpy())])
            best = int(query[np.argmax(scores[query])])
            add("cross_pool_medoid", individual[medoid], name, medoid)
            add("mean_token_logprob", individual[likelihood], name, likelihood)
            add("sample_oracle_100", individual[best], name, best)
            for k in (2, 4):
                fitted = clustering[name, k]
                valid = []
                for c, test_group in enumerate(fitted["test_groups"]):
                    entry = {"stem": stem, "range": region, "training_pool": name, "k": k, "cluster": c,
                             "n_train": len(fitted["groups"][c]), "n_test": len(test_group),
                             "medoid": fitted["medoids"][c], "dominant": c == fitted["dominant"],
                             "eligible": len(test_group) >= 10}
                    if entry["eligible"]:
                        metrics = map_metrics(pooled_map(maps, test_group, record, minimum), truth)
                        entry.update(metrics)
                        valid.append((c, metrics))
                    cluster_rows.append(entry)
                if not valid:
                    raise ValueError(f"{stem}: no evaluable clusters for {name} k={k}")
                best_cluster, best_metrics = max(valid, key=lambda pair: pair[1]["f1"])
                add(f"cluster_oracle_k{k}", best_metrics, name, cluster=best_cluster)
                dominant = dict(valid).get(fitted["dominant"])
                add(f"cluster_dominant_k{k}", dominant or {"f1": np.nan, "precision": np.nan, "recall": np.nan},
                    name, cluster=fitted["dominant"])
            if name == "A":
                selections[region] = {"oracle_200": oracle, "consensus_A": medoid, "likelihood_A": likelihood,
                                      "pooled_200": pooled_map(maps, np.arange(200), record, minimum)}
    diagnostics = {"stem": stem, "mean_pairwise_jaccard": float(similarity[np.triu_indices(200, 1)].mean()),
                   "distinct_maps": len({tuple(sorted(pairs)) for pairs in maps}),
                   "mean_n_contacts": float(np.mean([len(pairs) for pairs in maps])),
                   "selections": selections}
    return sample_rows, results, cluster_rows, diagnostics


def bootstrap(values: np.ndarray) -> dict:
    """Return a protein-level mean and paired bootstrap confidence interval."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    assert len(values)
    rng = np.random.default_rng(345)
    boot = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    return {"n": len(values), "mean": float(values.mean()), "ci_low": float(np.quantile(boot, .025)), "ci_high": float(np.quantile(boot, .975))}


def main() -> None:
    """Analyze every completed protein and save audit tables and selections."""
    bundle = json.loads(gzip.decompress(INPUTS.read_bytes()))
    samples, metrics, clusters, diagnostics = [], [], [], []
    case_names = set(json.loads((HERE / "data/case_notes.json").read_text()))
    case_rows, case_metadata = [], {}
    for item in bundle["proteins"]:
        frame = pd.read_parquet(CACHE / f"{item['truth']['stem']}.parquet").sort_values("rollout")
        assert len(frame) == 200 and frame.finish_reason.eq("stop").all()
        a, b, c, d = analyze_protein(item, frame)
        samples.extend(a); metrics.extend(b); clusters.extend(c); diagnostics.append(d)
        record = item["truth"]
        if record["stem"] in case_names:
            maps = contact_sets(frame, record)
            selected = d["selections"]["all"]
            case_metadata[record["stem"]] = {"L": record["L"], "resolved": record["resolved"]}
            truth_pairs = [(int(i), int(j)) for i, j, degree in record["contacts"]
                           if degree >= .001 and j-i >= 6 and i in record["resolved"] and j in record["resolved"]]
            variants = [("pooled", -1, selected["pooled_200"]),
                        ("oracle", selected["oracle_200"], maps[selected["oracle_200"]]),
                        ("consensus", selected["consensus_A"], maps[selected["consensus_A"]]),
                        ("truth", -1, truth_pairs)]
            for variant, rollout, pairs in variants:
                case_rows.extend({"stem": record["stem"], "variant": variant, "rollout": rollout, "i": i, "j": j} for i, j in pairs)
        print(item["truth"]["stem"], flush=True)
    pd.DataFrame(case_rows).to_csv(HERE / "data/whole_map_case_contacts.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    (HERE / "data/whole_map_case_metadata.json").write_text(json.dumps(case_metadata, separators=(",", ":")) + "\n")
    samples, metrics, clusters = pd.DataFrame(samples), pd.DataFrame(metrics), pd.DataFrame(clusters)
    samples.to_csv(HERE / "data/whole_map_sample_metrics.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    metrics.to_csv(HERE / "data/whole_map_methods.csv", index=False)
    clusters.to_csv(HERE / "data/whole_map_clusters.csv", index=False)
    pd.DataFrame([{k: v for k, v in d.items() if k != "selections"} for d in diagnostics]).to_csv(HERE / "data/whole_map_diversity.csv", index=False)
    (HERE / "data/whole_map_selections.json").write_text(json.dumps(diagnostics, separators=(",", ":")) + "\n")
    summary, deltas = [], []
    for region in ("all", "long"):
        sub = metrics[metrics["range"].eq(region)]
        # Average the two swapped sample pools within each protein first.
        agg = sub.groupby(["stem", "method"])[["f1", "precision", "recall", "ordered_fixed_r"]].mean()
        baseline = agg.xs("pooled_200", level="method")
        for method in agg.index.get_level_values("method").unique():
            values = agg.xs(method, level="method")
            for metric in ("f1", "precision", "recall", "ordered_fixed_r"):
                if not values[metric].notna().any():
                    continue
                summary.append({"range": region, "method": method, "metric": metric, **bootstrap(values[metric].to_numpy())})
                paired = values[metric] - baseline[metric]
                deltas.append({"range": region, "method": method, "metric": metric, "reference": "pooled_200", **bootstrap(paired.to_numpy())})
    pd.DataFrame(summary).to_csv(HERE / "data/whole_map_summary.csv", index=False)
    pd.DataFrame(deltas).to_csv(HERE / "data/whole_map_deltas.csv", index=False)
    saved = pd.read_csv(HERE / "data/per_protein.csv").set_index("stem")
    validation = []
    for region, column in (("all", "r_precision"), ("long", "long_r_precision")):
        for pool in ("A", "B"):
            fresh = metrics[metrics["range"].eq(region) & metrics.method.eq(f"pooled_100_{pool}")].set_index("stem").f1
            delta = fresh - saved[column]
            validation.append({"range": region, "pool": pool, "fresh_mean": float(fresh.mean()),
                               "reference_mean": float(saved[column].mean()), "pearson": float(fresh.corr(saved[column])),
                               **bootstrap(delta.to_numpy()), "within_005": bool(abs(delta.mean()) <= .005)})
    (HERE / "data/whole_map_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    cohorts = {"viral": {p["truth"]["stem"] for p in bundle["proteins"] if p["is_viral"]},
               "nonviral": {p["truth"]["stem"] for p in bundle["proteins"] if not p["is_viral"]},
               "saved_worst_quartile": set(saved.sort_values("r_precision").head(25).index)}
    subgroup_rows = []
    for cohort, stems in cohorts.items():
        for (region, method), sub in metrics[metrics.stem.isin(stems)].groupby(["range", "method"]):
            values = sub.groupby("stem").f1.mean().dropna()
            if len(values):
                subgroup_rows.append({"cohort": cohort, "range": region, "method": method, **bootstrap(values.to_numpy())})
    pd.DataFrame(subgroup_rows).to_csv(HERE / "data/whole_map_subgroups.csv", index=False)
    print(pd.DataFrame(summary).query("metric == 'f1'").to_string(index=False))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
