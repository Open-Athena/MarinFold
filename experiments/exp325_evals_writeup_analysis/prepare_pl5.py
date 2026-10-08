"""Prepare P@L/5 counterparts and the KNN comparison from saved predictions.

Use exp89's sequence-length cutoff and stable ranking over its frozen resolved
candidate pairs. No predictor is run. Individual maps keep emission order;
their missing ranks receive zero credit and their oracle is selected anew at
P@L/5. Reproducing every prior R-precision is a required source-identity check.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from generation.score_contacts import parse_rollout
from prepare import Sources, REPO, annotate, bootstrap, cohort_slices, sha256, summarize
from compute_metrics import metric_rows, resolved_pairs, true_matrix
from analyze_results import ordered_true_pairs, vote_matrix

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXP245 = REPO / "experiments/exp245_evals_foldbench_held_out_monomers/data"
EXP277 = REPO / "experiments/exp277_models_single_mpnn_pilot/data/eval_rollout_v2"
HF321 = "hf://buckets/open-athena/MarinFold/data/exp321/null-sequence-guidance-v1/full_iid_single/eval-val"


def rollout_precision(contacts: list[tuple[int, int]], true: set[tuple[int, int]], k: int) -> float:
    """Score distinct emitted contacts, retaining k when fewer were predicted."""
    if k < 1:
        raise ValueError("Precision needs a positive denominator")
    return len(set(list(dict.fromkeys(contacts))[:k]) & true) / k


def contact_sources(sources: Sources, targets: pd.DataFrame) -> pd.DataFrame:
    """Read complete precision tables; check reused rows against the old R scores."""
    stems = set(targets.stem)
    old = sources.csv(REPO / "experiments/exp78_evals_esmfold_contacts/data/contact_precision_all.csv", round_trip=True)
    # 7ur7_A and 8ah9_A also occur as denovo_pdb entries with different inputs.
    # The FoldBench instance takes precedence, exactly as in exp245's reuse.
    old = old.assign(priority=(old.dataset != "foldbench100").astype(int)).sort_values("priority", kind="stable")
    old = old.drop_duplicates(["stem", "model", "mode", "predictor", "range", "cut"])
    esm = sources.csv(DATA / "inputs/pl5_exp226_esm_precision.csv", round_trip=True)
    protenix = sources.csv(DATA / "inputs/pl5_exp226_protenix_precision.csv", round_trip=True).assign(model="protenix-v2")
    new = sources.csv(EXP245 / "baseline_precision_new.csv.gz", round_trip=True)
    # exp245 reran some exp226 proteins with corrected inputs. Its published
    # scores take precedence over the earlier exp226 runs for those overlaps.
    base = pd.concat([new, old, esm, protenix], ignore_index=True)
    base = base[(base.predictor == "structure") & base.stem.isin(stems)].copy()
    base["method"] = base.model.replace({"protenix-v2_msa": "protenix_msa", "protenix-v2_single_seq": "protenix_ss"})
    p = base.model == "protenix-v2"
    base.loc[p, "method"] = base.loc[p, "mode"].map({"msa": "protenix_msa", "single_seq": "protenix_ss"})
    # Keep exp245's reruns, then backfill the remaining archived proteins.
    base = base.drop_duplicates(["stem", "method", "range", "cut"])
    knn = sources.csv(EXP245 / "knn_precision_new.csv.gz", round_trip=True)
    knn = knn[knn.model == "seq-knn-k10-decontam"].assign(method="knn")
    mf = sources.csv(EXP277 / "contact_precision_all.csv", round_trip=True)
    mf = mf[mf.dataset == "foldbench_monomer"].assign(method="marinfold")
    test = sources.csv(DATA / "test_contact_metrics.csv", round_trip=True).assign(method="marinfold")
    frames = [base, knn, mf, test]
    for method in ("af2", "af3", "boltz2"):
        frames.append(sources.csv(DATA / f"{method}_contact_metrics.csv", round_trip=True))
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw.stem.isin(stems) & raw["range"].isin(["all", "long"]) & raw.cut.isin(["R", "L/5"])].copy()
    if raw.duplicated(["stem", "method", "range", "cut"]).any():
        raise ValueError("Duplicate source scores")
    if raw.groupby(["method", "range", "cut"]).size().ne(333).any():
        raise ValueError(f"Incomplete source matrix:\n{raw.groupby(['method', 'range', 'cut']).size()}")
    existing = sources.csv(DATA / "figure_rows.csv", round_trip=True)
    old_r = existing[existing.figure == "04_contacts"].copy()
    old_r["range"] = old_r.metric.map({"r_precision": "all", "r_precision_long": "long"})
    check = raw[raw.cut == "R"].merge(old_r[["stem", "method", "range", "value"]],
                                      on=["stem", "method", "range"], validate="one_to_one")
    bad = check[~np.isclose(check.precision, check.value, atol=1e-12, rtol=0)]
    if len(check) != 333 * 9 * 2 or not bad.empty:
        raise ValueError(f"Archived R-precision does not reproduce:\n{bad[['stem','method','range','precision','value']]}")
    result = raw[raw.cut == "L/5"].copy()
    result["metric"] = result["range"].map({"all": "p_at_l5", "long": "p_at_l5_long"})
    result["source_column"] = "precision"
    return annotate(result[["stem", "method", "metric", "precision", "n_top", "n_candidate", "n_true",
                            "source", "source_row", "source_column"]].rename(columns={"precision": "value"}), targets)


def sampling_sources(targets: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rescore the exact saved 100-map pools and verify their old R diagnostics."""
    truth_path = EXP245 / "gt_universe_scored.jsonl"
    truth = {r["stem"]: r for r in map(json.loads, truth_path.read_text().splitlines())}
    raw_manifest = {str(truth_path.relative_to(REPO)): dict(sha256=sha256(truth_path))}
    individuals, summaries, checks = [], [], []
    for target in targets[targets.designed == 0].sort_values("stem").itertuples():
        stem = target.stem
        record = truth[stem]
        if record["L"] != target.L:
            raise ValueError(f"Sequence length disagrees for {stem}")
        if target.eval_set == "eval-val":
            path = HERE / "scratch/pl5/exp321/full_iid_single/eval-val" / f"{stem}.parquet"
            frame = pd.read_parquet(path).sort_values("rollout").head(100)
            maps = [[(min(int(i), int(j)), max(int(i), int(j))) for i, j in row.contacts] for row in frame.itertuples()]
            valid = (frame.finished & (frame.malformed_contacts == 0)).tolist()
            ids = frame.rollout.astype(int).tolist()
            uri = f"{HF321}/{stem}.parquet"
        else:
            path = HERE / "scratch/contacts/results/rollouts" / f"{stem}.json"
            archive = json.loads(path.read_text())
            parsed = [parse_rollout(r) for r in archive["rollouts"]]
            maps = [pairs for pairs, _ in parsed]
            valid = [r["finish_reason"] == "stop" and malformed == 0
                     for r, (_, malformed) in zip(archive["rollouts"], parsed, strict=True)]
            ids = list(range(len(maps)))
            uri = "hf://buckets/open-athena/MarinFold/data/exp325-writeup-analysis/exp277-step266344/v6-ptm-ranking/contact_rollouts.tar.gz"
        if len(maps) != 100 or ids != list(range(100)):
            raise ValueError(f"Unexpected sample pool for {stem}")
        key = str(path.relative_to(HERE))
        digest = sha256(path)
        raw_manifest[key] = dict(sha256=digest, public_source=uri)
        length = int(record["L"])
        resolved = set(record["resolved"])
        scored = pd.DataFrame(metric_rows(vote_matrix(maps, length), true_matrix(length, record["contacts"]),
                              *resolved_pairs(np.asarray(record["resolved"])), length, with_precision=True))
        for region, minimum in (("all", 6), ("long", 24)):
            true = ordered_true_pairs(record, region)
            chosen = scored[(scored["range"] == region) & (scored.cut == "L/5")].iloc[0]
            k = int(chosen.n_top)
            if k != max(1, length // 5):
                raise ValueError(f"{stem}/{region} has fewer candidates than the requested L/5 cutoff")
            scores, r_scores = [], []
            for idx, (pairs, ok) in enumerate(zip(maps, valid, strict=True)):
                eligible = [(i, j) for i, j in pairs if i in resolved and j in resolved and j-i >= minimum]
                score = rollout_precision(eligible, true, k) if ok else 0.0
                r_score = rollout_precision(eligible, true, len(true)) if ok else 0.0
                scores.append(score)
                r_scores.append(r_score)
                individuals.append(dict(stem=stem, range=region, rollout=idx, valid=ok, L=length, n_top=k,
                    n_emitted=len(set(eligible)), p_at_l5=score, r_precision=r_score,
                    raw_source=key, raw_sha256=digest))
            r_consensus = scored[(scored["range"] == region) & (scored.cut == "R")].precision.iloc[0]
            checks.extend(dict(stem=stem, metric="r_precision" if region == "all" else "r_precision_long",
                               method=m, value=v) for m, v in
                          [("consensus", r_consensus), ("single", np.mean(r_scores)), ("best100", max(r_scores))])
            summaries.append(dict(stem=stem, range=region, L=length, n_top=k, N=100, invalid_rollouts=100-sum(valid),
                consensus=float(chosen.precision), single=float(np.mean(scores)), best100=max(scores),
                best_rollout=int(np.argmax(scores)), raw_source=key, raw_sha256=digest))
    old = pd.read_csv(DATA / "figure_rows.csv")
    old = old[old.figure == "06_sampling"]
    check = pd.DataFrame(checks).merge(old, on=["stem", "metric", "method"], suffixes=("_new", "_old"), validate="one_to_one")
    bad = check[~np.isclose(check.value_new, check.value_old, atol=1e-12, rtol=0)]
    if len(check) != 314 * 2 * 3 or not bad.empty:
        raise ValueError(f"Saved sampling pools do not reproduce R-precision:\n{bad[['stem','metric','method','value_new','value_old']]}")
    pd.DataFrame(individuals).to_csv(DATA / "pl5_sampling_individual.csv.gz", index=False,
                                    compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(summaries).to_csv(DATA / "pl5_sampling_per_protein.csv", index=False)
    return pd.DataFrame(summaries), raw_manifest


def main() -> None:
    """Freeze the numeric figure inputs, with protein-level paired uncertainty."""
    sources = Sources()
    targets = pd.read_csv(DATA / "targets.csv")
    contact = contact_sources(sources, targets).assign(figure="04_contacts_pl5")
    for row in contact.itertuples():
        if row.n_top != min(max(1, row.L // 5), row.n_candidate):
            raise ValueError(f"P@L/5 source length mismatch for {row.stem}/{row.method}")
    _, raw_manifest = sampling_sources(targets)
    sampling = sources.csv(DATA / "pl5_sampling_per_protein.csv", round_trip=True)
    parts = []
    for method in ("single", "consensus", "best100"):
        part = sampling[["stem", "range", "n_top", method, "source", "source_row"]].rename(columns={method: "value"})
        part["metric"] = part["range"].map({"all": "p_at_l5", "long": "p_at_l5_long"})
        parts.append(annotate(part.assign(method=method, source_column=method, figure="06_sampling_pl5"), targets))
    original = sources.csv(DATA / "figure_rows.csv", round_trip=True)
    knn_r = original[(original.figure == "04_contacts") & original.method.isin(["marinfold", "knn"])].copy()
    # Preserve the exact previously plotted values, including their CSV precision.
    # The referenced figure_rows.csv cells retain their own upstream lineage.
    knn_r["source_column"] = "value"
    knn = pd.concat([knn_r, contact[contact.method.isin(["marinfold", "knn"])]], ignore_index=True).assign(figure="04b_knn")
    rows = pd.concat([contact, *parts, knn], ignore_index=True)
    if rows.value.isna().any() or rows.duplicated(["figure", "stem", "method", "metric"]).any():
        raise ValueError("Invalid figure matrix")
    deltas = []
    for figure, left, right in [("04b_knn", "marinfold", "knn"), ("06_sampling_pl5", "best100", "consensus")]:
        for metric, group in rows[rows.figure == figure].groupby("metric"):
            for cohort, frame in cohort_slices(group).items():
                for tier in ["All depths", "<10", "10–99", "100–999", "≥1000"]:
                    subset = frame if tier == "All depths" else frame[frame.tier == tier]
                    if subset.empty:
                        continue
                    wide = subset.pivot(index="stem", columns="method", values="value").sort_index()
                    mean, lo, hi = bootstrap((wide[left] - wide[right]).to_numpy())
                    deltas.append(dict(figure=figure, metric=metric, cohort=cohort, tier=tier, method=left,
                                       reference=right, n=len(wide), delta=mean, ci_low=lo, ci_high=hi))
    rows.to_csv(DATA / "pl5_figure_rows.csv", index=False)
    summary = summarize(rows)
    summary.to_csv(DATA / "pl5_summary.csv", index=False)
    pd.DataFrame(deltas).to_csv(DATA / "pl5_paired_deltas.csv", index=False)
    names = ["pl5_figure_rows.csv", "pl5_summary.csv", "pl5_paired_deltas.csv", "pl5_sampling_per_protein.csv", "pl5_sampling_individual.csv.gz"]
    manifest = dict(preprocessor_sha256=sha256(Path(__file__)), sources=sources.records, raw_sources=raw_manifest,
        files={name: sha256(DATA / name) for name in names},
        cutoff="k=max(1,floor(L/5)), capped only by candidate count; L=frozen input sequence length, not resolved-residue count",
        contact_definition="pyconfind degree >=0.001; resolved pairs; sequence separation >=6 (all) or >=24 (long)",
        ranking="Main consensus ranks votes; structural predictors rank contact degree; KNN uses archived transferred scores. Stable candidate-order ties, no truth tie break.",
        sampling="Exact original first 100 iid maps. Individual maps rank by emission order; duplicates removed; unfilled ranks zero. Invalid/unfinished maps zero. Consensus uses all parsed maps; oracle reselected by P@L/5.",
        regression="All 5994 archived predictor R scores and 1884 sampling R scores reproduced within 1e-12",
        knn="k=10, native decontaminated corpus; does not index the additional ProteinMPNN redesign sequences",
        public_exp321=HF321)
    (DATA / "pl5_analysis.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary[(summary.cohort == "natural") & (summary.tier == "All depths")][["figure", "metric", "method", "n", "mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
