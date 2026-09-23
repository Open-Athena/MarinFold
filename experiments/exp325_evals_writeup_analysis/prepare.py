"""Prepare auditable figure tables; rendering never imports this module.

Reuse archived predictions, alignments and scores. This step performs joins,
complete-case selection and protein bootstraps, but no predictor inference.
Every plotted aggregate retains its contributing source rows in figure_rows.csv.
"""

import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE / "data"
FIG250 = "experiments/exp250_evals_exploration_notebook/figures/data"
EXP245 = "experiments/exp245_evals_foldbench_held_out_monomers/data"
EXP277 = "experiments/exp277_models_single_mpnn_pilot/data"
EXP311 = "experiments/exp311_evals_helico_exp277_contact_count_sweep/data"
EXP321 = "experiments/exp321_evals_null_sequence_contrastive_guidance_for_contacts/data"
TIERS = ["<10", "10–99", "100–999", "≥1000"]
SEED = 325
BOOTSTRAPS = 5000
CONTACT_NAMES = {
    "MarinFold exp277": "marinfold",
    "Protenix-v2 + MSA": "protenix_msa",
    "Protenix-v2 single-seq": "protenix_ss",
    "ESMFold2": "esmfold2",
    "ESMFold": "esmfold",
    "seq-KNN (decontaminated corpus)": "knn",
}
STRUCTURE_NAMES = {
    "oracle": "oracle",
    "off": "no_contacts",
    "protenix_v2_msa": "protenix_msa",
    "protenix_v2_single_seq": "protenix_ss",
    "esmfold2": "esmfold2",
    "esmfold": "esmfold",
}
FIGURES = {
    "01_predictors": ["af3", "af2", "boltz2", "protenix_msa", "esmfold2", "esmfold", "protenix_ss"],
    "02_oracle": ["oracle", "af3", "af2", "boltz2", "protenix_msa", "no_contacts"],
    "04_contacts": ["af3", "af2", "boltz2", "protenix_msa", "esmfold2", "esmfold", "marinfold", "knn", "protenix_ss"],
    "05_folding": ["af3", "af2", "boltz2", "protenix_msa", "esmfold2", "esmfold", "marinfold_helico", "protenix_ss", "no_contacts"],
    "06_sampling": ["single", "consensus", "best100"],
}


def sha256(path: Path) -> str:
    """Return the content digest of a small artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Sources:
    """Record each input once and attach zero-based source row indices."""

    def __init__(self) -> None:
        self.records: dict[str, dict] = {}

    def csv(self, path: Path) -> pd.DataFrame:
        """Read a source table and retain the identity of every input row."""
        key = str(path.relative_to(REPO))
        self.records[key] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        frame = pd.read_csv(path)
        frame["source"] = key
        frame["source_row"] = np.arange(len(frame))
        return frame


def depth_tier(depth: pd.Series) -> pd.Series:
    """Bin the query-inclusive sequence count; unknown depth stays unknown."""
    return pd.cut(depth, [0, 10, 100, 1000, np.inf], right=False, labels=TIERS).astype("str")


def bootstrap(values: np.ndarray, seed: int = SEED) -> tuple[float, float, float]:
    """Return a protein-weighted mean and percentile 95% bootstrap interval."""
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires a nonempty, finite protein-level vector")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(len(values), size=(BOOTSTRAPS, len(values)))].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def remote_inputs() -> dict:
    """Fetch missing pinned snapshots and reject any digest mismatch."""
    manifest = json.loads((HERE / "sources_remote.json").read_text())
    for name, record in manifest.items():
        path = DATA / "inputs" / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(record["url"]) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise ValueError(f"Remote source changed: {name}")
            path.write_bytes(payload)
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Input checksum mismatch: {name}")
    return manifest


def annotate(frame: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Join the unique FoldBench manifest without changing the metric rows."""
    out = frame.merge(targets, on="stem", how="left", validate="many_to_one")
    if out.eval_set.isna().any():
        raise ValueError("Metric rows without a FoldBench annotation")
    return out


def prepare_targets(sources: Sources) -> pd.DataFrame:
    """Join canonical target identities to the exact MSA used by Protenix."""
    targets = sources.csv(REPO / EXP245 / "eval_sets.csv")
    targets = targets.loc[targets.scorable == 1, ["stem", "eval_set", "designed", "is_viral", "seq_len"]].rename(columns={"seq_len": "L"})
    msa = sources.csv(DATA / "inputs/helico_msa_depth.csv")
    msa = msa.rename(columns={"target_id": "stem", "n_sequences": "msa_depth", "seq_len": "msa_query_length"})
    if len(targets) != 333 or targets.stem.duplicated().any() or msa.stem.duplicated().any():
        raise ValueError("Expected 333 unique scorable FoldBench targets")
    if set(targets.stem) != set(msa.stem):
        raise ValueError("MSA and evaluation universes differ")
    check = targets.merge(msa, on="stem", suffixes=("", "_msa"), validate="one_to_one")
    if not (check.eval_set == check.eval_set_msa).all() or not (check.designed == check.designed_msa).all():
        raise ValueError("MSA manifest has inconsistent split or design annotations")
    if not (check.msa_depth >= 1).all():
        raise ValueError("Query-inclusive MSA counts must be positive")
    targets = check[["stem", "eval_set", "designed", "is_viral", "L", "msa_query_length", "msa_depth"]].copy()
    targets["tier"] = depth_tier(targets.msa_depth)
    return targets


def prepare_structure(sources: Sources, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use one complete-case population for all three structure figures."""
    raw = sources.csv(REPO / FIG250 / "3_structure_accuracy/per_target.csv")
    raw = raw[raw.arm.isin(STRUCTURE_NAMES)].copy()
    # The upstream combined table has stem only on Helico rows; target_id is
    # the shared key on both the external predictors and Helico results.
    if not (raw.loc[raw.stem.notna(), "stem"] == raw.loc[raw.stem.notna(), "target_id"]).all():
        raise ValueError("Structural target_id disagrees with stem")
    raw["stem"] = raw.target_id
    raw["method"] = raw.arm.map(STRUCTURE_NAMES)
    if raw.duplicated(["stem", "method"]).any():
        raise ValueError("Duplicate structural predictor results")
    valid = raw[(raw.status == "ok") & raw[["gdt_ts", "lddt"]].notna().all(axis=1)]
    complete = valid.groupby("stem").method.nunique()
    complete = set(complete[complete == len(STRUCTURE_NAMES)].index)
    exclusions = targets[~targets.stem.isin(complete)].copy()
    exclusions["reason"] = "Absent or unsuccessful in archived exp250 structure complete-case table"
    raw = valid[valid.stem.isin(complete)]
    base = raw[["stem", "method", "gdt_ts", "lddt", "mean_plddt", "source", "source_row"]]
    base = base.melt(id_vars=["stem", "method", "mean_plddt", "source", "source_row"],
                     value_vars=["gdt_ts", "lddt"], var_name="metric", value_name="value")
    for method in ("af2", "af3", "boltz2"):
        additional = sources.csv(DATA / f"{method}_structure_metrics.csv")
        if additional.stem.duplicated().any() or set(additional.stem) != set(targets.stem):
            raise ValueError(f"Incomplete {method} structural baseline")
        additional = additional[additional.stem.isin(complete)].copy()
        additional = additional.assign(mean_plddt=np.nan).melt(
            id_vars=["stem", "method", "mean_plddt", "source", "source_row"],
            value_vars=["gdt_ts", "lddt"], var_name="metric", value_name="value")
        base = pd.concat([base, additional], ignore_index=True)
    return annotate(base, targets), exclusions


def prepare_contacts(sources: Sources, targets: pd.DataFrame) -> pd.DataFrame:
    """Join the fixed latest checkpoint to the archived structure/KNN contact baselines."""
    raw = sources.csv(REPO / EXP277 / "eval_rollout_v2/figure_per_protein.csv")
    raw = raw[raw.predictor.isin(CONTACT_NAMES)].copy()
    raw["method"] = raw.predictor.map(CONTACT_NAMES)
    raw["metric"] = "r_precision"
    if raw.duplicated(["stem", "method"]).any() or len(raw) != 116 * len(CONTACT_NAMES):
        raise ValueError("Incomplete or duplicate archived contact score matrix")
    columns = ["stem", "method", "metric", "value", "source", "source_row"]
    parts = [raw[columns]]
    if (DATA / "test_contact_metrics.csv").exists():
        new = sources.csv(DATA / "test_contact_metrics.csv")
        new = new[(new.cut == "R") & new["range"].isin(["all", "long"])].copy()
        new["method"] = "marinfold"
        new["metric"] = new["range"].map({"all": "r_precision", "long": "r_precision_long"})
        parts.append(new.rename(columns={"precision": "value"})[columns])
        baseline = sources.csv(DATA / "inputs/baseline_contact_metrics.csv")
        baseline["method"] = baseline.predictor.map(CONTACT_NAMES)
        baseline["metric"] = baseline["range"].map({"all": "r_precision", "long": "r_precision_long"})
        test = set(targets.loc[targets.eval_set == "eval-test", "stem"])
        baseline = baseline[(baseline["range"] == "long") | baseline.stem.isin(test)]
        parts.append(baseline.rename(columns={"precision": "value"})[columns])
        latest = sources.csv(REPO / EXP277 / "eval_rollout_v2/contact_precision_all.csv")
        latest = latest[(latest.dataset == "foldbench_monomer") & (latest.cut == "R") & (latest["range"] == "long")].copy()
        latest["method"], latest["metric"] = "marinfold", "r_precision_long"
        parts.append(latest.rename(columns={"precision": "value"})[columns])
    for method in ("af2", "af3", "boltz2"):
        additional = sources.csv(DATA / f"{method}_contact_metrics.csv")
        additional = additional[(additional.cut == "R") & additional["range"].isin(["all", "long"])].copy()
        additional["metric"] = additional["range"].map({"all": "r_precision", "long": "r_precision_long"})
        parts.append(additional.rename(columns={"precision": "value"})[columns])
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["stem", "method", "metric"]).any():
        raise ValueError("Duplicate contact scores after joining publication inference")
    return annotate(result, targets)


def prepare_folding(sources: Sources, targets: pd.DataFrame, structure: pd.DataFrame) -> pd.DataFrame:
    """Read the latest model's exact top-L / top-zero confidence-selected samples.

    The upstream table already applies confidence selection among three diffusion
    samples at each fixed cut. We do not choose the cut or sample using accuracy.
    Intersect with the available external predictors for matched GDT-TS and lDDT.
    """
    raw = sources.csv(REPO / EXP311 / "per_target_selection.csv")
    raw = raw[raw.metric.isin(["gdt_ts", "lddt"])]
    parts = []
    for method, column in [("marinfold_helico", "top_l_confidence"), ("no_contacts", "top_zero_confidence")]:
        part = raw[["stem", "metric", column, "source", "source_row"]].rename(columns={column: "value"})
        parts.append(annotate(part.assign(method=method), targets))
    if (DATA / "helico_folding_samples.csv").exists():
        new = sources.csv(DATA / "helico_folding_samples.csv")
        selected = new.sort_values(["stem", "arm", "ranking_score", "sample_idx"], ascending=[True, True, False, True]).drop_duplicates(["stem", "arm"])
        selected = selected.assign(method=selected.arm.map({"top_L": "marinfold_helico", "top_0": "no_contacts"}))
        selected = selected.melt(id_vars=["stem", "method", "source", "source_row"], value_vars=["gdt_ts", "lddt"], var_name="metric", value_name="value")
        parts.append(annotate(selected, targets))
    baselines = structure[structure.method.isin(["af2", "af3", "boltz2", "protenix_msa", "protenix_ss", "esmfold", "esmfold2"])]
    combined = pd.concat([baselines, *parts], ignore_index=True)
    counts = combined.groupby(["stem", "metric"]).method.nunique()
    complete = counts[counts == len(FIGURES["05_folding"])].reset_index()[["stem", "metric"]]
    return combined.merge(complete, on=["stem", "metric"], validate="many_to_one")


def prepare_sampling(sources: Sources, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use exp321's ordinary iid arm on all 97 eval-val proteins, at N=100.

    Its 'heldout' name denotes 81 withheld eval-val proteins, NOT eval-test.
    Combine those with the 16 development proteins without arm selection.
    Invalid individual rollouts receive zero in the audited fixed-R score.
    """
    archived = pd.concat([sources.csv(REPO / EXP321 / name) for name in
                          ["full_dev_natural.csv", "heldout_natural.csv"]], ignore_index=True)
    audit = archived[(archived["mode"] == "full_iid_single") & (archived.N == 100)].copy()
    if (DATA / "test_sampling_metrics.csv").exists():
        audit = pd.concat([audit, sources.csv(DATA / "test_sampling_metrics.csv")], ignore_index=True)
    rows = []
    for method, column in [("single", "mean_validity_gated_rollout_r_precision"),
                           ("consensus", "consensus_r_precision"),
                           ("best100", "validity_gated_oracle_r_precision")]:
        part = audit.copy()
        part["method"], part["value"] = method, part[column]
        rows.append(part)
    result = pd.concat(rows, ignore_index=True)
    result["metric"] = result["range"].map({"all": "r_precision", "long": "r_precision_long"})
    if result.duplicated(["stem", "metric", "method"]).any() or len(result) not in (97 * 2 * 3, 314 * 2 * 3):
        raise ValueError("Expected 97 or 314 paired natural sampling results")
    result = annotate(result[["stem", "method", "metric", "value", "source", "source_row"]], targets)
    if not set(result.eval_set).issubset({"eval-val", "eval-test"}):
        raise ValueError("Sampling diagnostic must only use natural proteins")
    diagnostics = audit[audit["range"] == "all"][["stem", "true_union_recall", "mean_pairwise_jaccard",
                                                    "unique_maps", "invalid_rollouts", "finished", "source", "source_row"]]
    return result, diagnostics


def cohort_slices(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return explicitly separate populations; never pool designs with natural proteins."""
    return {
        "natural": frame[frame.designed == 0],
        "eval-val": frame[frame.eval_set == "eval-val"],
        "eval-test": frame[frame.eval_set == "eval-test"],
        "designed": frame[frame.designed == 1],
        "viral": frame[(frame.designed == 0) & (frame.is_viral == 1)],
        "nonviral": frame[(frame.designed == 0) & (frame.is_viral == 0)],
    }


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate each visible group, saving the exact protein membership."""
    records = []
    for (figure, metric), group in rows.groupby(["figure", "metric"], sort=True):
        for cohort, subset in cohort_slices(group).items():
            for tier in [*TIERS, "All depths"]:
                cut = subset if tier == "All depths" else subset[subset.tier == tier]
                if cut.empty:
                    continue
                # Every method in a panel must contribute exactly the same proteins.
                population = [set(x.stem) for _, x in cut.groupby("method")]
                if any(x != population[0] for x in population):
                    raise ValueError(f"Unpaired population in {figure}/{cohort}/{tier}")
                for method, data in cut.groupby("method", sort=True):
                    data = data.sort_values("stem")
                    mean, lo, hi = bootstrap(data.value.to_numpy())
                    records.append({"figure": figure, "metric": metric, "cohort": cohort,
                                    "tier": tier, "method": method, "n": len(data),
                                    "mean": mean, "ci_low": lo, "ci_high": hi,
                                    "stems": "|".join(data.stem)})
    return pd.DataFrame(records)


def paired_deltas(rows: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap within-protein differences, rather than subtracting unpaired intervals."""
    contrasts = [("02_oracle", "oracle", "protenix_msa"),
                 ("04_contacts", "marinfold", "knn"),
                 ("05_folding", "marinfold_helico", "no_contacts"),
                 ("05_folding", "marinfold_helico", "protenix_ss"),
                 ("06_sampling", "best100", "consensus")]
    records = []
    for figure, method, reference in contrasts:
        for metric, group in rows[rows.figure == figure].groupby("metric"):
            for cohort, subset in cohort_slices(group).items():
                for tier in [*TIERS, "All depths"]:
                    cut = subset if tier == "All depths" else subset[subset.tier == tier]
                    if cut.empty:
                        continue
                    table = cut.pivot(index="stem", columns="method", values="value").sort_index()
                    mean, lo, hi = bootstrap((table[method] - table[reference]).to_numpy())
                    records.append(dict(figure=figure, metric=metric, cohort=cohort, tier=tier,
                                        method=method, reference=reference, n=len(table),
                                        delta=mean, ci_low=lo, ci_high=hi))
    return pd.DataFrame(records)


def prepare_confidence(sources: Sources, targets: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compare each oracle against equal-budget controls within the same protein."""
    path = DATA / "helico_confidence_samples.csv"
    if not path.exists():
        return {}
    raw = sources.csv(path)
    if len(raw) != 660 or raw.duplicated(["stem", "arm", "map_seed", "sample_idx"]).any():
        raise ValueError("Expected 20 proteins × 11 maps × 3 diffusion samples")
    counts = raw.groupby(["stem", "arm", "map_seed"])
    if not (counts.size() == 3).all():
        raise ValueError("Unequal confidence-selection budgets")
    for _, group in raw.groupby("stem"):
        if any(group[column].nunique() != 1 for column in ("n_present", "n_absent", "n_unknown")):
            raise ValueError("Control information budgets differ")
    selected = raw.sort_values(["stem", "arm", "map_seed", "ranking_score", "sample_idx"],
                               ascending=[True, True, True, False, True]).drop_duplicates(["stem", "arm", "map_seed"])
    selected = annotate(selected.drop(columns=["eval_set", "L"]), targets)
    rows = []
    for stem, group in selected.groupby("stem"):
        oracle = group[group.arm == "oracle"].iloc[0]
        for arm in ("uniform", "separation_matched"):
            random = group[group.arm == arm]
            if len(random) != 5:
                raise ValueError("Need five randomized maps for each control")
            for score in ("ranking_score", "mean_plddt"):
                difference = oracle[score] - random[score].to_numpy()
                rows.append({"stem": stem, "tier": oracle.tier, "eval_set": oracle.eval_set,
                             "arm": arm, "confidence": score,
                             "oracle_confidence": oracle[score], "random_mean_confidence": random[score].mean(),
                             "win_rate": float(np.mean((difference > 0) + 0.5 * (difference == 0))),
                             "oracle_rank_of_six": 1 + int((difference < 0).sum()) + 0.5 * int((difference == 0).sum()),
                             "oracle_gdt_ts": oracle.gdt_ts, "random_mean_gdt_ts": random.gdt_ts.mean(),
                             "oracle_lddt": oracle.lddt, "random_mean_lddt": random.lddt.mean(),
                             "oracle_source_row": oracle.source_row,
                             "random_source_rows": "|".join(random.source_row.astype(str))})
    per_protein = pd.DataFrame(rows)
    records = []
    for tier in [*TIERS, "All depths"]:
        part = per_protein if tier == "All depths" else per_protein[per_protein.tier == tier]
        for (arm, confidence), group in part.groupby(["arm", "confidence"]):
            mean, lo, hi = bootstrap(group.win_rate.to_numpy())
            records.append(dict(tier=tier, arm=arm, confidence=confidence, metric="win_rate",
                                mean=mean, ci_low=lo, ci_high=hi, n=len(group)))
    return {"confidence_per_map.csv": selected, "confidence_per_protein.csv": per_protein,
            "confidence_summary.csv": pd.DataFrame(records)}


def main() -> None:
    """Build compact tables and a digest manifest for offline rendering."""
    DATA.mkdir(exist_ok=True)
    remote = remote_inputs()
    sources = Sources()
    targets = prepare_targets(sources)
    structure, exclusions = prepare_structure(sources, targets)
    contacts = prepare_contacts(sources, targets)
    folding = prepare_folding(sources, targets, structure)
    sampling, diagnostics = prepare_sampling(sources, targets)
    rows = []
    for figure, methods in FIGURES.items():
        base = {"04_contacts": contacts, "05_folding": folding, "06_sampling": sampling}.get(figure, structure)
        rows.append(base[base.method.isin(methods)].assign(figure=figure))
    rows = pd.concat(rows, ignore_index=True)
    if not np.isfinite(rows.value).all() or not rows.value.between(0, 1).all():
        raise ValueError("Invalid accuracy value")
    tables = {"targets.csv": targets, "structure_exclusions.csv": exclusions,
              "figure_rows.csv": rows, "summary.csv": summarize(rows),
              "paired_deltas.csv": paired_deltas(rows), "sampling_diagnostics.csv": diagnostics}
    tables.update(prepare_confidence(sources, targets))
    # Training mix is a source-token inventory, not a measured exposure count.
    tables["training_sources.csv"] = sources.csv(REPO / EXP277 / "epoch_corpus_counts.csv")
    coverage = []
    for figure, group in rows.groupby("figure"):
        available = set(group.stem)
        for row in targets.itertuples():
            coverage.append(dict(figure=figure, stem=row.stem, eval_set=row.eval_set, tier=row.tier,
                                 available=row.stem in available,
                                 reason="" if row.stem in available else
                                 "Publication inference not yet present" if row.eval_set == "eval-test" and figure in {"04_contacts", "06_sampling"} else
                                 "Sampling diagnostic covers natural proteins only" if figure == "06_sampling" else
                                 "Excluded by verified-coordinate or matched-structure coverage; see structure_exclusions.csv and helico_folding_inputs.json"))
    tables["coverage.csv"] = pd.DataFrame(coverage)
    for name, frame in tables.items():
        frame.to_csv(DATA / name, index=False)
    manifest = {
        "schema_version": 2,
        "input_checkout": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "prepare_sha256": sha256(Path(__file__)),
        "lock_sha256": sha256(HERE / "uv.lock"),
        "inputs": sources.records, "remote_inputs": remote,
        "generation_provenance": {
            str(path.relative_to(HERE)): {"sha256": sha256(path)}
            for path in [HERE / "generation/checkpoint_manifest.json",
                         DATA / "alphafold_inputs.json", DATA / "boltz2_inputs.json",
                         *sorted(DATA.glob("*_run.json")),
                         *sorted(DATA.glob("helico_*_inputs.json"))]
        },
        "outputs": {name: {"sha256": sha256(DATA / name), "rows": len(table)} for name, table in tables.items()},
        "bootstrap": {"unit": "protein", "draws": BOOTSTRAPS, "seed": SEED, "interval": "percentile 95%"},
        "msa": "Exact Protenix-v2 + MSA A3M sequence count, including query; not Neff. Fixed [1,10), [10,100), [100,1000), [1000,infinity) bins.",
        "main_checkpoint": "contacts-v1-exp277-m2-p06-full-epoch-1.5B-step-266344",
        "training_raw_tokens": 248583762834,
        "main_recipe": "exp277 fixed exp82 rollout+resample; 100 rollouts, T=1, top_p=0.95, top_k=-1; votes only",
        "sampling_checkpoint": "contacts-v1-exp277-m2-p06-full-epoch-1.5B-step-266344",
        "sampling_recipe": "exp321 ordinary iid control, N=100, on 97 eval-val proteins plus new 217 publication test proteins. Same pool within protein. Consensus uses all parsed maps; unfinished/malformed individual rollouts score zero. Main contact panel follows exp277 and omits unfinished maps from votes.",
        "oracle_contacts": "Full ground-truth three-state map, including non-contacts; information upper bound, not a predictor",
        "sampling_oracle": "Max correct contacts among first R emitted pairs / R; short sets retain denominator R. Ground-truth selection.",
        "scope": "Publication reanalysis plus explicitly authorized inference on the fixed eval-test split; no model, cut-count, or sampling-setting selection on test.",
        "confidence_control": {"status": "complete" if "confidence_summary.csv" in tables else "pending", "selection": "20 preselected natural proteins, five per MSA tier; 11 maps and 3 diffusion samples per map"},
        "alphafold_protocol": json.loads((DATA / "alphafold_inputs.json").read_text()),
        "boltz2_protocol": json.loads((DATA / "boltz2_inputs.json").read_text()),
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(rows)} contributing rows for exp277 step 266344. No predictor inference.")


if __name__ == "__main__":
    main()
