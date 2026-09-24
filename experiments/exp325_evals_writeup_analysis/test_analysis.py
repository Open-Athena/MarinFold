"""Check scientific invariants across the real, archived input tables."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plan_missing import randomized_map
from prepare import depth_tier

DATA = Path(__file__).resolve().parent / "data"


def test_exact_msa_boundaries() -> None:
    assert depth_tier(pd.Series([1, 9, 10, 99, 100, 999, 1000])).tolist() == [
        "<10", "<10", "10–99", "10–99", "100–999", "100–999", "≥1000"]


def test_latest_checkpoint_and_authorized_test_split() -> None:
    rows = pd.read_csv(DATA / "figure_rows.csv")
    contacts = rows[(rows.figure == "04_contacts") & (rows.method == "marinfold") & (rows.metric == "r_precision")]
    val = contacts[contacts.eval_set == "eval-val"]
    assert len(val) == 97
    # Independently published exp277 headline, not derived from our summary.
    assert val.value.mean() == pytest.approx(0.5537468341913647, abs=1e-8)
    assert (contacts.eval_set == "eval-test").sum() == 217
    assert ((contacts.designed == 0) & (contacts.msa_depth < 10)).sum() == 5
    assert not rows.source.str.contains("exp254").any()
    assert not rows[(rows.method == "marinfold") | (rows.method == "marinfold_helico")].source.str.contains("exp250").any()


def test_matched_methods_and_explicit_missing_bins() -> None:
    rows = pd.read_csv(DATA / "figure_rows.csv")
    for _, group in rows.groupby(["figure", "metric"]):
        populations = [set(g.stem) for _, g in group.groupby("method")]
        assert all(p == populations[0] for p in populations)
        assert not group.duplicated(["stem", "method"]).any()
    summary = pd.read_csv(DATA / "summary.csv")
    latest = summary[(summary.figure == "04_contacts") & (summary.cohort == "natural")]
    assert "<10" in set(latest.tier)
    assert (latest[latest.tier == "<10"].n == 5).all()
    assert (summary.ci_low <= summary["mean"]).all()
    assert (summary["mean"] <= summary.ci_high).all()


def test_same_pool_sampling_and_fixed_denominator() -> None:
    rows = pd.read_csv(DATA / "figure_rows.csv")
    x = rows[(rows.figure == "06_sampling") & (rows.metric == "r_precision")]
    per_target = x.pivot(index="stem", columns="method", values="value")
    assert len(per_target) == 314
    val_stems = x.loc[x.eval_set == "eval-val", "stem"].unique()
    val = per_target.loc[val_stems]
    assert (per_target.best100 >= per_target.single).all()
    assert val.consensus.mean() == pytest.approx(0.5526335674, abs=1e-8)
    assert val.best100.mean() == pytest.approx(0.5243089658, abs=1e-8)
    assert set(x.eval_set) == {"eval-val", "eval-test"}


def test_source_row_roundtrip() -> None:
    rows = pd.read_csv(DATA / "figure_rows.csv")
    root = DATA.parents[2]
    for source, group in rows.groupby("source"):
        raw = pd.read_csv(root / source)
        selected = raw.iloc[group.source_row.astype(int)]
        identifier = "target_id" if "target_id" in raw else "stem"
        assert selected[identifier].tolist() == group.stem.tolist()


@pytest.mark.parametrize("match_separation", [False, True])
def test_random_control_preserves_information_budget(match_separation: bool) -> None:
    original = np.zeros((40, 40), dtype=np.uint8)
    i, j = np.triu_indices(40, k=6)
    # Leave a disconnected unresolved residue unknown in both arms.
    keep = (i != 12) & (j != 12)
    i, j = i[keep], j[keep]
    original[i, j] = original[j, i] = 1
    original[i[::7], j[::7]] = original[j[::7], i[::7]] = 2
    before = original.copy()
    result = randomized_map(original, seed=325, match_separation=match_separation)
    assert np.array_equal(original, before)  # no mutation of the oracle
    assert np.array_equal(result, result.T)
    assert np.array_equal(result == 0, original == 0)
    assert (result == 2).sum() == (original == 2).sum()
    assert not np.array_equal(result, original)
    if match_separation:
        bins = np.digitize(j - i, [12, 24])
        for group in range(3):
            assert (result[i[bins == group], j[bins == group]] == 2).sum() == (original[i[bins == group], j[bins == group]] == 2).sum()


def test_random_control_rejects_wrong_encoding() -> None:
    with pytest.raises(ValueError, match="0=unknown"):
        randomized_map(np.full((4, 4), -1), seed=0, match_separation=False)


def test_random_control_uses_residue_coordinates_with_extra_atom_tokens() -> None:
    # Unknown ligand tokens can sit between protein residues. Token distance
    # would put the first pair in the wrong separation bin.
    positions = np.array([0, 7, *([0] * 20), 23, 30])
    state = np.zeros((24, 24), dtype=np.uint8)
    for i, j, value in [(0, 1, 2), (0, 22, 1), (1, 22, 2), (0, 23, 2), (1, 23, 1), (22, 23, 1)]:
        state[i, j] = state[j, i] = value
    result = randomized_map(state, 325, True, positions)
    i, j = np.where(np.triu(state != 0, 1))
    bins = np.digitize(abs(positions[j] - positions[i]), [12, 24])
    for group in range(3):
        mask = bins == group
        assert (result[i[mask], j[mask]] == 2).sum() == (state[i[mask], j[mask]] == 2).sum()


def test_confidence_comparison_has_equal_information_and_sample_budgets() -> None:
    raw = pd.read_csv(DATA / "helico_confidence_samples.csv")
    assert len(raw) == 660
    assert (raw.groupby(["stem", "arm", "map_seed"]).size() == 3).all()
    assert (raw.groupby("stem")[["n_present", "n_absent", "n_unknown"]].nunique() == 1).all().all()
    prepared = pd.read_csv(DATA / "confidence_per_map.csv")
    assert len(prepared) == 220
    assert prepared.mean_plddt.notna().all()
    assert (prepared.groupby("tier").stem.nunique() == 5).all()


def test_prompt_length_is_distinct_from_resolved_msa_query_length() -> None:
    target = pd.read_csv(DATA / "targets.csv").set_index("stem").loc["8q79_A"]
    assert target.L == 236
    assert target.msa_query_length == 215


def test_structured_decoys_keep_every_seed_and_equal_helico_budgets() -> None:
    raw = pd.read_csv(DATA / "helico_structured_samples.csv", float_precision="round_trip")
    maps = pd.read_csv(DATA / "structured_decoy_maps.csv")
    selected = pd.read_csv(DATA / "structured_confidence_per_map.csv", float_precision="round_trip")
    assert len(raw) == 1515 and len(maps) == len(selected) == 505
    assert raw.groupby(["stem", "arm", "map_seed"]).sample_idx.apply(lambda x: set(x) == {0, 1, 2}).all()
    assert set(selected.eval_set) == {"eval-test"}
    assert selected.msa_depth.between(1, 9).all() and (selected.designed == 0).all()
    for stem, group in maps.groupby("stem"):
        assert len(group) == 101
        assert set(group[group.arm == "esmfold2"].map_seed) == set(range(100))
        assert group.n_unknown.nunique() == 1
        oracle = group[group.arm == "oracle"].iloc[0]
        assert oracle.contact_precision == oracle.contact_recall == oracle.contact_jaccard == 1
    roundtrip = raw.iloc[selected.source_row.astype(int)]
    assert roundtrip.stem.tolist() == selected.stem.tolist()
    np.testing.assert_allclose(roundtrip.ranking_score, selected.ranking_score, rtol=0, atol=0)
    maxima = raw.groupby(["stem", "arm", "map_seed"]).ranking_score.max()
    np.testing.assert_allclose(selected.set_index(["stem", "arm", "map_seed"]).ranking_score.sort_index(), maxima.sort_index())


def test_structured_oracle_rank_and_source_rows_match_sorted_pool() -> None:
    maps = pd.read_csv(DATA / "structured_confidence_per_map.csv", float_precision="round_trip")
    ranks = pd.read_csv(DATA / "structured_confidence_ranks.csv", float_precision="round_trip")
    assert len(ranks) == 10
    for rank in ranks.itertuples():
        group = maps[maps.stem == rank.stem]
        ordered = sorted(group[rank.confidence], reverse=True)
        positions = [i + 1 for i, value in enumerate(ordered) if value == rank.oracle_confidence]
        assert rank.rank_best == min(positions) and rank.rank_worst == max(positions)
        assert rank.rank_mid == np.mean(positions)
        decoys = group[group.arm == "esmfold2"]
        assert rank.unique_decoy_maps == decoys.map_sha256.nunique()
        assert set(map(int, rank.decoy_source_rows.split("|"))) == set(decoys.source_row)


def test_scatter_uses_original_esmfold2_accuracy_and_explicit_oracle_reference() -> None:
    joined = pd.read_csv(DATA / "structured_accuracy_confidence.csv", float_precision="round_trip")
    raw = pd.read_csv(DATA / "esmfold2_decoy_structure_metrics.csv", float_precision="round_trip")
    predictions = joined[joined.arm == "esmfold2"]
    original = raw.iloc[predictions.accuracy_source_row.astype(int)]
    assert len(predictions) == len(original) == 500
    assert original.stem.tolist() == predictions.stem.tolist()
    assert original.map_seed.tolist() == predictions.map_seed.tolist()
    assert original.structure_sha256.tolist() == predictions.structure_sha256.tolist()
    np.testing.assert_allclose(original.gdt_ts, predictions.source_structure_gdt_ts, rtol=0, atol=0)
    np.testing.assert_allclose(original.lddt, predictions.source_structure_lddt, rtol=0, atol=0)
    oracle = joined[joined.arm == "oracle"]
    assert len(oracle) == 5 and oracle.accuracy_source_row.isna().all()
    assert (oracle[["source_structure_gdt_ts", "source_structure_lddt"]] == 1).all().all()
    assert (oracle.accuracy_rule == "experimental reference compared with itself").all()


def test_test_folding_is_complete_and_selected_only_by_confidence() -> None:
    raw = pd.read_csv(DATA / "helico_folding_samples.csv")
    assert raw.stem.nunique() == 211
    assert len(raw) == 211 * 2 * 3
    assert (raw.groupby(["stem", "arm"]).size() == 3).all()
    rows = pd.read_csv(DATA / "figure_rows.csv")
    selected = rows[(rows.figure == "05_folding") & (rows.metric == "gdt_ts") &
                    (rows.method == "marinfold_helico") & (rows.eval_set == "eval-test")]
    assert len(selected) == 210
    for row in selected.itertuples():
        original = raw.iloc[row.source_row]
        candidates = raw[(raw.stem == row.stem) & (raw.arm == "top_L")]
        assert original.ranking_score == candidates.ranking_score.max()
        assert row.value == pytest.approx(original.gdt_ts)


def test_added_baseline_budgets_and_complete_coverage() -> None:
    protocol = json.loads((DATA / "alphafold_inputs.json").read_text())
    assert protocol["n_targets"] == 333
    targets = set(pd.read_csv(DATA / "targets.csv").stem)
    for method, budget in [("af2", 5), ("af3", 25), ("boltz2", 25)]:
        timings = pd.read_csv(DATA / f"{method}_timings.csv")
        assert set(timings.stem) == targets
        assert not timings.stem.duplicated().any()
        assert (timings.n_samples == budget).all()
        run = json.loads((DATA / f"{method}_run.json").read_text())
        assert run["n_candidates"] == budget * 333
        assert set(run["selected_sha256"]) == targets
        metrics = pd.read_csv(DATA / f"{method}_contact_metrics.csv")
        for region in ("all", "long"):
            rows = metrics[(metrics.cut == "R") & (metrics["range"] == region)]
            assert len(rows) == 333
            assert set(rows.stem) == targets


def test_added_baseline_preserves_original_populations() -> None:
    rows = pd.read_csv(DATA / "figure_rows.csv")
    for figure, expected in [("01_predictors", 305), ("02_oracle", 305),
                             ("04_contacts", 314), ("05_folding", 305)]:
        x = rows[(rows.figure == figure) & (rows.designed == 0)]
        assert {"af2", "af3", "boltz2"}.issubset(set(x.method))
        assert (x.groupby(["method", "metric"]).stem.nunique() == expected).all()


def test_boltz2_fixed_recipe_and_recorded_msa_processing() -> None:
    protocol = json.loads((DATA / "boltz2_inputs.json").read_text())
    assert (protocol["seed"], protocol["diffusion_samples"], protocol["recycling_steps"]) == (42, 25, 10)
    assert not any(protocol[k] for k in ("templates", "constraints", "affinity", "use_potentials", "subsample_msa"))
    timings = pd.read_csv(DATA / "boltz2_timings.csv")
    assert (timings.n_msa_sequences_processed >= 1).all()
    assert (timings.n_msa_sequences_processed <= np.minimum(8192, timings.msa_depth)).all()
    assert (timings.elapsed_seconds > 0).all()
    assert (timings.total_seconds >= timings.elapsed_seconds + timings.model_load_seconds).all()
    assert (timings.runner_tag == "modal-us-east").all()
