"""Validate exact-L/2 or L/5 folding and cache paired comparisons for the depth figure."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import TIERS, bootstrap

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
METRICS = ("gdt_ts", "lddt", "tm_score")


def main() -> None:
    """Audit raw maps/coordinates, select by confidence, and bootstrap proteins."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--divisor", type=int, choices=(2, 5), default=2)
    args = parser.parse_args()
    divisor = args.divisor
    phase = f"marinfold_l{divisor}"
    method = f"marinfold_helico_L{divisor}"
    arm = f"top_L{divisor}"
    references = ["marinfold_helico", "oracle_L2", "no_contacts"]
    if divisor == 5:
        references.append("marinfold_helico_L2")
    names = [f"{phase}_protocol.json", f"{phase}_targets.csv", f"{phase}_maps.csv",
             f"helico_{phase}_samples.csv", f"helico_{phase}_timings.csv", f"helico_{phase}_run.json",
             "oracle_l2_per_protein.csv", f"{phase}_baselines.csv", "helico_oracle_l2_run.json"]
    hashes = {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in names}
    targets = pd.read_csv(DATA / names[1])
    maps = pd.read_csv(DATA / names[2])
    samples = pd.read_csv(DATA / names[3], float_precision="round_trip")
    timings = pd.read_csv(DATA / names[4])
    protocol = json.loads((DATA / names[0]).read_text())
    run = json.loads((DATA / names[5]).read_text())
    if (run["contact_checkpoint"] != protocol["model"] or run["msa"]
        or run["helico_checkpoint_step"] != 6000 or run["seed_per_map"] != 42
        or run["n_diffusion_samples_per_cut"] != 3 or run["n_trunk_recycles"] != 6):
        raise ValueError("Predictor identity or inference recipe changed")
    oracle_run = json.loads((DATA / "helico_oracle_l2_run.json").read_text())
    for field in ("helico_source_sha", "helico_checkpoint_sha256"):
        if run[field] != oracle_run[field]:
            raise ValueError(f"Helico identity differs from oracle comparison: {field}")
    if run["oracle_budget_protocol_sha256"] != hashes[f"{phase}_protocol.json"]:
        raise ValueError("Inference used a different frozen protocol")
    if len(targets) != 305 or targets.stem.duplicated().any() or targets.designed.any():
        raise ValueError("Expected all 305 natural proteins")
    expected = {(stem, arm, 0, sample) for stem in targets.stem for sample in range(3)}
    keys = ["stem", "arm", "map_seed", "sample_idx"]
    if len(samples) != 915 or set(samples[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Incomplete or unexpected inference samples")
    if len(timings) != 305 or timings.stem.duplicated().any() or not (timings.elapsed_seconds > 0).all():
        raise ValueError("Incomplete inference timings")
    if len(maps) != 305 or maps.stem.duplicated().any() or not maps.n_present.eq(maps.L // divisor).all() or maps.n_absent.any():
        raise ValueError("Wrong frozen contact budget")
    checked = samples.merge(maps, on=keys[:3], suffixes=("", "_frozen"), validate="many_to_one")
    for field in ("L", "n_present", "n_absent", "n_unknown", "requested_contacts"):
        if not checked[field].eq(checked[f"{field}_frozen"]).all():
            raise ValueError(f"Wrong runtime {field}")
    if not np.isfinite(samples[[*METRICS, "ranking_score", "ptm"]]).all().all():
        raise ValueError("Nonfinite structural outcomes")
    with tarfile.open(HERE / f"scratch/{phase}_coordinates.tar.gz") as archive:
        for row in maps.itertuples():
            prefix = f"{phase}/{row.stem}/{arm}-0"
            with np.load(io.BytesIO(archive.extractfile(prefix + ".npz").read())) as raw:
                if hashlib.sha256(raw["contact_state"].tobytes()).hexdigest() != row.state_sha256:
                    raise ValueError(f"{row.stem}: raw conditioning differs")
                if raw["coords"].shape[0] != 3 or not np.isfinite(raw["coords"]).all():
                    raise ValueError(f"{row.stem}: invalid raw coordinates")
            record = json.load(archive.extractfile(prefix + ".json"))
            scored = samples[samples.stem == row.stem].sort_values("sample_idx")
            for metric in (*METRICS, "ranking_score", "ptm"):
                if not np.array_equal(scored[metric], [sample[metric] for sample in record["samples"]]):
                    raise ValueError(f"{row.stem}: scores differ from durable raw results")
    samples["source_row"] = np.arange(len(samples))
    selections = []
    for selector in ("ranking_score", "ptm"):
        selected = samples.sort_values(["stem", selector, "sample_idx"], ascending=[True, False, True]).drop_duplicates("stem")
        selections.append(selected.assign(selector=selector, source=f"helico_{phase}_samples.csv"))
    selected = pd.concat(selections, ignore_index=True)
    selected.to_csv(DATA / f"{phase}_selected.csv", index=False)
    protein = selected.melt(id_vars=["stem", "selector", "source", "source_row", "sample_idx", "n_present"],
                            value_vars=list(METRICS), var_name="metric", value_name="value")
    protein = protein.assign(method=method).merge(
        targets[["stem", "eval_set", "msa_depth", "tier", "L_exp245"]], on="stem", validate="many_to_one")
    protein.to_csv(DATA / f"{phase}_per_protein.csv", index=False)
    # Keep the earlier top-L and oracle arms frozen; do not reinterpret their selectors.
    old = pd.read_csv(DATA / f"{phase}_baselines.csv")
    if len(old) != 305 * 2 * len(references) or set(old.stem) != set(targets.stem) or old.duplicated(["stem", "method", "metric"]).any():
        raise ValueError("Frozen baselines do not match the new cohort")
    old = old.assign(selector="ranking_score")
    comparison = pd.concat([protein, old], ignore_index=True)
    summaries, deltas = [], []
    for cohort in ("natural", "eval-val", "eval-test"):
        cohort_rows = comparison if cohort == "natural" else comparison[comparison.eval_set == cohort]
        for tier in [*TIERS, "All depths"]:
            cut = cohort_rows if tier == "All depths" else cohort_rows[cohort_rows.tier == tier]
            for (selector, metric, summary_method), group in cut.groupby(["selector", "metric", "method"]):
                mean, lo, hi = bootstrap(group.value.to_numpy())
                summaries.append(dict(cohort=cohort, tier=tier, selector=selector, metric=metric, method=summary_method,
                    n=len(group), mean=mean, ci_low=lo, ci_high=hi))
            for metric in ("gdt_ts", "lddt"):
                paired = cut[(cut.selector == "ranking_score") & (cut.metric == metric)].pivot(index="stem", columns="method", values="value")
                if paired.empty:
                    continue
                if paired.isna().any().any():
                    raise ValueError("Unmatched budget comparison")
                for reference in references:
                    difference = paired[method] - paired[reference]
                    mean, lo, hi = bootstrap(difference.to_numpy())
                    deltas.append(dict(cohort=cohort, tier=tier, metric=metric, reference=reference,
                        n=len(paired), delta=mean, ci_low=lo, ci_high=hi))
    pd.DataFrame(summaries).to_csv(DATA / f"{phase}_summary.csv", index=False)
    pd.DataFrame(deltas).to_csv(DATA / f"{phase}_deltas.csv", index=False)
    outputs = [f"{phase}_selected.csv", f"{phase}_per_protein.csv", f"{phase}_summary.csv", f"{phase}_deltas.csv"]
    manifest = dict(protocol=protocol, sources=hashes, files={name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in outputs},
                    audit="All 305 maps, 915 coordinate sets and raw metric rows checked", n_targets=305, n_maps=305, n_samples=915)
    (DATA / f"{phase}_analysis.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pd.DataFrame(summaries).query("cohort == 'natural' and metric == 'gdt_ts' and selector == 'ranking_score'").pivot(index="tier", columns="method", values="mean").to_string())


if __name__ == "__main__":
    main()
