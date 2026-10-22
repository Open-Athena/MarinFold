"""Compare unconditioned MarinFold support for AFDB and experimental-only contacts."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import fsspec
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_summary import save_plot_with_meta
from protocol import remaining_metrics

HERE = Path(__file__).resolve().parent


def exclusive_preference(scores: np.ndarray, af_only: list, pdb_only: list) -> float | None:
    """Return probability that a PDB-only contact outranks an AFDB-only contact.

    This within-model contrast treats the two contradictory label sets equally,
    irrespective of their sizes; ties receive half credit. It is distinct from
    comparing full-map R-precision with different R or from cross-model AUC.
    """
    if not af_only or not pdb_only:
        return None
    af = np.array([scores[i, j] for i, j in af_only])
    pdb = np.array([scores[i, j] for i, j in pdb_only])
    return float(((pdb[:, None] > af[None, :]) + .5 * (pdb[:, None] == af[None, :])).mean())


def main() -> None:
    """Require complete predictions for the frozen case cohort before comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=HERE / "data/afdb_inference_plan.json")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    raw = args.plan.read_bytes()
    plan = json.loads(raw)
    monomers = plan["eval_set"] == "afdb-training-monomers"
    prefix = "monomer" if monomers else "afdb"
    plot_dir = HERE / "plots" if monomers else HERE / "plots/superseded_complexes"
    digest = hashlib.sha256(raw).hexdigest()
    fs, root = fsspec.core.url_to_fs(args.run)
    rows, timings, ensemble_rows = [], [], []
    for target in plan["targets"]:
        matrices = []
        gated_matrices = []
        for repeat in range(plan["repeats"]):
            base = f"{root}/units/{target['stem']}/random-r{repeat}-k0000"
            with fs.open(base + ".complete.json") as stream:
                marker = json.load(stream)
            if marker["plan_sha256"] != digest or marker["n_rollouts"] != 100:
                raise ValueError(f"Invalid case provenance: {base}")
            if marker["completion_policy"] != "retain_all_samples_count_generation_errors":
                raise ValueError(f"Unexpected sample selection policy: {base}")
            with fs.open(base + ".npz", "rb") as stream:
                payload = stream.read()
            if hashlib.sha256(payload).hexdigest() != marker["files_sha256"][".npz"]:
                raise ValueError("Case vote checksum mismatch")
            with np.load(io.BytesIO(payload)) as archive:
                matrices.append(archive["score"].astype(float) / 100)
                gated_matrices.append(archive["score_valid_rollouts"].astype(float) / 100)
            timings.append({k: v for k, v in marker.items() if k != "files_sha256"})
        for repeat, scores in enumerate(matrices):
            af = remaining_metrics(scores, target["afdb_pairs"], target["resolved"], [])
            pdb = remaining_metrics(scores, target["pdb_pairs"], target["resolved"], [])
            gated_af = remaining_metrics(gated_matrices[repeat], target["afdb_pairs"], target["resolved"], [])
            gated_pdb = remaining_metrics(gated_matrices[repeat], target["pdb_pairs"], target["resolved"], [])
            mean_af = float(np.mean([scores[i, j] for i, j in target["afdb_only"]]))
            mean_pdb = float(np.mean([scores[i, j] for i, j in target["pdb_only"]]))
            rows.append({"uniprot": target["uniprot"], "pdb": target["pdb"], "chain": target["chain"], "repeat": repeat,
                         "afdb_r_precision": af["r_precision"], "pdb_r_precision": pdb["r_precision"],
                         "afdb_only_mean_vote": mean_af, "pdb_only_mean_vote": mean_pdb,
                         "exclusive_vote_delta_pdb_minus_afdb": mean_pdb - mean_af,
                         "pdb_only_ranking_preference": exclusive_preference(scores, target["afdb_only"], target["pdb_only"]),
                         "validity_gated_afdb_r_precision": gated_af["r_precision"],
                         "validity_gated_pdb_r_precision": gated_pdb["r_precision"],
                         "validity_gated_pdb_only_ranking_preference": exclusive_preference(gated_matrices[repeat], target["afdb_only"], target["pdb_only"]),
                         "interpretation": target["interpretation"]})
            for conformer in target.get("pdb_ensemble", []):
                metrics = remaining_metrics(scores, conformer["pdb_pairs"], target["resolved"], [])
                af_vote = float(np.mean([scores[i, j] for i, j in conformer["afdb_only"]]))
                pdb_vote = float(np.mean([scores[i, j] for i, j in conformer["pdb_only"]]))
                ensemble_rows.append({"uniprot": target["uniprot"], "repeat": repeat, "model": conformer["model"],
                                      "afdb_r_precision": af["r_precision"], "pdb_r_precision": metrics["r_precision"],
                                      "afdb_only_mean_vote": af_vote, "pdb_only_mean_vote": pdb_vote,
                                      "exclusive_vote_delta_pdb_minus_afdb": pdb_vote - af_vote,
                                      "pdb_only_ranking_preference": exclusive_preference(scores, conformer["afdb_only"], conformer["pdb_only"])})
    pd.DataFrame(rows).to_csv(HERE / f"data/{prefix}_marinfold_per_repeat.csv", index=False)
    pd.DataFrame(timings).to_csv(HERE / f"data/{prefix}_timings.csv", index=False)
    if ensemble_rows:
        pd.DataFrame(ensemble_rows).to_csv(HERE / f"data/{prefix}_nmr_preference.csv", index=False)
    summary = pd.DataFrame(rows).groupby(["uniprot", "pdb", "chain", "interpretation"], as_index=False).mean(numeric_only=True).drop(columns="repeat")
    summary.to_csv(HERE / f"data/{prefix}_marinfold_preference.csv", index=False)
    ordered = summary.set_index("uniprot").loc[[target["uniprot"] for target in plan["targets"]]]
    positions = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    for offset, label, color, suffix in ((-.18, "AFDB v4", "#CC6B25", "afdb"),
                                         (.18, "Later PDB", "#167CA0", "pdb")):
        axes[0].bar(positions + offset, ordered[f"{suffix}_only_mean_vote"], width=.34, color=color, label=label)
        axes[1].bar(positions + offset, ordered[f"{suffix}_r_precision"], width=.34, color=color, label=label)
        for index, accession in enumerate(ordered.index):
            repeats = [row for row in rows if row["uniprot"] == accession]
            for ax, field in zip(axes, (f"{suffix}_only_mean_vote", f"{suffix}_r_precision")):
                ax.scatter(np.full(len(repeats), index + offset), [row[field] for row in repeats],
                           s=13, facecolors="white", edgecolors="black", linewidths=.6, zorder=3)
    for ax in axes:
        ax.set_xticks(positions, [f"{target['protein_name'] if monomers else target['uniprot']}\n{target['pdb'].upper()} chain {target['chain']}" for target in plan["targets"]], fontsize=9)
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        if not monomers:
            ax.axvline(2.5, color="gray", linestyle=":", linewidth=1)
    if ensemble_rows:
        ensemble = pd.DataFrame(ensemble_rows).groupby(["uniprot", "model"]).mean(numeric_only=True)
        for index, accession in enumerate(ordered.index):
            if accession not in ensemble.index.get_level_values("uniprot"):
                continue
            for offset, suffix in ((-.18, "afdb"), (.18, "pdb")):
                for ax, field in zip(axes, (f"{suffix}_only_mean_vote", f"{suffix}_r_precision")):
                    values = ensemble.loc[accession, field]
                    ax.vlines(index + offset, values.min(), values.max(), color="black", linewidth=1.7, zorder=4)
                    ax.hlines([values.min(), values.max()], index + offset - .055, index + offset + .055, color="black", linewidth=1.7, zorder=4)
    axes[0].set_ylabel("Mean rollout vote on exclusive contacts")
    axes[1].set_ylabel("R-precision against each structure")
    axes[0].legend(frameon=False)
    title = "Verified monomer comparisons: moderate AFDB/PDB disagreements" if monomers else "Does MarinFold favor AFDB training contacts or later experimental contacts?"
    fig.suptitle(title + "\nSequence only • exp277 step 266344 • two sets of 100 rollouts", fontsize=13)
    if monomers:
        fig.supxlabel("Pru p 7 whiskers: range across 20 NMR conformers (not confidence intervals). White dots: rollout repeats.", fontsize=9)
    context = ("These are isolated biological monomers, not established catastrophic fold errors. Pru p 7 bars use deposited model 1; black whiskers span the 20-conformer NMR ensemble after averaging the two repeats, not confidence intervals."
               if monomers else "The first three cases have large assembly-bound discrepancies; the others are contact/state controls.")
    save_plot_with_meta(fig, str(plot_dir / f"{prefix}_marinfold_preference.png"),
                        caption="Bars average two 100-rollout repeats; white dots show each repeat. Left: mean support over contacts unique to each structure, on common resolved residues. Right: R-precision against each contact set, whose sizes differ. " + context + " Predictions have no structure contacts in their prompts.")
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
