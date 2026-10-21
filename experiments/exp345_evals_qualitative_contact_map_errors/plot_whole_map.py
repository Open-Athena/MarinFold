"""Plot the predeclared whole-map probe and the original eight case studies."""

import json
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import HERE
from analyze_whole_map import map_metrics
from build_summary import save_plot_with_meta
from plot_results import CASES, draw_map

METHODS = [
    ("pooled_200", "Pooled top-R (200)"),
    ("sample_mean_200", "Mean individual"),
    ("cross_pool_medoid", "Independent consensus"),
    ("mean_token_logprob", "Mean log-probability"),
    ("sample_oracle_100", "Oracle sample (100)"),
    ("sample_oracle_200", "Oracle sample (200)"),
    ("cluster_dominant_k4", "Largest cluster (k=4)"),
    ("cluster_oracle_k4", "Oracle cluster (k=4)"),
]


def main() -> None:
    """Save aggregate comparisons and pooled / oracle / consensus / truth maps."""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    results = pd.read_csv(HERE / "data/whole_map_methods.csv")
    summary = pd.read_csv(HERE / "data/whole_map_summary.csv")
    case_rows = pd.read_csv(HERE / "data/whole_map_case_contacts.csv.gz")
    case_metadata = json.loads((HERE / "data/whole_map_case_metadata.json").read_text())
    all_f1 = results[results["range"].eq("all")].groupby(["stem", "method"]).f1.mean().unstack()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for offset, region, color in ((-.14, "all", "#233d69"), (.14, "long", "#087e83")):
        sub = summary[summary["range"].eq(region) & summary.metric.eq("f1")].set_index("method").loc[[m for m, _ in METHODS]]
        axes[0].errorbar(sub["mean"], np.arange(len(METHODS))+offset,
                        xerr=[sub["mean"]-sub.ci_low, sub.ci_high-sub["mean"]],
                        fmt="o", capsize=3, color=color, label="All (≥6)" if region=="all" else "Long (≥24)")
    axes[0].set(yticks=np.arange(len(METHODS)), yticklabels=[label for _, label in METHODS],
                xlabel="Mean full-map F1 · 95% protein-bootstrap interval", xlim=(0, 1), title="Oracle gains are diagnostic, not deployable")
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left")
    axes[1].scatter(all_f1.pooled_200, all_f1.sample_oracle_200, c="#233d69", alpha=.65, s=23)
    axes[1].plot([0, 1], [0, 1], "--", color="#b9bdc1")
    for stem in chain.from_iterable(CASES):
        axes[1].annotate(stem, (all_f1.loc[stem, "pooled_200"], all_f1.loc[stem, "sample_oracle_200"]),
                         xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1].set(xlabel="Pooled top-R F1 (200 samples)", ylabel="Best complete individual F1 (oracle of 200)",
                xlim=(0, 1), ylim=(0, 1), title="Does a better whole map exist among the samples?")
    save_plot_with_meta(fig, HERE / "plots/whole_map_diagnostics.png", caption="97 eval-val proteins, 200 samples each. Means weight proteins equally. Truth-oracle selectors are upper bounds within these samples; they do not establish physically valid structures. Cluster results use independent-pool assignment and voting, with ≥10 assigned samples.")
    plt.close(fig)
    for page, cases in enumerate(CASES, 1):
        fig, axes = plt.subplots(4, 4, figsize=(15, 15), layout="constrained")
        for row, stem in enumerate(cases):
            record = case_metadata[stem]
            case = case_rows[case_rows.stem.eq(stem)]
            columns = []
            for variant, label in (("pooled", "Pooled top-R"), ("oracle", "Oracle"), ("consensus", "Consensus"), ("truth", "Truth")):
                rows = case[case.variant.eq(variant)]
                pairs = list(rows[["i", "j"]].itertuples(index=False, name=None))
                if variant in {"oracle", "consensus"}:
                    label += f" #{int(rows.rollout.iloc[0])+1}"
                columns.append((label, pairs))
            truth = set(columns[-1][1])
            for col, (label, pairs) in enumerate(columns):
                ax = axes[row,col]
                mask = np.zeros((record["L"],record["L"]), dtype=bool)
                for a,b in pairs:
                    mask[a,b] = True
                for pos in sorted(set(range(record["L"])) - set(record["resolved"])):
                    ax.axhspan(pos+.5,pos+1.5,color="#dfe2e5",lw=0)
                    ax.axvspan(pos+.5,pos+1.5,color="#dfe2e5",lw=0)
                draw_map(ax,mask,"#087e83" if col==3 else "#233d69",size=2)
                score = map_metrics(pairs,truth)
                ax.set(xlim=(0,record["L"]+1),ylim=(record["L"]+1,0),aspect="equal",xlabel="Residue j",ylabel=f"{stem}\nResidue i" if col==0 else "Residue i",
                       title=label+(f" · F1 {score['f1']:.1%}" if col<3 else f" · {len(truth)} contacts"))
                ax.set_facecolor("#fafaf8")
        save_plot_with_meta(fig, HERE / f"plots/whole_map_cases_{page}.png", caption="Original eight case studies revisited using fresh samples. The oracle is the individual map maximizing all-range F1 across 200 samples; consensus chooses from A by similarity to B without truth. Gray marks unresolved residues. Samples have variable contact counts; pooled top-R has exactly the truth contact count.")
        plt.close(fig)


if __name__ == "__main__":
    main()
