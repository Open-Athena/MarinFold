"""Score the completed spike-in sweep and plot remaining-contact dose responses."""

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
from plot_length import interval
from protocol import remaining_metrics

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Require the full frozen plan before emitting a headline dose-response plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("/data/exp369/spiking_plan.json"))
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    raw = args.plan.read_bytes()
    plan = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    fs, root = fsspec.core.url_to_fs(args.run)
    rows, timings = [], []
    for target in plan["targets"]:
        for repeat in range(plan["repeats"]):
            grid = target["doses_by_repeat"][str(repeat)]
            conditions = [(policy, k) for k in sorted({d["k"] for d in grid})
                          for policy in (["random"] if k == 0 else ["random", "sse_balanced"])]
            matrices = {}
            for policy, count in conditions:
                mode = f"{policy}-r{repeat}-k{count:04d}"
                base = f"{root}/units/{target['stem']}/{mode}"
                with fs.open(base + ".complete.json") as stream:
                    complete = json.load(stream)
                if complete["plan_sha256"] != digest or complete["n_rollouts"] != plan["n_rollouts"]:
                    raise ValueError(f"Condition provenance mismatch: {base}")
                if any(complete[key] for key in ("n_unfinished", "n_malformed", "n_out_of_range")):
                    raise ValueError(f"Invalid completions: {base}")
                with fs.open(base + ".npz", "rb") as stream:
                    payload = stream.read()
                if hashlib.sha256(payload).hexdigest() != complete["files_sha256"][".npz"]:
                    raise ValueError(f"Vote checksum mismatch: {base}")
                with np.load(io.BytesIO(payload)) as saved:
                    matrices[policy, count] = saved["score"]
                timings.append({k: v for k, v in complete.items() if k != "files_sha256"})
            for dose in grid:
                count = dose["k"]
                for policy in ("random", "sse_balanced"):
                    supplied = target["orders"][f"{policy}:{repeat}"][:count]
                    score = matrices[policy if count else "random", count]
                    for contact_range, separation in (("all", 6), ("long", 24)):
                        conditioned = remaining_metrics(score, target["truth"], target["resolved"], supplied, separation)
                        baseline = remaining_metrics(matrices["random", 0], target["truth"], target["resolved"], supplied, separation)
                        rows.append({"stem": target["stem"], "L": target["L"], "repeat": repeat, "policy": policy,
                                     "axis": dose["axis"], "dose": dose["dose"], "n_supplied": count,
                                     "supplied_fraction": count / len(target["truth"]), "range": contact_range,
                                     **conditioned, "iid_r_precision": baseline["r_precision"]})
    frame = pd.DataFrame(rows)
    frame["delta"] = frame.r_precision - frame.iid_r_precision
    frame.to_csv(HERE / "data/spiking_per_repeat.csv", index=False)
    pd.DataFrame(timings).to_csv(HERE / "data/timings.csv", index=False)
    keys = ["stem", "policy", "axis", "dose", "range"]
    proteins = frame.groupby(keys, as_index=False).agg(
        r_precision=("r_precision", "mean"), iid_r_precision=("iid_r_precision", "mean"),
        delta=("delta", "mean"), n_supplied=("n_supplied", "mean"),
        supplied_fraction=("supplied_fraction", "mean"), n_remaining_true=("n_remaining_true", "mean"))
    proteins.to_csv(HERE / "data/spiking_per_protein.csv", index=False)
    summary = []
    for key, group in proteins.groupby(["policy", "axis", "dose", "range"]):
        row = dict(zip(["policy", "axis", "dose", "range"], key))
        row.update(n_proteins=len(group), mean_count=group.n_supplied.mean(),
                   mean_fraction=group.supplied_fraction.mean(), mean_remaining_true=group.n_remaining_true.mean())
        for metric in ("r_precision", "iid_r_precision", "delta"):
            valid = group[metric].dropna().to_numpy()
            row[f"{metric}_n_defined"] = len(valid)
            row[metric] = float(valid.mean()) if len(valid) else None
            lower, upper = interval(valid) if len(valid) else (None, None)
            row[f"{metric}_lower"], row[f"{metric}_upper"] = lower, upper
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(HERE / "data/spiking_summary.csv", index=False)
    for contact_range in ("all", "long"):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        for column, (axis, title) in enumerate((("count", "Number of supplied true contacts"),
                                                ("fraction", "Fraction of all true contacts supplied"),
                                                ("sse_round", "Contacts per coverable secondary-structure element"))):
            for policy, color, label in (("random", "#007A87", "Random"), ("sse_balanced", "#C15C20", "SSE-balanced")):
                data = summary.loc[(summary["range"] == contact_range) & (summary.axis == axis) & (summary.policy == policy)].sort_values("dose")
                for index, metric in enumerate(("r_precision", "delta")):
                    ax = axes[index, column]
                    ax.plot(data.dose, data[metric], color=color, marker="o", label=label)
                    ax.fill_between(data.dose, data[f"{metric}_lower"], data[f"{metric}_upper"], color=color, alpha=.15)
                    if index == 0:
                        ax.plot(data.dose, data.iid_r_precision, color=color, linestyle="--", alpha=.7)
                    ax.set_xlabel(title)
                    ax.grid(alpha=.18)
                    ax.spines[["top", "right"]].set_visible(False)
                    if axis == "count":
                        ax.set_xscale("symlog", linthresh=1)
            axes[0, column].set_ylim(0, 1)
            axes[1, column].axhline(0, color="gray", linewidth=.7)
        axes[0, 0].set_ylabel("Remaining-contact R-precision")
        axes[1, 0].set_ylabel("Gain over matched unconditioned baseline")
        axes[0, 0].legend(frameon=False)
        fig.suptitle(f"exp277 step 266344 • eval-val • {contact_range}-range contacts\n100 rollouts × 2 context draws; 95% protein-bootstrap intervals", fontsize=13)
        save_plot_with_meta(fig, str(HERE / f"plots/spiking_{contact_range}.png"),
                            caption="Supplied pairs never score. Dashed curves: unconditioned scores on the identical remaining pair universe. Count doses omit proteins with no truth remaining; see CSV denominators. SSE quotas saturate elements with too few available contacts; random uses the same contact count.")
        plt.close(fig)
    thresholds = []
    for (policy, axis), group in summary.loc[summary["range"] == "all"].groupby(["policy", "axis"]):
        for label, mask in (("gain_at_least_0.05_with_positive_95ci", (group.delta >= .05) & (group.delta_lower > 0)),
                            ("remaining_r_precision_at_least_0.80", group.r_precision >= .8)):
            hit = group.loc[mask].sort_values("dose")
            thresholds.append({"policy": policy, "axis": axis, "criterion": label,
                               "first_tested_dose": None if hit.empty else float(hit.iloc[0].dose)})
    (HERE / "data/spiking_thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")


if __name__ == "__main__":
    main()
