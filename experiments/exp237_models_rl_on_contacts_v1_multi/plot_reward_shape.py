# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Two figures about the reward itself — issue #237.

**Figure 1, the dose-response.** The headline. Every scored checkpoint's
R-precision against how far its policy moved, with the budget-matched plain
baseline drawn on. This is the figure that says #208's negative result is the far
end of a curve rather than a verdict: all three rewards improve consensus at
small KL and damage it at large, and #208 ran every one of its arms past the
peak.

**Figure 2, the shape of arm M-C's reward.** #237 carries #208's rule that the
expectation calculation is done on paper before the run. M-C's advantage is
centred so ``E[A] = 0`` holds exactly per prompt — and the policy still shrank.
This plots why the identity was not enough: the marginal distribution has a
**55 % atom at exactly zero**, i.e. more than half of all sections change no vote
at all, and those sections carry a mean advantage of −0.062 once the group is
centred.

The figure also carries its own refutation, which is the point of drawing it:
arm M-F's reward has no such atom and collapsed volume by the same factor. So the
atom is a contributing bias, not the cause. See RESULTS.md.

    python plot_reward_shape.py --out plots/
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from build_summary import save_plot_with_meta  # noqa: E402

#: Terminal KL of each scored checkpoint, from `summarize_runs.py`.
KL = {
    "exp230_step1988": 0.0, "m_0_step8": 0.0, "m_c_step18": 0.0072,
    "m_f_step18": 0.0136, "m_b_step36": 0.0163, "m_f_step36": 0.0306,
    "m_b_step80": 0.4863,
}
ARM_OF = {"m_c": "M-C", "m_f": "M-F", "m_b": "M-B", "m_0": "M-0", "exp230": "warm start"}
COLORS = {"M-C": "#1f77b4", "M-F": "#d62728", "M-B": "#2ca02c",
          "M-0": "#7f7f7f", "warm start": "#000000"}
#: #230's budget-matched plain baseline, legacy 554 — the primary criterion.
PLAIN22 = {"consensus": 0.5896, "best": 0.5680}
MODES = [("consensus", "consensus over one rollout's sections"),
         ("best", "best section (ORACLE)"),
         ("last", "last section")]


def arm_of(label: str) -> str:
    for k, v in ARM_OF.items():
        if label.startswith(k):
            return v
    return label


def dose_response(data: Path, out: Path) -> None:
    rows = []
    for label, kl in KL.items():
        f = data / f"agg_modes_{label}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        for mode, _ in MODES:
            key = f"legacy554/{mode}"
            if key in d:
                rows.append(dict(label=label, arm=arm_of(label), kl=kl, mode=mode,
                                 r_prec=d[key]["r_prec"]))
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharex=True)
    for ax, (mode, title) in zip(axes, MODES):
        d = df[df["mode"] == mode].sort_values("kl")
        # A line through the points ordered by KL: the arms differ, but the
        # question is whether DISTANCE orders the outcome, so they belong on one
        # trace as well as in their own colours.
        ax.plot(d["kl"], d["r_prec"], "-", color="0.6", lw=1.0, zorder=1)
        for arm, g in d.groupby("arm"):
            ax.scatter(g["kl"], g["r_prec"], s=70, color=COLORS.get(arm), zorder=3,
                       label=arm, edgecolor="white", linewidth=1.0)
        base = d[d["label"] == "exp230_step1988"]["r_prec"]
        if len(base):
            ax.axhline(base.iloc[0], color="k", lw=0.8, ls=":")
            ax.text(0.02, base.iloc[0], "  warm start", fontsize=7, va="bottom",
                    transform=ax.get_yaxis_transform())
        if mode in PLAIN22:
            ax.axhline(PLAIN22[mode], color="crimson", lw=1.2, ls="-.")
            ax.text(0.98, PLAIN22[mode], "plain, 22 rollouts  ", color="crimson",
                    fontsize=7, ha="right", va="bottom",
                    transform=ax.get_yaxis_transform())
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("policy KL from the warm start")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("R-precision (all), legacy 554")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("exp237 — every reward helps at small KL and damages at large\n"
                 "#208 ran all of its arms past the peak; the two that stayed under "
                 "KL 0.0015 never moved at all", fontsize=11)
    fig.tight_layout()
    save_plot_with_meta(
        fig, out / "dose_response.png", dpi=150,
        caption=(
            "R-precision against distance moved. Consensus peaks at KL 0.007-0.016; #208 "
            "ran every arm past it. Red = the budget-matched bar, which nothing reaches."))
    print(f"wrote {out}/dose_response.png")


def reward_shape(data: Path, out: Path) -> None:
    f = data / "phase0_per_section.parquet"
    if not f.exists():
        print(f"skipping reward-shape figure: no {f}")
        return
    d = pd.read_parquet(f)
    g = d.groupby(["dataset", "stem"])["marg"]
    d["adv"] = g.transform(lambda v: (v - v.mean()) / (v.std() if v.std() > 0 else np.nan))
    zero = d["marg"] == 0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    lim = np.percentile(np.abs(d["marg"]), 99.5)
    ax.hist(d.loc[~zero, "marg"].clip(-lim, lim), bins=80, color="#1f77b4",
            alpha=0.85, label="changes the vote")
    ax.bar([0], [zero.sum()], width=lim / 40, color="crimson",
           label=f"changes nothing ({zero.mean():.0%})")
    ax.axvline(d["marg"].mean(), color="k", ls="--", lw=1.2)
    ax.text(d["marg"].mean(), ax.get_ylim()[1] * 0.9, "  mean", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("section leave-one-out marginal: C(all) - C(all without k)")
    ax.set_ylabel("sections (log)")
    ax.set_title("arm M-C's reward has an atom at exactly zero", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.hist(d.loc[~zero, "adv"].dropna(), bins=60, color="#1f77b4", alpha=0.85,
            label="changes the vote")
    ax.hist(d.loc[zero, "adv"].dropna(), bins=60, color="crimson", alpha=0.65,
            label="changes nothing")
    ax.axvline(0, color="k", lw=1.0)
    ax.set_xlabel("advantage after group centring: (m_k - group mean) / group sd")
    ax.set_ylabel("sections")
    ax.set_title(f"a section that changes nothing averages "
                 f"{d.loc[zero, 'adv'].mean():+.3f}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle("exp237 — why $E[A] = 0$ holding exactly was not enough", fontsize=11)
    fig.tight_layout()
    save_plot_with_meta(
        fig, out / "reward_shape.png", dpi=150,
        caption=(
            f"Arm M-C's reward before training: {zero.mean():.0%} of sections change no vote, "
            f"and average {d.loc[zero, 'adv'].mean():+.3f} once centred. M-F has no such atom "
            f"and collapsed the same -- so this is not the cause. See RESULTS.md."))
    print(f"wrote {out}/reward_shape.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("plots"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dose_response(a.data, a.out)
    reward_shape(a.data, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
