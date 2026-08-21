# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Each arm's own reward over training — issue #237.

**There is no single "reward curve" for this experiment**, because the arms do
not share a reward: M-B is scored on ``max_k F1(section k)``, M-F on
``F1(last section)``, and M-C on per-section leave-one-out marginals, which are
not one number per rollout at all. Plotting them on one axis would be a category
error. So each arm is drawn against **its own objective**, and a second panel
puts every arm on the one axis they do share — the rollout's own consensus
R-precision, which is the quantity the deployed metric is computed from.

**Rolling medians, and that is not cosmetic.** A batch is 8 proteins and the
per-batch statistics are dominated by *which* 8: the zero-LR control, whose
policy did not change at all, produced swings of 2-4x in these same quantities
over 8 batches. The raw series is drawn faint behind each median so the reader
can see how much of it is the protein draw.

Runs are given explicitly rather than globbed, because a resumed arm writes a
second log (`.partN.log`) and the two halves are different segments of one
trajectory, not two runs.

    python plot_reward_curves.py --run "M-C lr1e-5=<log>" ... --out plots/
"""

import argparse
import ast
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from build_summary import save_plot_with_meta  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*m")
EXP = re.compile(r"\[exp237-metrics\] batch=(\d+) (.*)$")
STEP_DICT = re.compile(r"trainer:train:\d+ - (\{'policy_entropy'.*\})\s*$")
WINDOW = 6

#: Each arm's own reward, and the label for it.
OBJECTIVE = {
    "M-B":  ("best_f1", "max$_k$ F1(section $k$)"),
    "M-BC": ("best_f1", "max$_k$ F1 (blended with consensus)"),
    "M-F":  ("last_f1", "F1(last section)"),
    "M-FC": ("last_f1", "F1(last section) (blended with consensus)"),
    # M-K's reward IS this column, exactly -- the rollout's own consensus
    # R-precision. M-C's is derived from it (per-section leave-one-out marginals
    # on the same quantity), and M-0 has no reward at all.
    "M-C":  ("consensus_rprec", "the rollout's own consensus (M-C's marginals derive from it)"),
    "M-K":  ("consensus_rprec", "the rollout's own consensus — M-K's reward, exactly"),
    "M-0":  ("consensus_rprec", "consensus (control, lr 0)"),
}
#: Colour by ARM, so the two M-B runs read as one arm at two learning rates;
#: they are separated by line style instead. Keyed on the arm alone because the
#: lr reaches this script as a string from two sources ("1e-5" from a filename,
#: "1e-05" from the reduced frame) and a lookup on the pair silently falls back
#: to matplotlib's default cycle, which is how the first render came out.
COLOR = {"M-B": "#1a7f4b", "M-C": "#1f77b4", "M-F": "#d62728",
         "M-BC": "#9467bd", "M-FC": "#e08214", "M-K": "#111111", "M-0": "#9aa5b1"}


def style(label: str) -> dict:
    arm = label.split()[0]
    slow = "3e-6" in label
    # M-C and M-K plot the SAME column (the rollout's own consensus) and their
    # early trajectories nearly coincide, so colour alone hides one behind the
    # other. Dash patterns, not just hue, keep every run readable.
    ls = {"M-0": (0, (4, 3)), "M-C": (0, (5, 2)), "M-FC": (0, (1, 1.2))}.get(arm, "-")
    if slow:
        ls = (0, (7, 2, 1, 2))
    return dict(color=COLOR.get(arm, "#444"), ls=ls,
                alpha=0.85 if (slow or arm == "M-0") else 1.0,
                lw=1.4 if arm == "M-0" else (2.0 if slow else 2.4),
                zorder=5 if arm == "M-C" else (4 if arm == "M-K" else 2))


def pretty(label: str) -> str:
    return label.replace("lr1e-05", "lr 1e-5").replace("lr3e-6", "lr 3e-6").replace("lr0.0", "lr 0")


def parse(path: Path) -> pd.DataFrame:
    rows, kls = {}, []
    for raw in path.read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw)
        m = EXP.search(line)
        if m:
            kv = {}
            for tok in m.group(2).split():
                k, _, v = tok.partition("=")
                try:
                    kv[k] = float(v)
                except ValueError:
                    pass
            rows[int(m.group(1))] = kv
            continue
        m = STEP_DICT.search(line)
        if m:
            try:
                kls.append(float(ast.literal_eval(m.group(1)).get("policy_kl", float("nan"))))
            except (ValueError, SyntaxError):
                pass
    df = pd.DataFrame([dict(step=k, **v) for k, v in sorted(rows.items())])
    df["policy_kl"] = pd.Series(kls[: len(df)]).reindex(range(len(df))).values
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--offset", action="append", default=[], metavar="LABEL=N",
                    help="add N to a resumed run's step index")
    ap.add_argument("--csv", type=Path, default=None,
                    help="the reduced per-step frame (data/training_steps.csv.gz). Arm M-B's "
                         "first log was overwritten by its own resume before run_arm.sh learned "
                         "to rotate logs, so its steps 1-36 survive only here.")
    ap.add_argument("--out", type=Path, default=Path("plots"))
    a = ap.parse_args()

    offsets = dict(kv.split("=") for kv in a.offset)
    runs = {}
    if a.csv:
        frame = pd.read_csv(a.csv)
        for (arm, lr), g in frame.groupby(["arm", "lr"]):
            runs[f"{arm} lr{lr}"] = g.sort_values("step").reset_index(drop=True)
            print(f"[curves] {arm} lr{lr:<8} {len(g):>4} batches  (from {a.csv.name})")
    for spec in a.run:
        label, _, path = spec.partition("=")
        df = parse(Path(path))
        if label in offsets:
            df["step"] = df["step"] + int(offsets[label])
        runs[label] = df
        print(f"[curves] {label:<14} {len(df):>4} batches, "
              f"terminal KL {df['policy_kl'].dropna().iloc[-1]:.4f}"
              if df["policy_kl"].notna().any() else f"[curves] {label}: {len(df)} batches")

    a.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    for label, df in runs.items():
        arm = label.split()[0]
        col, _ = OBJECTIVE[arm]
        if col not in df:
            continue
        g = df.dropna(subset=[col]).sort_values("step")
        st = style(label)
        ax.plot(g["step"], g[col], color=st["color"], lw=0.6, alpha=0.2)
        ax.plot(g["step"], g[col].rolling(WINDOW, min_periods=3).median(),
                label=f"{pretty(label)} · {col}", **st)
    ax.set_xlabel("training step")
    ax.set_ylabel("reward (rolling median of 6 batches)")
    ax.set_ylim(0.15, 0.63)
    ax.annotate("M-F's continuation exits here → 0.006 by step 120",
                xy=(60, 0.17), fontsize=7.5, color=COLOR["M-F"])
    ax.set_title("each arm against its OWN objective", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower left")

    ax = axes[1]
    for label, df in runs.items():
        if "consensus_rprec" not in df:
            continue
        g = df.dropna(subset=["consensus_rprec"]).sort_values("step")
        st = style(label)
        ax.plot(g["step"], g["consensus_rprec"], color=st["color"], lw=0.6, alpha=0.2)
        ax.plot(g["step"], g["consensus_rprec"].rolling(WINDOW, min_periods=3).median(),
                label=pretty(label), **st)
    ax.set_xlabel("training step")
    ax.set_ylabel("within-rollout consensus R-precision")
    ax.set_ylim(0.15, 0.63)
    ax.set_title("the one axis every arm shares (training batches, not eval)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower left")

    fig.suptitle("exp237 — reward over training. Faint lines are raw batches: that swing is "
                 "the protein draw,\nand the lr-0 control produced it with a policy that never "
                 "changed.", fontsize=10.5)
    fig.tight_layout()
    save_plot_with_meta(
        fig, a.out / "reward_curves.png", dpi=150,
        caption=(
            "Each arm against its own reward (left) and the shared consensus axis (right), "
            "rolling median of 6 batches. Training-batch values on 8 proteins, not eval."))
    print(f"wrote {a.out}/reward_curves.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
