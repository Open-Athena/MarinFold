# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn exp237's training logs into the per-step table — issue #237.

SkyRL's configured logger is ``console``, so the run's record IS the log file.
Two things are parsed out of it and joined on the step index:

* SkyRL's own per-step metric dict (``policy_kl``, ``policy_entropy``,
  ``grad_norm``, the timings, and ``minibatch_rollout_logprobs_abs_diff_mean``);
* this experiment's ``[exp237-metrics]`` line, which carries the columns #237
  asks every run to report — ``sections_per_rollout``, ``union_pairs``,
  ``total_votes``, ``votes_per_pair`` and ``mean_jaccard``.

**``policy_kl`` is the most useful column in the output.** #208's lesson is that
several arms "did nothing" simply because they never moved, and a null result at
a learning rate that does not move the policy is not a result. Any arm finishing
below ~0.0015 is untested, not negative.

``minibatch_rollout_logprobs_abs_diff_mean`` is the second: it is the trainer /
engine agreement, ~0.017 in a healthy unsharded run and >1 when SkyRL's policy
sharding has pushed a divergent copy into the engines. It is the cheapest
tripwire in the stack and it fires before the wasted compute, not after.

    python summarize_runs.py --logs ~/exp237_logs --out data/
"""

import argparse
import ast
import glob
import json
import re
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP_DICT = re.compile(r"trainer:train:\d+ - (\{'policy_entropy'.*\})\s*$")
EXP_LINE = re.compile(r"\[exp237-metrics\] batch=(\d+) (.*)$")
KILL = re.compile(r"KILL CRITERION met for \[(.*?)\]")


def parse_log(path: Path) -> dict:
    steps: dict[int, dict] = {}
    batches: dict[int, dict] = {}
    killed = None
    for raw in path.read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw)
        m = STEP_DICT.search(line)
        if m:
            try:
                d = ast.literal_eval(m.group(1))
            except (ValueError, SyntaxError):
                continue
            steps[len(steps) + 1] = {k: float(v) for k, v in d.items()
                                     if isinstance(v, (int, float))}
            continue
        m = EXP_LINE.search(line)
        if m:
            kv = {}
            for tok in m.group(2).split():
                if "=" in tok:
                    k, _, v = tok.partition("=")
                    try:
                        kv[k] = float(v)
                    except ValueError:
                        pass
            batches[int(m.group(1))] = kv
            continue
        m = KILL.search(line)
        if m:
            killed = m.group(1)
    rows = []
    for i in sorted(set(steps) | set(batches)):
        row = {"step": i}
        row.update(batches.get(i, {}))
        row.update(steps.get(i, {}))
        rows.append(row)
    return {"rows": rows, "killed": killed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="dir of exp237_<arm>_lr<lr>.log files")
    ap.add_argument("--out", type=Path, default=Path("data"))
    a = ap.parse_args()

    import pandas as pd

    a.out.mkdir(parents=True, exist_ok=True)
    frames, summary = [], {}
    for path in sorted(glob.glob(f"{a.logs.rstrip('/')}/exp237_*.log")):
        p = Path(path)
        arm = p.stem.replace("exp237_", "").split("_lr")[0].replace("_", "-").upper()
        lr = p.stem.split("_lr")[-1]
        parsed = parse_log(p)
        if not parsed["rows"]:
            continue
        df = pd.DataFrame(parsed["rows"])
        df.insert(0, "lr", lr)
        df.insert(0, "arm", arm)
        frames.append(df)
        last = df.iloc[-1]
        first = df.iloc[0]

        def get(col, row=last):
            v = row.get(col)
            return float(v) if v is not None and v == v else None

        summary[arm] = dict(
            lr=lr, steps=int(len(df)), killed=parsed["killed"],
            terminal_kl=get("policy_kl"),
            max_logprob_gap=float(df["minibatch_rollout_logprobs_abs_diff_mean"].max())
            if "minibatch_rollout_logprobs_abs_diff_mean" in df else None,
            sections_first=get("sections_per_rollout", first),
            sections_last=get("sections_per_rollout"),
            union_first=get("union_pairs", first), union_last=get("union_pairs"),
            votes_per_pair_first=get("votes_per_pair", first),
            votes_per_pair_last=get("votes_per_pair"),
            jaccard_first=get("mean_jaccard", first), jaccard_last=get("mean_jaccard"),
            consensus_first=get("consensus_rprec", first),
            consensus_last=get("consensus_rprec"),
            last_f1_first=get("last_f1", first), last_f1_last=get("last_f1"),
            best_f1_first=get("best_f1", first), best_f1_last=get("best_f1"),
            precision_first=get("precision", first), precision_last=get("precision"),
            dead_prompts=get("dead_prompts"),
        )

    if not frames:
        raise SystemExit(f"no exp237_*.log with parseable steps under {a.logs}")
    pd.concat(frames, ignore_index=True).to_csv(a.out / "training_steps.csv.gz", index=False)
    (a.out / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    hdr = (f"{'arm':<6}{'lr':>8}{'steps':>6}{'KL':>9}{'lp-gap':>8}{'sections':>18}"
           f"{'union':>16}{'votes/pair':>16}{'jaccard':>16}")
    print(hdr)
    print("-" * len(hdr))
    for arm, s in sorted(summary.items()):
        def pair(a_, b_):
            if a_ is None or b_ is None:
                return "-"
            return f"{a_:.2f}->{b_:.2f}" if abs(a_) >= 1 else f"{a_:.3f}->{b_:.3f}"
        print(f"{arm:<6}{s['lr']:>8}{s['steps']:>6}"
              f"{(s['terminal_kl'] if s['terminal_kl'] is not None else float('nan')):>9.4f}"
              f"{(s['max_logprob_gap'] if s['max_logprob_gap'] is not None else float('nan')):>8.3f}"
              f"{pair(s['sections_first'], s['sections_last']):>18}"
              f"{pair(s['union_first'], s['union_last']):>16}"
              f"{pair(s['votes_per_pair_first'], s['votes_per_pair_last']):>16}"
              f"{pair(s['jaccard_first'], s['jaccard_last']):>16}")
        if s["killed"]:
            print(f"       ^ STOPPED on the preregistered kill criterion: {s['killed']}")
    print("\npolicy_kl below ~0.0015 means the arm never moved -- untested, not negative.")
    print("lp-gap is trainer/engine logprob disagreement: ~0.017 healthy, >0.1 means the "
          "policy and the inference engines have diverged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
