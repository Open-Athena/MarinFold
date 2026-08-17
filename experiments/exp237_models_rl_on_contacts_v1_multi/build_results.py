# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect every arm's aggregation-mode scores into one table — issue #237.

`score_agg_modes.py` (#230's, run unchanged by `run_eval.sh`) writes one JSON per
scored checkpoint, keyed ``<cut>/<mode>``. This joins them against the two
numbers #237 is judged by and prints the table the README carries.

**The bar is the budget-matched plain baseline, not #230's own multi number.**
#230 measured 22 independent plain rollouts, consensus-voted, at **0.5896** on
the legacy 554, against **0.5673** for one multi rollout's ~22 sections. Beating
0.5673 says RL improved the format; beating 0.5896 says the format plus RL beats
simply sampling 22 times, which nothing has done yet. Both are printed, and the
success criterion is the second.

`best` is an **ORACLE** — it selects the section using ground truth — and is
labelled as such everywhere it appears. It bounds what a perfect selector could
reach from one rollout; nothing can be deployed at that number.

    python build_results.py --eval ~/exp237_data/eval --base ~/exp230_data/... --out data/
"""

import argparse
import glob
import json
from pathlib import Path

#: #230's published numbers, legacy 554, R-precision (all). The two bars.
PLAIN22_CONSENSUS = 0.5896      # budget-matched plain baseline -- the bar that matters
MULTI_CONSENSUS_230 = 0.5673    # the warm start's own multi consensus
MULTI_BEST_230 = 0.5342         # ORACLE
MULTI_LAST_230 = 0.4566
MULTI_SECOND_LAST_230 = 0.4284

CUTS = ("legacy554", "eval2", "eval2_natural", "eval2_lt30")
MODES = ("consensus", "best", "last", "second_last")


def load(root: str) -> dict[str, dict]:
    """-> {label: report}, one per scored checkpoint under `root`."""
    out = {}
    for path in sorted(glob.glob(f"{root.rstrip('/')}/**/agg_modes_*.json", recursive=True)):
        p = Path(path)
        if p.name.endswith("_per_rollout.parquet"):
            continue
        if p.name == "agg_modes_all.json":
            # This script's own output. Globbing it back in on a re-run would
            # nest the whole table inside itself and print nonsense.
            continue
        label = p.stem.replace("agg_modes_", "")
        out[label] = json.loads(p.read_text())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="root holding <arm>_step<N>/agg_modes_*.json")
    ap.add_argument("--baseline", default=None,
                    help="#230's own agg_modes_finetune.json, re-scored or copied")
    ap.add_argument("--out", type=Path, default=Path("data"))
    a = ap.parse_args()

    reports = load(a.eval)
    if a.baseline and Path(a.baseline).exists():
        reports["exp230_step1988"] = json.loads(Path(a.baseline).read_text())
    if not reports:
        raise SystemExit(f"no agg_modes_*.json under {a.eval}")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "agg_modes_all.json").write_text(json.dumps(reports, indent=2) + "\n")

    for cut in CUTS:
        print(f"\n=== {cut}: R-precision (all) ===")
        hdr = f"{'checkpoint':<26}" + "".join(f"{m:>14}" for m in MODES) + f"{'sections':>10}"
        print(hdr)
        print("-" * len(hdr))
        for label in sorted(reports):
            r = reports[label]
            cells = []
            sections = None
            for mode in MODES:
                d = r.get(f"{cut}/{mode}")
                cells.append(f"{d['r_prec']:>14.4f}" if d else f"{'-':>14}")
                if d and sections is None:
                    sections = d.get("n_sections")
            print(f"{label:<26}" + "".join(cells)
                  + (f"{sections:>10.2f}" if sections else f"{'-':>10}"))
        if cut == "legacy554":
            print(f"{'#230 checkpoint (pub.)':<26}{MULTI_CONSENSUS_230:>14.4f}"
                  f"{MULTI_BEST_230:>14.4f}{MULTI_LAST_230:>14.4f}"
                  f"{MULTI_SECOND_LAST_230:>14.4f}{22.02:>10.2f}")
            print(f"{'plain, 22 rollouts (bar)':<26}{PLAIN22_CONSENSUS:>14.4f}"
                  f"{0.5680:>14.4f}{'-':>14}{'-':>14}{22.0:>10.2f}")
    print("\n`best` is an ORACLE -- it selects the section with ground truth. A ceiling.")
    print(f"PRIMARY criterion: consensus on legacy554 > {PLAIN22_CONSENSUS} "
          f"(the budget-matched plain baseline).")
    print(f"Beating #230's own {MULTI_CONSENSUS_230} is necessary but NOT sufficient.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
