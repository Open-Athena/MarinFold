# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Project the full-run wall-clock from a measured single-task rate.

The pipeline skill is explicit that a smoke rate is a *lower bound*: startup,
preemption retries and straggler tails land on top, and the gap can be 2-5x.
So this prints a band, not a number, and it prints what the band is made of.

    uv run python project_full_run.py --backbones-per-second 5.9 --tasks 28
"""

from __future__ import annotations

import argparse

BACKBONES = 3_963_003          # contacts_v1_decontam train rows (#225)
DESIGNS = 8
TOKENS_PER_DOC = 1246          # published corpus mean num_tokens


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbones-per-second", type=float, required=True,
                    help="Measured single-task rate on a realistic length band.")
    ap.add_argument("--tasks", type=int, default=28,
                    help="1xH100 tasks; check live idle GPUs before choosing.")
    ap.add_argument("--slack", type=float, nargs=2, default=(1.0, 3.0),
                    help="Multipliers bounding the projection (skill: 2-5x gaps "
                         "are normal; 1-3x assumes the band sample is honest).")
    args = ap.parse_args()

    per_task = BACKBONES / args.tasks
    base_h = per_task / args.backbones_per_second / 3600
    lo, hi = (base_h * s for s in args.slack)
    docs = BACKBONES * DESIGNS

    print(f"{BACKBONES:,} backbones x {DESIGNS} designs = {docs:,} documents")
    print(f"  ~{docs * TOKENS_PER_DOC / 1e9:.0f} B tokens")
    print(f"\n{args.tasks} tasks x {args.backbones_per_second:.1f} backbones/s "
          f"= {args.tasks * args.backbones_per_second:.0f}/s aggregate")
    print(f"  {per_task:,.0f} backbones per task")
    print(f"  wall-clock: {base_h:.1f} h nominal, {lo:.1f}-{hi:.1f} h with slack")
    print(f"  GPU-hours consumed: {args.tasks * base_h:.0f}-{args.tasks * hi:.0f}")

    print("\nfor reference:")
    for n in (14, 28, 56, 112):
        h = BACKBONES / n / args.backbones_per_second / 3600
        print(f"  {n:4d} tasks ({n // 8:2d} nodes): {h:5.1f} h nominal, "
              f"{h * args.slack[1]:5.1f} h with slack")


if __name__ == "__main__":
    main()
