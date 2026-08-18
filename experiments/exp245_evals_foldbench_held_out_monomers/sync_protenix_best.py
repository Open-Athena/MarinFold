# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 6c -- fetch only the Protenix structure each protein is scored on.

exp12's app writes 5 seeds x 8 diffusion samples per protein and mode, plus a
per-seed distogram, and the distograms are 77 % of the bytes. Syncing all of it
for 209 proteins is ~6 GB of small-file transfers, which measured at ~50 kB/s
through the Modal API -- more than a day. Nothing here needs it: exp245 reports
the **structure** readout (the one #213 published), so the only file that gets
scored is the top-ranked sample's mmCIF.

So this fetches, per (mode, protein):

* the five ``*_summary_confidence_sample_0.json`` files, one per seed, to
  compare seeds; and
* the winning seed's ``*_sample_0.cif``.

**The assumption, and its check.** Protenix's dumper sorts samples within a seed
by ``ranking_score``, so ``sample_0`` is that seed's best -- exp74's
``select_best`` docstring states it and reads all eight anyway. This relies on
it, and checks it on every seed directory a full sync already brought down:
24/24 complete seed directories have ``sample_0`` as the maximum, 0 violations.
``--verify-all`` re-reads all eight per seed and fails on any violation, at
eight times the requests.

The output tree is exp74's ``best/{mode}/{stem}/structure.cif`` layout minus the
distogram, so the scorer consumes it unchanged.

    uv run --extra gt python sync_protenix_best.py
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import upstream as U

DATA = U.DATA
PRED_ROOT = U.WORK / "pred"
VOLUME = "foldbench-protenix-runs"
MODES = ("single_seq", "msa")
SEEDS = (1, 2, 3, 4, 5)
REPORT = DATA / "protenix_selection.csv"


def read_json(handle, path: str) -> dict | None:
    try:
        payload = b"".join(handle.read_file(path))
    except Exception:  # noqa: BLE001 - a missing seed is reported, not fatal
        return None
    return json.loads(payload)


def main() -> int:
    import modal

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=DATA / "predictor_manifest_new.csv")
    parser.add_argument("--verify-all", action="store_true",
                        help="read all eight samples per seed instead of sample_0")
    args = parser.parse_args()

    stems = pd.read_csv(args.manifest).stem.tolist()
    handle = modal.Volume.from_name(VOLUME)
    rows = []
    for mode in MODES:
        destination = PRED_ROOT / f"protenix-v2_{mode}"
        destination.mkdir(parents=True, exist_ok=True)
        for index, stem in enumerate(stems, 1):
            target = destination / stem / "structure.cif"
            best = None
            for seed in SEEDS:
                prefix = f"/{mode}/{stem}/seed_{seed}/{stem}_summary_confidence_sample_"
                samples = range(8) if args.verify_all else (0,)
                for sample in samples:
                    payload = read_json(handle, f"{prefix}{sample}.json")
                    if payload is None:
                        continue
                    score = float(payload["ranking_score"])
                    if args.verify_all and sample > 0 and best and best[0] == seed \
                            and score > best[1]:
                        raise AssertionError(
                            f"{mode}/{stem} seed {seed}: sample {sample} outranks "
                            "sample 0, so the dumper's ordering cannot be relied on"
                        )
                    if best is None or score > best[1]:
                        best = (seed, score, sample)
            if best is None:
                rows.append({"mode": mode, "stem": stem, "seed": None,
                             "ranking_score": None, "status": "no samples"})
                continue
            seed, score, sample = best
            if not (target.exists() and target.stat().st_size > 0):
                target.parent.mkdir(parents=True, exist_ok=True)
                source = f"/{mode}/{stem}/seed_{seed}/{stem}_sample_{sample}.cif"
                with target.open("wb") as sink:
                    for chunk in handle.read_file(source):
                        sink.write(chunk)
            rows.append({"mode": mode, "stem": stem, "seed": seed,
                         "sample": sample, "ranking_score": score, "status": "ok"})
            if index % 25 == 0:
                print(f"  [protenix] {mode}: {index}/{len(stems)}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(REPORT, index=False)
    print(frame.groupby(["mode", "status"]).size().to_string(), flush=True)
    print(f"[protenix] selection -> {REPORT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
