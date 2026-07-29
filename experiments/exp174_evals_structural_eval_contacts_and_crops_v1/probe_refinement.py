# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""E3 — does the model actually sharpen when a box is re-shown?

contacts-and-crops-v1 trains Pass 2 with a per-box refinement schedule: on a
box's ``i``-th appearance the atoms carry Gaussian noise σ = 1/(i+1)² Å, so a
box's first read is coarse (~1 Å, its tenths digit near-noise) and repeated
reads sharpen toward a crisp tenths. Plan F's inner loop *assumes the model
learned to condition on that visit count* — it re-shows a box specifically to
walk it down the schedule. Whether it did is an empirical question, and this is
the cheapest way to answer it.

The probe, per (protein, box, visit index ``i``):

1. Teacher-force a prompt from **ground truth**: the sequence section, a Pass-1
   section synthesized from the true coordinates (σ=2 Å, as in training), then
   ``i`` prior crops of the *same* box rendered at their training noise levels
   (σ = 1/(j+1)² for j < i), then the box's header.
2. Sample the crop body.
3. Compare every emitted atom position to the truth.

If the model learned the schedule, mean error falls as 1/(i+1)². If it is flat,
re-visits still buy independent samples but nothing more, and Plan F's sweep
budget should go into more samples per box instead.

``--control`` runs the length-matched null: the same number of prior crops, but
for *other* boxes, so the context is as long and the visit index is 0. It
separates "the model counts visits" from "the model likes more context".

Usage::

    uv run python probe_refinement.py --model _scratch/models/cc1mix5-step50000 \\
        --gt-dir _scratch/gt --out data/probe_refinement.csv --n-proteins 40
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import canonical_pdb
from document_codec import (
    CoordinateEstimate,
    Observation,
    crop_header,
    pass1_budget,
    parse_observations,
    place_in_cube,
    render_crop,
    sequence_prefix,
    start_index,
    synthesize_pass1,
)
from marinfold.document_structures.contacts_and_crops_v1 import GenerationConfig
from sampler import SamplingConfig, load_sampler

MAX_VISIT_INDEX = 5


def truth_estimate(gt) -> CoordinateEstimate:
    """Ground-truth coordinates in the shape ``synthesize_pass1`` consumes."""
    estimate = CoordinateEstimate()
    for res_id, atom_name, coord in zip(
        gt.res_id.tolist(), gt.atom_name.tolist(), gt.coord
    ):
        estimate.add(
            Observation(
                seq_index=res_id - 1,
                atom_name=atom_name,
                position=np.asarray(coord, dtype=np.float64),
                variance=1.0,
                source="crop",
                visit_index=0,
            )
        )
    return estimate


def noised_estimate(truth: CoordinateEstimate, keys, sigma: float, rng: random.Random):
    """A copy of ``truth`` for ``keys`` with σ Å isotropic noise — one training read."""
    noisy = CoordinateEstimate()
    for key in keys:
        position = truth.position(key) + np.array(
            [rng.gauss(0.0, sigma) for _ in range(3)]
        )
        noisy.add(
            Observation(
                seq_index=key[0],
                atom_name=key[1],
                position=position,
                variance=1.0,
                source="crop",
                visit_index=0,
            )
        )
    return noisy


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-proteins", type=int, default=40)
    ap.add_argument("--boxes-per-protein", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--coord-temperature", type=float, default=1.0)
    ap.add_argument("--struct-temperature", type=float, default=1.0)
    ap.add_argument("--control", action="store_true",
                    help="length-matched null: prior crops name *other* boxes")
    args = ap.parse_args(argv)

    records = [json.loads(line) for line in (args.gt_dir / "gt_index.jsonl").open()]
    rng = random.Random(args.seed)
    rng.shuffle(records)
    # Keep the probe on chains where Pass 1 covers the structure comfortably, so
    # the measurement is about refinement rather than about truncation.
    records = [r for r in records if r["L"] <= 250][: args.n_proteins]

    sampler = load_sampler(Path(args.model))
    config = SamplingConfig(
        coord_temperature=args.coord_temperature,
        struct_temperature=args.struct_temperature,
    )
    generator = torch.Generator(device=sampler.device).manual_seed(args.seed)
    schedule = GenerationConfig()

    rows = []
    started = time.time()
    for n, record in enumerate(records, start=1):
        sequence = record["input_seq"]
        entry_id = record["record_id"]
        prefix = sequence_prefix(entry_id, sequence)
        if prefix is None:
            continue
        start = start_index(entry_id, sequence)
        gt = canonical_pdb.read_structure(
            args.gt_dir / "gt_structures" / record["dataset"] / f"{record['stem']}.pdb"
        )
        # Place into the format's cube first: a depositor's frame has negative
        # coordinates, which the <xyz-DDD> vocabulary cannot express.
        truth = place_in_cube(truth_estimate(gt), random.Random(args.seed))
        cap, _ = pass1_budget(len(sequence))
        pass1_tokens = synthesize_pass1(
            truth, start=start, cap_tokens=cap, rng=random.Random(args.seed)
        )
        base = prefix + pass1_tokens

        cells = truth.occupied_cells()
        populated = [c for c, keys in cells.items() if len(keys) >= 5]
        if not populated:
            continue
        rng.shuffle(populated)

        for cell in populated[: args.boxes_per_protein]:
            members = cells[cell]
            true_positions = {key: truth.position(key) for key in members}
            for visit in range(MAX_VISIT_INDEX + 1):
                forced: list[str] = []
                for prior in range(visit):
                    if args.control:
                        # Same context length, different box: visit index stays 0.
                        other = populated[(populated.index(cell) + prior + 1) % len(populated)]
                        noisy = noised_estimate(
                            truth, cells[other], schedule.refine_sigma(0), rng
                        )
                        forced += render_crop(other, cells[other], noisy, start=start)
                    else:
                        noisy = noised_estimate(
                            truth, members, schedule.refine_sigma(prior), rng
                        )
                        forced += render_crop(cell, members, noisy, start=start)
                forced += crop_header(cell)

                prompt_ids = sampler.encode(base)
                body = sampler.sample(
                    prompt_ids,
                    n_samples=1,
                    config=config,
                    max_new_tokens=4 * (len(members) + 8),
                    forced_ids=sampler.encode(forced),
                    stop_token_ids=[sampler.crop_id],
                    generator=generator,
                )[0]

                errors = []
                for observation in parse_observations(
                    crop_header(cell) + sampler.decode(body),
                    start=start,
                    length=len(sequence),
                ):
                    if observation.source != "crop":
                        continue
                    truth_position = true_positions.get(observation.key)
                    if truth_position is None:
                        continue
                    errors.append(
                        float(np.linalg.norm(observation.position - truth_position))
                    )
                rows.append(
                    {
                        "record_id": entry_id,
                        "cell": "/".join(str(c) for c in cell),
                        "visit_index": visit,
                        "schedule_sigma_a": schedule.refine_sigma(visit),
                        "n_members": len(members),
                        "n_emitted_scored": len(errors),
                        "mean_error_a": float(np.mean(errors)) if errors else float("nan"),
                        "median_error_a": float(np.median(errors)) if errors else float("nan"),
                        "control": bool(args.control),
                    }
                )
        if n % 5 == 0:
            print(f"  ...{n}/{len(records)} proteins ({time.time() - started:.0f}s)", flush=True)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    summary = frame.groupby("visit_index").agg(
        n=("mean_error_a", "size"),
        mean_error_a=("mean_error_a", "mean"),
        median_error_a=("median_error_a", "median"),
        schedule_sigma_a=("schedule_sigma_a", "first"),
        mean_emitted=("n_emitted_scored", "mean"),
    )
    print(f"[probe] {len(frame)} (protein, box, visit) trials -> {args.out}")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
