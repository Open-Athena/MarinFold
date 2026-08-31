# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Score #245's FoldBench monomers with one checkpoint — issue #250.

#232's newer `m2-p06` final (step 363,000) beats the sweep final it replaced,
but its 2026-08-24 evaluation deliberately left `eval-test` unscored, and it
never produced the dense contact maps Helico's conditioned arm needs. Both gaps
are the same job: run exp82's rollout+resample recipe over the 333 monomers and
keep the full matrix, not just the metric.

What comes out feeds two figures:

* `metrics.csv` — per protein, per separation range, per cut, computed with
  #89's implementation (the same one #245 published its baselines with) so the
  new model can be drawn beside them.
* `dense/<dataset>__<stem>.npz` — the [L, L] vote matrix, which is what Helico
  conditions on.

`--validate` reruns an already-published checkpoint instead, so the difference
between this pipeline and the published numbers can be measured rather than
assumed. Do that before believing a new model's score: everything here is
sampled, and a pipeline that quietly disagrees with #245's would show up as a
model result.

    # one shard per GPU, on a multi-GPU box
    python score_foldbench_rollouts.py --model contacts-v1-exp232-m2-p06-train-1.5B --gpus 8

    # the control: the checkpoint #245 already published numbers for
    python score_foldbench_rollouts.py --validate --eval-sets eval-val --gpus 8
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "figures"))

import figlib  # noqa: E402  — the shared provenance/metric helpers live next door

#: The checkpoint every #250 figure should be drawn from, and the one whose
#: published scores make the control meaningful.
DEFAULT_MODEL = "contacts-v1-exp232-m2-p06-train-1.5B"
VALIDATION_MODEL = "contacts-v1-exp232-m2-p06-1.5B"
#: #245's published predictor name for the validation model, for the comparison it enables.
VALIDATION_PREDICTOR = "#232 m2-p06 (decontaminated)"

OUT_ROOT = HERE / "data/foldbench_rescore"


def log(message: str) -> None:
    print(f"[exp250] {message}", file=sys.stderr, flush=True)


def targets_for(inputs, eval_sets, limit):
    """#245's monomers in the requested eval sets, longest first.

    Longest first because the shards are round-robin over this order: rollout
    cost grows with length, so dealing the long proteins out one per shard
    keeps the shards within a few minutes of each other instead of leaving one
    holding every large target.
    """
    targets = figlib.load_foldbench_universe(inputs)
    targets = targets[targets.eval_set.isin(eval_sets)]
    targets = targets.sort_values(["L", "stem"], ascending=[False, True]).reset_index(drop=True)
    return targets.head(limit) if limit else targets


def ground_truth(inputs):
    """(dataset, stem) -> #245's scored ground-truth record."""
    payload = inputs.fetch(f"{figlib.EXP245}/gt_universe_scored.jsonl").decode()
    records = [json.loads(line) for line in payload.splitlines() if line.strip()]
    return {(record["dataset"], record["stem"]): record for record in records}


def run_shard(arguments) -> int:
    """Score this shard's slice of the targets, one protein at a time."""
    import numpy as np
    import pandas as pd
    import torch

    from marinfold.document_structures.contacts_v1 import (
        InferenceConfig, predict, structure_from_sequence)

    out = OUT_ROOT / arguments.tag
    (out / "dense").mkdir(parents=True, exist_ok=True)
    inputs = figlib.Inputs()
    targets = targets_for(inputs, arguments.eval_sets, arguments.limit)
    truth = ground_truth(inputs)
    mine = targets[targets.index % arguments.num_shards == arguments.shard]

    pending = []
    for row in mine.itertuples():
        destination = out / "dense" / f"{row.dataset}__{row.stem}.npz"
        if destination.exists() and not arguments.force:
            continue
        pending.append(row)
    log(f"shard {arguments.shard}/{arguments.num_shards}: {len(mine)} targets, "
        f"{len(pending)} to do")
    if not pending:
        return 0

    dtype = "bfloat16" if torch.cuda.is_available() else "float32"
    config = InferenceConfig(
        model=arguments.model, backend=arguments.backend, method="rollout", keep_matrix=True,
        n_rollouts=arguments.n_rollouts, temperature=1.0, top_p=0.95, top_k=-1, dtype=dtype,
        min_seq_separation=figlib.MIN_SEPARATION,
        gpu_memory_utilization=arguments.gpu_memory_utilization)

    rows, started = [], time.time()
    structures = [structure_from_sequence(row.input_seq, entry_id=row.stem) for row in pending]
    # One engine for the whole shard: `predict` loads the backend once and then walks the
    # structures in order, so the 90 s model load is paid once rather than 40 times.
    for row, record in zip(pending, predict(config, structures=structures)):
        if record["entry_id"] != row.stem:
            raise SystemExit(f"predict returned {record['entry_id']} for {row.stem}; the "
                             "record order is not the structure order and every matrix "
                             "below would be attached to the wrong protein")
        score = np.nan_to_num(np.asarray(record["score_matrix"], dtype=float), nan=-1e9)
        votes = np.where(score > -1e8, np.floor(score), 0).astype(np.int16)
        np.savez_compressed(out / "dense" / f"{row.dataset}__{row.stem}.npz",
                            score=score.astype(np.float32), votes=votes)
        metrics = figlib.score_metrics(score, truth[(row.dataset, row.stem)])
        rows.append(metrics.assign(dataset=row.dataset, stem=row.stem, eval_set=row.eval_set,
                                   L=row.L))
        headline = metrics[(metrics.range == "all") & (metrics.cut == "R")].iloc[0]
        log(f"  {row.stem:12s} L={row.L:<5d} R={headline.value:.3f} "
            f"({time.time() - started:.0f}s elapsed)")

    frame = pd.concat(rows, ignore_index=True)
    frame.to_csv(out / f"metrics-shard{arguments.shard:02d}.csv", index=False)
    manifest = {
        "model": arguments.model, "shard": arguments.shard, "num_shards": arguments.num_shards,
        "eval_sets": arguments.eval_sets, "n_targets": len(pending),
        "recipe": {"method": "rollout+resample+tiebreak", "n_rollouts": arguments.n_rollouts,
                   "temperature": 1.0, "top_p": 0.95, "top_k": -1, "dtype": dtype,
                   "backend": arguments.backend,
                   "min_seq_separation": figlib.MIN_SEPARATION},
        "seconds": time.time() - started,
        "git": figlib.git_state(), "machine": figlib.machine(),
        "packages": figlib.package_versions(), "inputs": inputs.records,
    }
    (out / f"manifest-shard{arguments.shard:02d}.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    log(f"shard {arguments.shard} done in {(time.time() - started) / 60:.1f} min")
    return 0


def fan_out(arguments) -> int:
    """One child process per GPU, each pinned to its own card."""
    children = []
    for shard in range(arguments.gpus):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard))
        argv = [sys.executable, str(Path(__file__).resolve()),
                "--model", arguments.model, "--tag", arguments.tag,
                "--shard", str(shard), "--num-shards", str(arguments.gpus),
                "--n-rollouts", str(arguments.n_rollouts),
                "--backend", arguments.backend,
                "--gpu-memory-utilization", str(arguments.gpu_memory_utilization),
                *(["--force"] if arguments.force else []),
                *(["--limit", str(arguments.limit)] if arguments.limit else []),
                "--eval-sets", *arguments.eval_sets]
        stdout = (OUT_ROOT / arguments.tag / f"shard{shard:02d}.log")
        stdout.parent.mkdir(parents=True, exist_ok=True)
        handle = stdout.open("w")
        children.append((shard, subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT),
                         handle))
        log(f"started shard {shard} on GPU {shard} -> {stdout}")
    failures = []
    for shard, child, handle in children:
        code = child.wait()
        handle.close()
        log(f"shard {shard} exited {code}")
        if code != 0:
            failures.append(shard)
    if failures:
        raise SystemExit(f"shards {failures} failed; their logs are in {OUT_ROOT / arguments.tag}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--validate", action="store_true",
                        help=f"score {VALIDATION_MODEL}, whose numbers #245 already published")
    parser.add_argument("--tag", default=None, help="output subdirectory (default: the model)")
    parser.add_argument("--eval-sets", nargs="+",
                        default=["eval-val", "eval-test", "eval-denovo"])
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--gpus", type=int, default=0, help="fan out over this many GPUs")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="smoke test: this many targets")
    parser.add_argument("--force", action="store_true", help="rescore proteins already written")
    arguments = parser.parse_args()

    arguments.model = arguments.model or (VALIDATION_MODEL if arguments.validate
                                          else DEFAULT_MODEL)
    arguments.tag = arguments.tag or (arguments.model + ("-validate" if arguments.validate else ""))
    if arguments.gpus:
        return fan_out(arguments)
    return run_shard(arguments)


if __name__ == "__main__":
    sys.exit(main())
