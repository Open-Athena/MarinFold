# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Drive the two-stage local pilot sweep for issue #262.

Stage 1 tunes the learning rate **per arm** on a short budget. That is not
politeness: removing RoPE changes the scale of the attention logits, so an arm
comparison at one shared learning rate confounds the architecture with the
optimizer, and the issue flags exactly that. Stage 2 then re-runs each arm at
its own best rate over a longer budget with several seeds, which is what gives
the comparison a noise floor to be read against.

Runs are sequential on one GPU and each writes its own JSON, so the sweep is
resumable: an existing result file is skipped.
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

from arms import ARMS

HERE = Path(__file__).resolve().parent


def run_one(*, arm: str, learning_rate: float, seed: int, tokens: int, out: Path, python: str, extra: list[str]) -> Path:
    destination = out / f"{arm}-lr{learning_rate:g}-s{seed}.json"
    if destination.is_file():
        print(f"[sweep] skip {destination.name} (already done)", flush=True)
        return destination
    command = [
        python, str(HERE / "train_pilot.py"),
        "--arm", arm, "--learning-rate", repr(learning_rate), "--seed", str(seed),
        "--tokens", str(tokens), "--out", str(out), *extra,
    ]
    print(f"[sweep] {' '.join(command[2:])}", flush=True)
    started = time.monotonic()
    subprocess.run(command, check=True, cwd=HERE.parent)
    print(f"[sweep] done in {(time.monotonic() - started) / 60:.1f} min", flush=True)
    return destination


def final_loss(path: Path) -> float:
    return json.loads(path.read_text())["final"]["val_nll"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-tokens", type=int, default=100_000_000)
    parser.add_argument("--stage2-tokens", type=int, default=300_000_000)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", type=Path, default=Path("data/pilot"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--stage", choices=["1", "2", "all"], default="all")
    arguments = parser.parse_args()

    stage1 = arguments.out / "stage1"
    stage2 = arguments.out / "stage2"
    stage1.mkdir(parents=True, exist_ok=True)
    stage2.mkdir(parents=True, exist_ok=True)

    best: dict[str, float] = {}
    if arguments.stage in ("1", "all"):
        for arm, learning_rate in itertools.product(ARMS, arguments.learning_rates):
            run_one(
                arm=arm.key, learning_rate=learning_rate, seed=0,
                tokens=arguments.stage1_tokens, out=stage1, python=arguments.python, extra=[],
            )
    for arm in ARMS:
        candidates = {
            learning_rate: stage1 / f"{arm.key}-lr{learning_rate:g}-s0.json"
            for learning_rate in arguments.learning_rates
        }
        available = {rate: path for rate, path in candidates.items() if path.is_file()}
        if not available:
            continue
        best[arm.key] = min(available, key=lambda rate: final_loss(available[rate]))
        print(f"[sweep] {arm.key}: best lr {best[arm.key]:g} "
              f"({', '.join(f'{r:g}={final_loss(p):.4f}' for r, p in sorted(available.items()))})", flush=True)
    (arguments.out / "best_learning_rates.json").write_text(json.dumps(best, indent=2))

    if arguments.stage in ("2", "all"):
        for arm, seed in itertools.product(ARMS, arguments.seeds):
            run_one(
                arm=arm.key, learning_rate=best[arm.key], seed=seed,
                tokens=arguments.stage2_tokens, out=stage2, python=arguments.python, extra=[],
            )


if __name__ == "__main__":
    main()
