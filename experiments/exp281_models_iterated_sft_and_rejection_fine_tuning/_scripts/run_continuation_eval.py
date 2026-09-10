"""Run both format modes sequentially on one GPU, sharing one model download."""

import argparse
import json
import subprocess
from pathlib import Path

import fsspec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "configs/continuation_plan.json").read_text())
    jobs = [job for job in plan if job["group"] in (1, 2)]
    models = {job["arguments"][job["arguments"].index("--model") + 1]
              for job in jobs if job["stage"] == "generate"}
    if len(models) != 1:
        raise ValueError("The combined evaluation must share one immutable checkpoint")
    commands = []
    for job in jobs:
        entrypoint = "generate.py" if job["stage"] == "generate" else "evaluate.py"
        commands.append(["uv", "run", "--project", str(root), "--locked", "--no-dev",
                         "--extra", "generation", "python", str(root / entrypoint), *job["arguments"]])
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return
    # Generation's URI-keyed cache persists across both processes. Optimizer
    # state is excluded, so this is one sub-10-GB model mirror, not two copies.
    model = models.pop()
    with fsspec.open(f"{model}/_SUCCESS.json") as handle:
        manifest = json.load(handle)
    model_bytes = sum(record["bytes"] for name, record in manifest.items() if name != "trainer.pt")
    if model_bytes >= 9_000_000_000:
        raise ValueError("Model exceeds the bounded evaluation transfer budget")
    print(json.dumps({"model": model, "model_transfer_bytes": model_bytes, "shared_download": True}), flush=True)
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
