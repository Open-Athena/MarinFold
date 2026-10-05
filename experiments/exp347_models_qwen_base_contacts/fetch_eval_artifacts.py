"""Fetch small evaluation artifacts through one of this experiment's live Iris pods."""

import argparse
import base64
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--files", nargs="+", default=["summary.json", "metrics.csv", "timings.csv"]
    )
    args = parser.parse_args()
    code = "\n".join(
        [
            "import base64,json,fsspec",
            f"prefix={args.prefix!r}",
            f"names={args.files!r}",
            "result={}",
            "for name in names:",
            "    with fsspec.open(prefix+'/'+name,'rb') as handle:",
            "        result[name]=base64.b64encode(handle.read()).decode()",
            "print('ARTIFACT_JSON='+json.dumps(result))",
        ]
    )
    command = [
        "uv",
        "run",
        "--project",
        "/home/bizon/git/marin",
        "--package",
        "marin-iris",
        "iris",
        "--cluster",
        "marin",
        "task",
        "exec",
        args.task,
        "--",
        ".venv/bin/python",
        "-c",
        code,
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    lines = [
        line.removeprefix("ARTIFACT_JSON=")
        for line in result.stdout.splitlines()
        if line.startswith("ARTIFACT_JSON=")
    ]
    if len(lines) != 1:
        raise ValueError("Expected one artifact response from the task")
    for name, encoded in json.loads(lines[0]).items():
        path = args.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(encoded))
        print(path)


if __name__ == "__main__":
    main()
