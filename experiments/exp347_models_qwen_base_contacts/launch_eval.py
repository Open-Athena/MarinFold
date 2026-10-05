"""Submit explicit co-located eval shards with the unchanged exp89 metric source."""

import argparse
import io
import json
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path

from common import FORMATS

IMAGE = "vllm/vllm-openai@sha256:5f5e535216848d0c52159c8c13a0af04be5f6fe1a84e79914300610796f76d40"
MARINFOLD_REVISION = "62de4334"
HERE = Path(__file__).resolve().parent
METRIC_REFERENCE = (
    HERE.parent / "exp89_evals_contacts_v1_model_on_eval_set/compute_metrics.py"
)


def requirements() -> list[str]:
    """Pin the complete auxiliary dependency closure without resolving the vLLM stack."""
    packages = {
        p["name"]: p for p in tomllib.loads((HERE / "uv.lock").read_text())["package"]
    }
    pending = ["fsspec", "s3fs", "pandas", "scikit-learn", "pyarrow"]
    selected = set()
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        selected.add(name)
        pending.extend(d["name"] for d in packages[name].get("dependencies", []))
    return [f"{name}=={packages[name]['version']}" for name in sorted(selected)]


def build_bundle(destination: Path, targets: Path | None = None) -> None:
    """Bundle only the evaluator, frozen eval-val inputs, and exact metric reference."""
    for name in ["common.py", "eval_contract.py", "eval_worker.py"]:
        shutil.copy2(HERE / name, destination / name)
    shutil.copy2(
        targets or HERE / "data/eval_val.jsonl", destination / "eval_val.jsonl"
    )
    shutil.copy2(METRIC_REFERENCE, destination / "metric_reference.py")
    revision = subprocess.check_output(
        ["git", "rev-parse", MARINFOLD_REVISION], cwd=HERE, text=True
    ).strip()
    archive = subprocess.check_output(
        ["git", "archive", f"{revision}:marinfold"], cwd=HERE.parents[1]
    )
    with tempfile.TemporaryDirectory(prefix="exp347-package-") as source:
        with tarfile.open(fileobj=io.BytesIO(archive)) as handle:
            handle.extractall(source, filter="data")
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(destination), source],
            check=True,
        )
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "export TOKENIZERS_PARALLELISM=false",
            "export OMP_NUM_THREADS=4",
            "eval_python=$(command -v python3)",
            'uv pip install --python "$eval_python" --no-deps '
            + shlex.join(requirements()),
            'uv pip install --python "$eval_python" --no-deps marinfold-*.whl',
            'export VLLM_PORT=$(uv run --no-project --python "$eval_python" python -c \'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()\')',
            'exec uv run --no-project --python "$eval_python" python eval_worker.py --targets eval_val.jsonl --metric-reference metric_reference.py "$@"',
            "",
        ]
    )
    (destination / "bootstrap.sh").write_text(script)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--format", choices=FORMATS, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--job-prefix", required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--reference-e8", action="store_true")
    args = parser.parse_args()
    shards = 1 if args.smoke else args.shards
    args.record.parent.mkdir(parents=True, exist_ok=True)
    jobs = []
    with tempfile.TemporaryDirectory(prefix="exp347-eval-") as directory:
        build_bundle(Path(directory), args.targets)
        for shard in range(shards):
            name = f"{args.job_prefix}-s{shard:02d}"
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
                "job",
                "run",
                "--target-cluster",
                "cw-us-east-02a",
                "--priority",
                "batch",
                "--user",
                "timodonnell",
                "--job-name",
                name,
                "--no-wait",
                "--enable-extra-resources",
                "--gpu",
                "H100x1",
                "--cpu",
                "8",
                "--memory",
                "64GB",
                "--disk",
                "128GB",
                "--no-sync",
                "--timeout",
                "86400",
                "--task-image",
                IMAGE,
                "--",
                "bash",
                "bootstrap.sh",
                "--checkpoint",
                args.checkpoint,
                "--format",
                args.format,
                "--out",
                args.out,
                "--shard",
                str(shard),
                "--shards",
                str(shards),
            ]
            if args.smoke:
                command += ["--limit", "1"]
            if args.reference_e8:
                command += ["--reference-e8", "--max-model-len", "8192"]
            print(shlex.join(command), flush=True)
            subprocess.run(command, cwd=directory, check=True)
            jobs.append({"job": f"/timodonnell/{name}", "command": command})
            args.record.write_text(
                json.dumps(
                    {
                        "checkpoint": args.checkpoint,
                        "format": args.format,
                        "out": args.out,
                        "image": IMAGE,
                        "jobs": jobs,
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
