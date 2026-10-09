"""Submit the in-region exporter without printing authentication material."""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from submit import HERE, IRIS, S3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="v1-20261009")
    parser.add_argument("--phase", choices=["smoke", "production"], default="production")
    parser.add_argument("--target-cluster", default="cw-us-east-02a")
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    token = subprocess.check_output(["hf", "auth", "token"], text=True).strip()
    destination = f"hf://buckets/open-athena/MarinFold/data/exp354-short-rollout-consensus/{args.run_id}/{args.phase}"
    command = [IRIS, "--cluster=marin", "job", "run", "--target-cluster", args.target_cluster,
        "--priority", "batch", "--enable-extra-resources", "--user", "bizon",
        "--job-name", f"exp354-export-{args.run_id}-{args.phase}", "--cpu", "2",
        "--memory", "8GB", "--disk", "8GB", "--no-sync", "--no-wait",
        "--max-retries", "2", "--timeout", "3600", "-e", "HF_TOKEN", token,
        "--", "bash", "-lc",
        "unset FSSPEC_S3_CONFIG_KWARGS; uv run --no-project --with 'huggingface_hub==2.1.1' --with 's3fs==2026.1.0' "
        "--with 'fsspec==2026.1.0' --with 'aiobotocore==2.26.0' python export_results.py "
        f"--source {S3}/{args.run_id}/{args.phase} --destination {destination} "
        f"--expected {1 if args.phase == 'smoke' else 97} --wait-seconds {args.wait_seconds}"]
    with tempfile.TemporaryDirectory(prefix="exp354-export-") as directory:
        shutil.copy2(HERE / "export_results.py", Path(directory) / "export_results.py")
        result = subprocess.run(command, cwd=directory, text=True, capture_output=True)
        print((result.stdout + result.stderr).replace(token, "[REDACTED]"))
        if result.returncode:
            raise RuntimeError(f"export submission failed: exit {result.returncode}")


if __name__ == "__main__":
    main()
