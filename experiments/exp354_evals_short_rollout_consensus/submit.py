"""Build an immutable eval-val-only bundle and submit root GPU shards."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
IRIS = "/home/bizon/git/marin-freshiris/.venv/bin/iris"
S3 = "s3://marin-us-east-02a/MarinFold/exp354_evals_short_rollout_consensus"
REVISION = "d1bea417a64cc042ad931422200c3edeb873f2e0"


def prepare(bundle: Path) -> dict:
    """Filter immutable #245 inputs and record all source/payload digests."""
    source = ROOT / "experiments/exp245_evals_foldbench_held_out_monomers/data"
    expected = {
        "eval_targets_foldbench_monomers.parquet": "2eb4f1fee148fe2d6601bd171ef6e9431b96f38c82eaed1ad119a069a13f1fb8",
        "gt_universe_scored.jsonl": "f30c23e3d2fbab245755fc01548388b41730ddfa45da87325539698cadb153e5",
        "eval_sets.csv": "b13d060a091240921bc8466acecc9fa6ccbb45a56de4efda02cd903f6abf9861",
    }
    for name, digest in expected.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"input identity mismatch: {name}")
    manifest = pd.read_csv(source / "eval_sets.csv")
    stems = set(manifest.loc[manifest.eval_set == "eval-val", "stem"])
    assert len(stems) == 97
    targets = {r["stem"]: r for r in pq.read_table(source / "eval_targets_foldbench_monomers.parquet").to_pylist()}
    ground_truth = [json.loads(line) for line in (source / "gt_universe_scored.jsonl").read_text().splitlines()]
    selected = [{**record, "input_seq": targets[record["stem"]]["input_seq"]}
                for record in ground_truth if record["stem"] in stems]
    assert len(selected) == 97
    for record in selected:
        assert record["L"] == len(record["input_seq"])
    (bundle / "targets.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selected))
    for name in ("worker.py", "contacts.py"):
        shutil.copy2(HERE / name, bundle / name)
    shutil.copy2(ROOT / "experiments/exp89_evals_contacts_v1_model_on_eval_set/compute_metrics.py", bundle)
    shutil.copy2(ROOT / "experiments/exp277_models_single_mpnn_pilot/data/publication_manifest.json", bundle)
    provenance = dict(eval_set="eval-val", n_proteins=97, source_sha256=expected,
        marinfold_revision=REVISION,
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
        bundle_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir()})
    (bundle / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="v1-20261009")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num-shards", type=int, default=12)
    parser.add_argument("--shards", type=int, nargs="+")
    parser.add_argument("--target-cluster", default="cw-us-east-02a")
    parser.add_argument("--gpu", default="H100")
    parser.add_argument("--image", default="vllm/vllm-openai:v0.9.2")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    phase = "smoke" if args.smoke else "production"
    out = f"{S3}/{args.run_id}/{phase}"
    with tempfile.TemporaryDirectory(prefix="exp354-bundle-") as temporary:
        bundle = Path(temporary)
        provenance = prepare(bundle)
        shards = [0] if args.smoke else (args.shards or list(range(args.num_shards)))
        jobs = []
        for shard in shards:
            job_name = f"exp354-{args.run_id}-{phase}-{shard:02d}"
            shell = "\n".join([
                "set -euo pipefail",
                "unset FSSPEC_S3_CONFIG_KWARGS",
                "VLLM_PY=''",
                'for candidate in /app/.venv/bin/python /usr/local/bin/python /usr/bin/python3 /opt/venv/bin/python python3 python; do if "$candidate" -c "import vllm" >/dev/null 2>&1; then VLLM_PY="$candidate"; break; fi; done',
                'test -n "$VLLM_PY"',
                '"$VLLM_PY" -m pip install --quiet "fsspec==2026.1.0" "s3fs==2026.1.0" "aiobotocore==2.26.0" "pyarrow>=20" "scikit-learn>=1.6" "pandas>=2"',
                f'"$VLLM_PY" -m pip install --quiet --no-deps "marinfold @ git+https://github.com/Open-Athena/MarinFold.git@{REVISION}#subdirectory=marinfold"',
                'export VLLM_PORT=$("$VLLM_PY" -c \'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()\')',
                f'exec "$VLLM_PY" worker.py --out {out} --shard {shard} --num-shards {args.num_shards}' + (' --limit 1' if args.smoke else ''),
            ])
            command = [IRIS, "--cluster=marin", "job", "run", "--target-cluster", args.target_cluster,
                "--priority", "batch", "--enable-extra-resources", "--user", "bizon",
                "--job-name", job_name, "--gpu", f"{args.gpu}x1", "--cpu", "8", "--memory", "64GB",
                "--disk", "32GB", "--max-retries", "2", "--timeout", "7200", "--no-wait",
                "--no-sync", "--task-image", args.image,
                "--", "bash", "-lc", shell]
            print(f"Submit {job_name}: {out}", flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=bundle, check=True)
            jobs.append(f"/bizon/{job_name}")
        provenance.update(jobs=jobs, output_uri=out, target_cluster=args.target_cluster,
                          num_shards=args.num_shards, phase=phase, gpu=args.gpu, image=args.image)
        destination = HERE / "data" / f"{args.run_id}-{phase}-submission.json"
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
