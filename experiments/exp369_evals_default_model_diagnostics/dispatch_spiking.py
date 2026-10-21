"""Prepare or launch independent batch-priority CoreWeave spiking shards."""

import argparse
import gzip
import hashlib
import io
import json
import shlex
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
IMAGE = "vllm/vllm-openai:v0.9.2"
ROOT = "s3://marin-us-east-02a/MarinFold/exp369/spiking-v2"


def bundle(plan: Path) -> bytes:
    """Package exact worker inputs and the local contacts-v1 implementation."""
    files = {name: HERE / name for name in ("spiking_worker.py", "protocol.py", "stage_model.py")}
    files["plan.json"] = plan
    files["checkpoint_manifest.json"] = HERE.parent / "exp277_models_single_mpnn_pilot/data/publication_manifest.json"
    library = REPO / "marinfold/marinfold"
    files.update({"library/marinfold/" + str(path.relative_to(library)): path
                  for path in library.rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.suffix in (".py", ".yaml")})
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, path in sorted(files.items()):
            payload = path.read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return gzip.compress(stream.getvalue(), mtime=0)


def bootstrap(shard: int, shards: int, uri: str, digest: str, out: str, smoke: bool) -> str:
    """Construct the foreign-container bootstrap with its dependency closure."""
    extra = " --stem 7qp5_A --counts 0,8,32" if smoke else ""
    return f"""set -euo pipefail
unset FSSPEC_S3_CONFIG_KWARGS
VLLM_PY=$(command -v python3)
uv pip install --python "$VLLM_PY" --quiet --no-deps \\
  fsspec==2026.1.0 s3fs==2026.1.0 aiobotocore==2.26.0 botocore==1.41.5 \\
  aiohttp==3.14.3 aiohappyeyeballs==2.7.1 aioitertools==0.13.0 aiosignal==1.4.0 \\
  attrs==26.1.0 frozenlist==1.8.0 idna==3.20 jmespath==1.1.0 multidict==6.9.1 \\
  propcache==0.5.4 python-dateutil==2.9.0.post0 six==1.17.0 urllib3==2.8.0 \\
  wrapt==1.17.4rc1 yarl==1.25.1 gemmi==0.7.5
mkdir -p /tmp/exp369
"$VLLM_PY" - <<'PY'
import fsspec,hashlib,io,tarfile
with fsspec.open({uri!r}, 'rb') as stream:
    payload = stream.read()
if hashlib.sha256(payload).hexdigest() != {digest!r}:
    raise ValueError('Workspace digest mismatch')
with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
    archive.extractall('/tmp/exp369')
PY
export PYTHONPATH=/tmp/exp369/library:/tmp/exp369
export VLLM_LOGGING_LEVEL=ERROR
"$VLLM_PY" /tmp/exp369/stage_model.py --manifest /tmp/exp369/checkpoint_manifest.json --out /tmp/exp369-model
export VLLM_PORT=$("$VLLM_PY" -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
exec "$VLLM_PY" /tmp/exp369/spiking_worker.py --model /tmp/exp369-model \\
  --plan /tmp/exp369/plan.json --out {shlex.quote(out)} --shard {shard}/{shards}{extra}
"""


def main() -> None:
    """Dry-run by default; --submit uploads the frozen bundle and launches jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("/data/exp369/spiking_plan.json"))
    parser.add_argument("--shards", type=int, default=12)
    parser.add_argument("--only-shards", help="Comma-separated shard indices to submit from the full interleaved partition")
    parser.add_argument("--iris-config", type=Path, help="Cluster connection config matching the installed Iris revision")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cases", action="store_true", help="Run the six verified AFDB/PDB cases")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.shards < 1:
        raise ValueError("At least one shard required")
    if args.cases and args.smoke:
        raise ValueError("Cases and spiking smoke are different cohorts")
    if args.cases:
        args.plan = HERE / "data/afdb_inference_plan.json"
    payload = bundle(args.plan)
    digest = hashlib.sha256(payload).hexdigest()
    uri = f"{ROOT}/inputs/{digest}.tar.gz"
    shards = 1 if args.smoke or args.cases else args.shards
    indices = list(range(shards)) if args.only_shards is None else list(map(int, args.only_shards.split(",")))
    if len(set(indices)) != len(indices) or not indices or any(not 0 <= i < shards for i in indices):
        raise ValueError("Invalid subset of shard indices")
    phase = "cases" if args.cases else "smoke" if args.smoke else "full"
    out = f"{ROOT}/{phase}"
    commands = {index: bootstrap(index, shards, uri, digest, out, args.smoke) for index in indices}
    plan = json.loads(args.plan.read_text())
    summary = {"phase": phase, "n_gpus": len(indices), "n_shards": shards, "shard_indices": indices, "priority": 3, "image": IMAGE,
               "bundle_sha256": digest, "bundle_bytes": len(payload), "checkpoint_bytes_per_gpu": 5885618986,
               "checkpoint_transfer": "CoreWeave S3 to CoreWeave H100 only", "out": out,
               "eval_set": plan["eval_set"], "n_targets": len(plan["targets"]), "submit": args.submit}
    (HERE / "data" / f"dispatch_{phase}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not args.submit:
        return
    # These cluster libraries are needed only for an actual submission.
    import s3fs
    from fray.iris_backend import FrayIrisClient
    from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
    from iris.cli.connect import open_iris_client

    fs = s3fs.S3FileSystem(profile="cw", client_kwargs={"endpoint_url": "https://cwobject.com"},
                         config_kwargs={"s3": {"addressing_style": "virtual"}})
    with fs.open(uri.removeprefix("s3://"), "wb") as stream:
        stream.write(payload)
    ids = []
    with open_iris_client(config_file=args.iris_config,
                          cluster_name=None if args.iris_config else "cw-rno2a", workspace=None) as iris:
        client = FrayIrisClient.from_iris_client(iris)
        for index, command in commands.items():
            request = JobRequest(name=f"exp369-{phase}-{digest[:8]}-s{index}of{shards}",
                                 entrypoint=Entrypoint.from_binary("bash", ["-lc", command]),
                                 resources=ResourceConfig.with_gpu("H100", count=1, image=IMAGE, cpu=8, ram="64g", disk="128g"),
                                 environment=create_environment(docker_image=IMAGE, env_vars={}, setup_scripts=[]),
                                 replicas=1, processes_per_task=1, priority=3,
                                 max_retries_failure=0, max_retries_preemption=10)
            job = client.submit(request)
            ids.append(str(job.job_id))
            (HERE / "data" / f"jobs_{phase}.json").write_text(json.dumps(summary | {"job_ids": ids}, indent=2) + "\n")
            print(job.job_id, flush=True)


if __name__ == "__main__":
    main()
