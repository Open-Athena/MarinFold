"""Launch one or more ESM metadata-curation shards in AWS us-west-2."""

import argparse
import hashlib
import json
import shlex
import tarfile
from pathlib import Path

import boto3

from aws_fleet import drain, max_concurrent

BUCKET = "marinfold-exp91-usw2"
IMAGE = "ami-04678417fc39d7171"
SUBNET = "subnet-00caee8a9828c5c0c"
PROFILE = "marinfold-exp91-instance-profile"
INSTANCE_TYPE = "m7i.4xlarge"
LOCATED_PREFIX = "exp292/production-v1/esm/locator/located"


def user_data(source_key: str, output_prefix: str, shard: str, max_clusters: int | None) -> str:
    """Render one shard-specific, self-terminating bootstrap."""
    download = (
        "import boto3,tarfile; "
        "boto3.client('s3',region_name='us-west-2').download_file("
        f"{BUCKET!r},{source_key!r},'/opt/exp292/source.tar.gz'); "
        "tarfile.open('/opt/exp292/source.tar.gz').extractall('/opt/exp292',filter='data')"
    )
    cap = f" --max-clusters {max_clusters}" if max_clusters is not None else ""
    return f"""#!/bin/bash
set -eu
shutdown -h +300
trap 'shutdown -h now' EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv
python3 -m venv /opt/exp292-bootstrap
/opt/exp292-bootstrap/bin/pip install --quiet uv boto3
mkdir -p /opt/exp292
/opt/exp292-bootstrap/bin/python -c {shlex.quote(download)}
/opt/exp292-bootstrap/bin/python /opt/exp292/aws_metadata_bootstrap.py --bucket {shlex.quote(BUCKET)} --located-prefix {shlex.quote(LOCATED_PREFIX)} --output-prefix {shlex.quote(output_prefix)} --shard {shlex.quote(shard)}{cap}
"""


def main() -> None:
    """Stage frozen code once and optionally launch requested shards."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--max-clusters", type=int)
    parser.add_argument(
        "--max-concurrent", type=int, default=max_concurrent(INSTANCE_TYPE)
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if not args.run_name.replace("-", "").isalnum():
        raise ValueError("Run name must contain only letters, digits and hyphens")
    if any(len(shard) != 2 or any(c not in "0123456789abcdef" for c in shard) for shard in args.shard):
        raise ValueError("Shards must be two lowercase hexadecimal characters")
    args.record_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    bundle = args.record_dir / "source.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for name in [
            "curate_esm_metadata.py",
            "sequence_exclusion.py",
            "build_esm_plan.py",
            "locate_esm_rows.py",
            "production_policy.py",
            "structure_audit.py",
            "aws_metadata_bootstrap.py",
            "pyproject.toml",
            "uv.lock",
        ]:
            archive.add(root / name, arcname=name)
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    source_key = args.output_prefix + "/launches/" + args.run_name + "/source.tar.gz"
    base_config = {
        "ImageId": IMAGE,
        "InstanceType": INSTANCE_TYPE,
        "MinCount": 1,
        "MaxCount": 1,
        "SubnetId": SUBNET,
        "Placement": {"AvailabilityZone": "us-west-2a"},
        "IamInstanceProfile": {"Name": PROFILE},
        "InstanceInitiatedShutdownBehavior": "terminate",
        "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled"},
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": 100,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ],
    }
    record = {
        "status": "prepared",
        "region": "us-west-2",
        "source": f"s3://{BUCKET}/{source_key}",
        "located": f"s3://{BUCKET}/{LOCATED_PREFIX}",
        "results": f"s3://{BUCKET}/{args.output_prefix}",
        "bundle_sha256": digest,
        "bundle_bytes": bundle.stat().st_size,
        "shards": args.shard,
        "max_concurrent": args.max_concurrent,
        "max_clusters": args.max_clusters,
        "instances": [],
    }
    launch_path = args.record_dir / "launch.json"
    launch_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    if not args.launch:
        return
    s3 = boto3.client("s3", region_name="us-west-2")
    s3.upload_file(str(bundle), BUCKET, source_key)
    ec2 = boto3.client("ec2", region_name="us-west-2")

    def persist() -> None:
        launch_path.write_text(json.dumps(record, indent=2) + "\n")

    def launch_one(shard: str) -> str:
        name = f"{args.run_name}-{shard}"
        config = {
            **base_config,
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": name},
                        {"Key": "Experiment", "Value": "exp292"},
                        {"Key": "Stage", "Value": "esm-metadata"},
                        {"Key": "Shard", "Value": shard},
                    ],
                }
            ],
            "ClientToken": hashlib.sha256((name + digest).encode()).hexdigest(),
        }
        response = ec2.run_instances(
            **config,
            UserData=user_data(
                source_key, args.output_prefix, shard, args.max_clusters
            ),
        )
        return response["Instances"][0]["InstanceId"]

    def note(shard: str, instance_id: str) -> None:
        # Persist before the next request so a limit refusal cannot strand an
        # already-running worker with no provenance.
        record["instances"].append({"shard": shard, "instance_id": instance_id})
        persist()
        print(f"launched shard={shard} {instance_id}", flush=True)

    record["status"] = "launching"
    persist()
    drain(
        ec2,
        args.shard,
        launch_one,
        concurrency=args.max_concurrent,
        poll_seconds=args.poll_seconds,
        on_launch=note,
    )
    record["status"] = "launched"
    persist()
    print(json.dumps({"launched": len(record["instances"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
