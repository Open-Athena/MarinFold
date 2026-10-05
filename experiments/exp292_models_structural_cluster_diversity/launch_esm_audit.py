"""Launch a bounded CPU curation audit beside the original ESM membership.

Uses the existing exp91 instance profile and no inbound access or new IAM grants.
Only code and two small metadata files cross regions; the 10.77 GB membership
stays in us-west-2. The instance terminates after the job or after three hours.
"""

import argparse
import hashlib
import json
import shlex
import tarfile
from pathlib import Path

import boto3

BUCKET = "marinfold-exp91-usw2"
IMAGE = "ami-04678417fc39d7171"
SUBNET = "subnet-00caee8a9828c5c0c"
PROFILE = "marinfold-exp91-instance-profile"


def main() -> None:
    """Stage reviewable inputs and optionally launch the region-pinned worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument(
        "--instance-type", choices=["m7i.4xlarge", "m7i.8xlarge"], default="m7i.4xlarge"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--per-bin", type=int, default=4)
    parser.add_argument("--seed", type=int, default=292)
    args = parser.parse_args()
    if not args.run_name.replace("-", "").isalnum():
        raise ValueError("Run name must contain only letters, digits and hyphens")
    args.record_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    bundle = args.record_dir / "source.tar.gz"
    worker_config = args.record_dir / "worker-config.json"
    worker_config.write_text(
        json.dumps(
            {
                "arguments": [
                    "--workers",
                    str(args.workers),
                    "--per-bin",
                    str(args.per_bin),
                    "--seed",
                    str(args.seed),
                ]
            },
            indent=2,
        )
        + "\n"
    )
    with tarfile.open(bundle, "w:gz") as archive:
        for name in [
            "sample_esm.py",
            "structure_audit.py",
            "aws_audit_bootstrap.py",
            "pyproject.toml",
            "uv.lock",
        ]:
            archive.add(root / name, arcname=name)
        archive.add(args.plan, arcname="inputs/plan.parquet")
        archive.add(args.retained, arcname="inputs/retained.parquet")
        archive.add(worker_config, arcname="worker-config.json")
        provenance = args.plan.with_name("sampling.json")
        if provenance.exists():
            archive.add(provenance, arcname="inputs/sampling.json")
    if bundle.stat().st_size > 100_000_000:
        raise ValueError("Expected a small metadata/code bundle below 100 MB")
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    prefix = "exp292/audits/" + args.run_name
    source_key = prefix + "/source.tar.gz"
    results_prefix = prefix + "/results"
    download = f"import boto3,tarfile; boto3.client('s3',region_name='us-west-2').download_file({BUCKET!r},{source_key!r},'/opt/exp292/source.tar.gz'); tarfile.open('/opt/exp292/source.tar.gz').extractall('/opt/exp292',filter='data')"
    user_data = f"""#!/bin/bash
set -eu
shutdown -h +180
trap 'shutdown -h now' EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv
python3 -m venv /opt/exp292-bootstrap
/opt/exp292-bootstrap/bin/pip install --quiet uv boto3
mkdir -p /opt/exp292
/opt/exp292-bootstrap/bin/python -c {shlex.quote(download)}
/opt/exp292-bootstrap/bin/python /opt/exp292/aws_audit_bootstrap.py --bucket {shlex.quote(BUCKET)} --prefix {shlex.quote(results_prefix)}
"""
    (args.record_dir / "user-data.sh").write_text(user_data)
    config = {
        "ImageId": IMAGE,
        "InstanceType": args.instance_type,
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
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": args.run_name},
                    {"Key": "Experiment", "Value": "exp292"},
                ],
            }
        ],
        "ClientToken": hashlib.sha256((args.run_name + digest).encode()).hexdigest(),
    }
    record = {
        "region": "us-west-2",
        "source": f"s3://{BUCKET}/{source_key}",
        "results": f"s3://{BUCKET}/{results_prefix}",
        "bundle_sha256": digest,
        "bundle_bytes": bundle.stat().st_size,
        "config": config,
    }
    (args.record_dir / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    if not args.launch:
        return
    s3 = boto3.client("s3", region_name="us-west-2")
    s3.upload_file(str(bundle), BUCKET, source_key)
    ec2 = boto3.client("ec2", region_name="us-west-2")
    response = ec2.run_instances(**config, UserData=user_data)
    record["instance_id"] = response["Instances"][0]["InstanceId"]
    (args.record_dir / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Launched " + record["instance_id"], flush=True)


if __name__ == "__main__":
    main()
