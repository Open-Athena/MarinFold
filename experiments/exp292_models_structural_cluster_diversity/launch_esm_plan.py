"""Launch the ESM production planning pass in AWS us-west-2."""

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
    """Stage immutable code and metadata, then optionally launch one planner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--droplist", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if not args.run_name.replace("-", "").isalnum():
        raise ValueError("Run name must contain only letters, digits and hyphens")
    args.record_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    bundle = args.record_dir / "source.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for name in [
            "build_esm_plan.py",
            "aws_plan_bootstrap.py",
            "pyproject.toml",
            "uv.lock",
        ]:
            archive.add(root / name, arcname=name)
        archive.add(args.droplist, arcname="inputs/droplist_final.parquet")
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    prefix = "exp292/production-v1/esm/plan-v5"
    source_key = prefix + "/launches/" + args.run_name + "/source.tar.gz"
    download = (
        "import boto3,tarfile; "
        "boto3.client('s3',region_name='us-west-2').download_file("
        f"{BUCKET!r},{source_key!r},'/opt/exp292/source.tar.gz'); "
        "tarfile.open('/opt/exp292/source.tar.gz').extractall('/opt/exp292',filter='data')"
    )
    user_data = f"""#!/bin/bash
set -eu
shutdown -h +420
trap 'shutdown -h now' EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv
python3 -m venv /opt/exp292-bootstrap
/opt/exp292-bootstrap/bin/pip install --quiet uv boto3
mkdir -p /opt/exp292
/opt/exp292-bootstrap/bin/python -c {shlex.quote(download)}
/opt/exp292-bootstrap/bin/python /opt/exp292/aws_plan_bootstrap.py --bucket {shlex.quote(BUCKET)} --prefix {shlex.quote(prefix)}
"""
    (args.record_dir / "user-data.sh").write_text(user_data)
    config = {
        "ImageId": IMAGE,
        "InstanceType": "m7i.8xlarge",
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
                    "VolumeSize": 300,
                    "VolumeType": "gp3",
                    "Iops": 6000,
                    "Throughput": 500,
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
                    {"Key": "Stage", "Value": "esm-plan"},
                ],
            }
        ],
        "ClientToken": hashlib.sha256((args.run_name + digest).encode()).hexdigest(),
    }
    record = {
        "status": "prepared",
        "region": "us-west-2",
        "source": f"s3://{BUCKET}/{source_key}",
        "results": f"s3://{BUCKET}/{prefix}",
        "bundle_sha256": digest,
        "bundle_bytes": bundle.stat().st_size,
        "config": config,
    }
    launch_path = args.record_dir / "launch.json"
    launch_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    if not args.launch:
        return
    s3 = boto3.client("s3", region_name="us-west-2")
    s3.upload_file(str(bundle), BUCKET, source_key)
    response = boto3.client("ec2", region_name="us-west-2").run_instances(
        **config, UserData=user_data
    )
    record["status"] = "launched"
    record["instance_id"] = response["Instances"][0]["InstanceId"]
    launch_path.write_text(json.dumps(record, indent=2) + "\n")
    print("Launched " + record["instance_id"], flush=True)


if __name__ == "__main__":
    main()
