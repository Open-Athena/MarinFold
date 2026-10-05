#!/bin/bash
set -eu
shutdown -h +420
trap 'shutdown -h now' EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv
python3 -m venv /opt/exp292-bootstrap
/opt/exp292-bootstrap/bin/pip install --quiet uv boto3
mkdir -p /opt/exp292
/opt/exp292-bootstrap/bin/python -c 'import boto3,tarfile; boto3.client('"'"'s3'"'"',region_name='"'"'us-west-2'"'"').download_file('"'"'marinfold-exp91-usw2'"'"','"'"'exp292/production-v1/esm/plan-build/launches/exp292-esm-plan-v1/source.tar.gz'"'"','"'"'/opt/exp292/source.tar.gz'"'"'); tarfile.open('"'"'/opt/exp292/source.tar.gz'"'"').extractall('"'"'/opt/exp292'"'"',filter='"'"'data'"'"')'
/opt/exp292-bootstrap/bin/python /opt/exp292/aws_plan_bootstrap.py --bucket marinfold-exp91-usw2 --prefix exp292/production-v1/esm/plan-build
