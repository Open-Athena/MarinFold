# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Move the eval assets GCS -> CoreWeave S3 from a GCS-local marin pod (#160).

CoreWeave pods cannot read GCS, so a CoreWeave eval needs its models and targets
in CoreWeave object storage. Copying them from the workstation costs ~16 minutes
per model over a ~2.9 MB/s uplink; copying them from a **marin CPU pod** — which
is GCS-local and has a datacenter link to ``cwobject.com`` — costs about a
minute. This is ``stage_checkpoint.py``'s pattern, generalised to a list of
directories and made into its own launcher.

Credentials never reach the process table: ``iris job run`` is invoked with
``check=False`` and its argv is never printed, because ``CalledProcessError``
renders the full command line and these values would land in a traceback and in
any log that captures it.

    set -a; source ~/.config/marin/cw-rno2a.env; set +a
    uv run --no-project python stage_to_cw.py --assets models/exp120-base,eval_targets.parquet
"""
from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
from pathlib import Path

MARIN = Path(os.environ.get("MARIN_CHECKOUT", "/home/bizon/git/marin-freshiris"))
IRIS = os.environ.get("IRIS_BIN", str(MARIN / ".venv/bin/iris"))

GCS = ("gs://marin-us-central1/protein-structure/MarinFold/"
       "exp160_backtracking_training/eval")
S3 = ("s3://marin-us-east-02a/protein-structure/MarinFold/"
      "exp160_backtracking_training/eval")

# Runs on the pod. Kept inline (base64) rather than bundled so the job needs no
# workspace: it is a handful of boto3 calls and gcsfs reads.
POD_SCRIPT = '''
import os, sys, time
from pathlib import Path
import boto3, gcsfs
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

gcs_root, s3_root, assets = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
# "src=dst" renames on the way across; bare "src" keeps its name. The run's own
# export is named for its step, which is not the name the eval arm is keyed on.
assets = [tuple(a.split("=", 1)) if "=" in a else (a, a) for a in assets]
fs = gcsfs.GCSFileSystem()
bucket, prefix = s3_root[len("s3://"):].split("/", 1)
s3 = boto3.client("s3", endpoint_url="https://cwobject.com",
                  aws_access_key_id=os.environ["CW_KEY_ID"],
                  aws_secret_access_key=os.environ["CW_KEY_SECRET"],
                  config=Config(s3={"addressing_style": "virtual"}), region_name="auto")
transfer = TransferConfig(multipart_threshold=64 * 1024**2,
                          multipart_chunksize=64 * 1024**2, max_concurrency=8)

for asset, dst_name in assets:
    src = f"{gcs_root.rstrip('/')}/{asset}"
    keys = [k for k in fs.find(src[len("gs://"):]) if not k.endswith("/")]
    if not keys:
        raise SystemExit(f"nothing under {src}")
    print(f"[stage] {asset}: {len(keys)} object(s)", flush=True)
    for k in keys:
        rel = k[len(src[len("gs://"):]):].lstrip("/")
        local = Path("/tmp/stage") / (rel or Path(asset).name)
        local.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        fs.get(k, str(local))
        key = (f"{prefix.rstrip('/')}/{dst_name}/{rel}" if rel
               else f"{prefix.rstrip('/')}/{dst_name}")
        s3.upload_file(str(local), bucket, key, Config=transfer)
        mb = local.stat().st_size / 1e6
        print(f"[stage]   {key}  {mb:.1f} MB in {time.time() - t0:.0f}s", flush=True)
        local.unlink()
print("[stage] DONE", flush=True)
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True,
                    help="comma-separated paths relative to --gcs; 'src=dst' renames")
    ap.add_argument("--gcs", default=GCS)
    ap.add_argument("--s3", default=S3)
    ap.add_argument("--job-name", default="exp160-stage-to-cw")
    ap.add_argument("--zone", default="us-central1-a")
    a = ap.parse_args()

    for var in ("CW_KEY_ID", "CW_KEY_SECRET"):
        if not os.environ.get(var):
            raise SystemExit(f"{var} not set — `set -a; source ~/.config/marin/cw-rno2a.env`")

    script_b64 = base64.b64encode(POD_SCRIPT.encode()).decode()
    bootstrap = f"""
set -euo pipefail
echo "[stage] host=$(hostname)"
mkdir -p /tmp/stage
echo {script_b64} | base64 -d > /tmp/stage_to_cw.py
exec uv run --no-project --with gcsfs --with boto3 python /tmp/stage_to_cw.py \\
    '{a.gcs}' '{a.s3}' '{a.assets}'
""".strip()

    command = [
        IRIS, "--cluster=marin", "job", "run",
        "--job-name", a.job_name, "--no-wait", "--enable-extra-resources",
        "--priority", "interactive", "--zone", a.zone,
        "--cpu", "4", "--memory", "16GB", "--disk", "64GB",
        "-e", "CW_KEY_ID", os.environ["CW_KEY_ID"],
        "-e", "CW_KEY_SECRET", os.environ["CW_KEY_SECRET"],
        "--", "bash", "-lc", bootstrap,
    ]
    print(f"[stage] {a.gcs} -> {a.s3}\n[stage] assets: {a.assets}")
    # check=False on purpose: CalledProcessError would render argv, credentials
    # included, into the traceback.
    result = subprocess.run(command, cwd=MARIN, check=False)
    if result.returncode != 0:
        print(f"[stage] iris job run failed (exit {result.returncode}) — argv withheld "
              "because it carries CoreWeave credentials", file=sys.stderr)
        return result.returncode
    print(f"[stage] submitted /bizon/{a.job_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
