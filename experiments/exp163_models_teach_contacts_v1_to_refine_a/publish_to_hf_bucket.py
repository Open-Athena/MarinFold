# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Publish an exp163 artifact from GCS to the **public** ``open-athena/MarinFold`` bucket.

Why: the trained checkpoints live on private GCS / CoreWeave S3, so a Colab notebook
cannot reach them. The HF bucket is public and anonymously readable, which is what makes
an auth-free notebook possible.

Runs cloud-side on a marin CPU pod -- the workstation uplink is ~2.5MB/s, so pushing a
2.9GB bf16 checkpoint from here would take ~20 minutes. The pod needs ``HF_TOKEN`` (an
**open-athena-scoped** token; a personal-only token 403s on this bucket).

``sync_bucket`` requires a LOCAL directory on one side, so the flow is
GCS -> pod tmpdir -> bucket.

Layout follows the bucket's existing convention -- ``checkpoints/<wandb-run-name>/hf/
step-<N>/`` -- and the tokenizer is co-located with the weights.

    python -m publish_to_hf_bucket \\
        --src gs://marin-us-east5/MarinFold/exp163/tpu/tpuF-bf16/step-404 \\
        --dest-prefix checkpoints/plm-exp163-refine-cv1-1_5b-lr1e-4-e1-cos-tpuF/hf/step-404
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

BUCKET = "open-athena/MarinFold"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="gs:// directory to publish")
    ap.add_argument("--dest-prefix", required=True, help="path inside the bucket")
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import gcsfs
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN must be set (open-athena-scoped; personal tokens 403)")

    fs = gcsfs.GCSFileSystem()
    src = a.src.split("://", 1)[1].rstrip("/")
    files = fs.find(src)
    if not files:
        raise SystemExit(f"nothing at {a.src}")
    total = sum(fs.info(f)["size"] for f in files)
    print(f"[publish] {len(files)} file(s), {total/1e9:.2f} GB from {a.src}", flush=True)

    # Refuse to publish weights whose rope config would silently mis-load for any
    # reader older than transformers 5.x -- the bug that invalidated a whole round of
    # this experiment's numbers. A published artifact must not carry it.
    cfgs = [f for f in files if f.endswith("config.json")]
    if cfgs:
        with fs.open(cfgs[0], "rb") as fh:
            cfg = json.load(fh)
        if cfg.get("rope_theta") is None:
            raise SystemExit("[publish] FATAL: config.json has no top-level rope_theta "
                             "(rope_parameters-only). Stage it through stage_v3_to_gcs "
                             "first, which translates the key.")
        print(f"[publish] rope_theta={cfg['rope_theta']} ok", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        for n, f in enumerate(files, 1):
            local = os.path.join(tmp, f.rsplit("/", 1)[-1])
            fs.get(f, local)
            print(f"[publish]   pulled {n}/{len(files)} {f.rsplit('/', 1)[-1]}", flush=True)
        dest = f"hf://buckets/{a.bucket}/{a.dest_prefix.strip('/')}"
        print(f"[publish] sync -> {dest}", flush=True)
        api = HfApi(token=token)
        plan = api.sync_bucket(source=tmp, dest=dest, dry_run=a.dry_run, verbose=True)
        print(f"[publish] done: {plan}", flush=True)


if __name__ == "__main__":
    main()
