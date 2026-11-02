# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish exp163's arm-F export to an HF model repo — issue #200.

WHY A REPO AND NOT THE BUCKET. The checkpoint is already public in the
open-athena *bucket*, but `vLLMInferenceContext.__init__` resolves its tokenizer
with `levanter.tokenizers.load_tokenizer(VLLMEngineConfig.model_name)`, which
accepts a local directory, a `mirror://` ref, or an HF Hub **repo id** — and a
bucket path is none of those. vLLM streams the weights from GCS happily, so the
failure would land only on the tokenizer, inside a rollout worker, after the gang
had scheduled.

SOURCE IS THE HF BUCKET, NOT GCS. The two copies of arm F are not
interchangeable. `gs://.../exp163/tpu/tpuF-bf16/step-404` carries the tokenizer
levanter exported, where id 7 still spells `<contacts-and-distances-v1>`; only
the bucket copy carries exp163's renamed tokenizer and its
`special_tokens_map.json`. Weights are the same bf16 2.94 GB either way, and
nothing in exp200 reads the sentinel by string, so this is about publishing an
artifact a reader can trust rather than about behaviour.

Runs cloud-side: the workstation uplink is ~2.5 MB/s and this moves ~2.9 GB.

Two things are verified before anything is uploaded, because both are silent when
wrong and both have cost this project a round of results before:

* **rope_theta.** Levanter writes Llama3 rope under `rope_parameters` and leaves
  the top-level key null; readers older than transformers 5 then fall back to
  default rope, a 50x wrong base frequency.
* **the RENAMED tokenizer.** Token id 7 must spell `<contacts-v1.multi>`. The
  published contacts-v1 tokenizer spells it `<contacts-and-distances-v1>`, and
  shipping that one would leave the multi-draft mode sentinel pointing at a token
  the model never trained on in-document — generating fluent, wrong output.
"""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import fsspec

MULTI_TOKEN = "<contacts-v1.multi>"
MULTI_ID = 7


def stage(src: str, dest: Path) -> list[str]:
    """Copy a remote checkpoint directory to local disk.

    Handles `hf://buckets/<org>/<bucket>/<prefix>` through the bucket API rather
    than fsspec: `snapshot_download` does not see buckets at all, and the generic
    filesystem path is the one that historically resolves `buckets/...` as a
    dataset repo and 404s.
    """
    t0 = time.time()
    if src.startswith("hf://buckets/"):
        from huggingface_hub import HfApi

        rest = src[len("hf://buckets/") :].rstrip("/")
        org, bucket, prefix = rest.split("/", 2)
        api = HfApi()
        paths = [p.path for p in api.list_bucket_tree(f"{org}/{bucket}") if p.path.startswith(prefix)]
        if not paths:
            raise SystemExit(f"no objects under {src}")
        # token=False: the bucket is public, and an org-scoped token is not needed
        # to READ it (only bucket writes require one).
        api.download_bucket_files(
            f"{org}/{bucket}", [(path, str(dest / path.rsplit("/", 1)[-1])) for path in paths], token=False
        )
        names = [path.rsplit("/", 1)[-1] for path in paths]
    else:
        fs, _ = fsspec.core.url_to_fs(src)
        names = []
        for path in fs.ls(src.rstrip("/"), detail=False):
            name = path.rsplit("/", 1)[-1]
            if not name:
                continue
            with fs.open(path, "rb") as fh, open(dest / name, "wb") as out:
                while chunk := fh.read(32 << 20):
                    out.write(chunk)
            names.append(name)
    for name in names:
        print(f"[publish] staged {name} ({(dest / name).stat().st_size / 1e6:.0f} MB)", flush=True)
    print(f"[publish] staged {len(names)} files in {time.time() - t0:.0f}s", flush=True)
    return names


def verify(local: Path) -> dict:
    """Refuse to publish a checkpoint that would silently misbehave."""
    config = json.loads((local / "config.json").read_text())
    if config.get("rope_theta") is None:
        raise SystemExit(
            "FATAL: config.json has no top-level rope_theta (levanter wrote it under "
            "'rope_parameters'). Readers older than transformers 5 fall back to default "
            "rope SILENTLY. Repair with exp163's stage_v3_to_gcs.py first."
        )

    tokenizer_path = local / "tokenizer.json"
    if not tokenizer_path.exists():
        raise SystemExit("FATAL: no tokenizer.json — the whole point of this repo is co-location")
    vocab = json.loads(tokenizer_path.read_text())["model"]["vocab"]
    by_id = {i: t for t, i in vocab.items()}
    spelled = by_id.get(MULTI_ID)
    if spelled != MULTI_TOKEN:
        raise SystemExit(
            f"FATAL: token id {MULTI_ID} spells {spelled!r}, expected {MULTI_TOKEN!r}. "
            "This is the plain contacts-v1 tokenizer, not exp163's renamed one; the "
            "multi-draft sentinel would point at a token the model never trained on."
        )

    print(f"[publish] OK rope_theta={config['rope_theta']} vocab_size={config['vocab_size']} "
          f"id{MULTI_ID}={spelled}", flush=True)
    return config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="stage and verify, do not upload")
    a = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and not a.dry_run:
        raise SystemExit("HF_TOKEN is required (a token with repo.write on the target namespace)")

    with tempfile.TemporaryDirectory(prefix="exp200-publish-") as tmp:
        local = Path(tmp)
        stage(a.src, local)
        config = verify(local)
        if a.dry_run:
            print("[publish] dry run: verified, nothing uploaded")
            return 0

        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)
        print(f"[publish] uploading to https://huggingface.co/{a.repo}", flush=True)
        t0 = time.time()
        api.upload_folder(
            folder_path=str(local),
            repo_id=a.repo,
            repo_type="model",
            commit_message=(
                "exp163 arm F (plm-exp163-refine-cv1-1_5b-lr1e-4-e1-cos-tpuF step-404), "
                "bf16, with the renamed contacts-v1.multi tokenizer"
            ),
        )
        print(f"[publish] uploaded in {time.time() - t0:.0f}s", flush=True)

    files = sorted(f.rfilename for f in api.model_info(a.repo, files_metadata=False).siblings)
    print(f"[publish] repo now holds: {files}", flush=True)
    print(f"[publish] DONE https://huggingface.co/{a.repo} (vocab_size={config['vocab_size']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
