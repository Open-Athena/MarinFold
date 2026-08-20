#!/usr/bin/env bash
# Mirror the exp199 checkpoint from the open-athena HF **bucket** into GCS
# us-central1, in bf16, from inside the marin cluster.
#
# WHY NOT exp169's stage_from_hf_gcs.sh. That one calls `snapshot_download`,
# which resolves HF *repos*. exp199 lives in the open-athena **bucket**
# (`hf://buckets/open-athena/MarinFold/...`), and buckets are a different
# namespace that snapshot_download cannot see at all — it fails with a repo-not-
# found rather than with anything that points at the real problem. The bucket API
# is `list_bucket_tree` / `download_bucket_files`, and the bucket is anonymously
# readable so `token=False` is correct and avoids picking up a workstation token
# that is scoped to the wrong org.
#
# WHY bf16. TPU parameters are bf16 and vLLM shards the checkpoint as it loads;
# handing it fp32 weights is a known failure rather than a silent cast.
#
# WHY ON A POD. The workstation uplink is ~2.5 MB/s, so pushing a 3 GB export
# from here costs ~20 minutes; a cluster pod pulls from the HF CDN and writes to
# in-region GCS in about a minute.
#
#   ./stage_model_gcs.sh [<bucket-path>] [<gs://dest>]
set -euo pipefail

SRC="${1:-open-athena/MarinFold/checkpoints/prot-exp199-cw-cv1-s02-m1-p06-aug/hf/step-145199}"
DEST="${2:-gs://marin-us-central1/protein-structure/MarinFold/exp208/models/exp199}"
IRIS=${IRIS_BIN:-/home/bizon/git/marin-freshiris/.venv/bin/iris}
NAME="exp208-stage-exp199"

read -r -d '' PODSCRIPT <<'PYEOF' || true
import json, os, sys, time
from pathlib import Path

import fsspec
import torch
from huggingface_hub import download_bucket_files, list_bucket_tree
from safetensors.torch import load_file, save_file

src, dest = sys.argv[1], sys.argv[2]
bucket, prefix = src.split("/", 2)[0] + "/" + src.split("/", 2)[1], src.split("/", 2)[2]

t0 = time.time()
# `prefix` is POSITIONAL (there is no `path=` kwarg), and the listing yields
# BucketFile / BucketFolder -- only the former carries a size.
entries = [e for e in list_bucket_tree(bucket, prefix, recursive=True, token=False)
           if getattr(e, "size", None) is not None]
names = [Path(e.path).name for e in entries]
print(f"[stage] {src} holds {sorted(names)}", flush=True)
# The tokenizer MUST travel with the weights: the eval worker builds prompts with
# AutoTokenizer.from_pretrained(model_dir), and a missing tokenizer.json there is
# a failure minutes into a TPU job rather than at submit time.
assert "tokenizer.json" in names, f"no tokenizer beside the weights in {src}"

# `download_bucket_files` takes explicit (remote, local) PAIRS; there is no
# `local_dir` argument that mirrors the remote tree for you.
staged = Path("/tmp/hf_src")
staged.mkdir(parents=True, exist_ok=True)
download_bucket_files(
    bucket, [(e, str(staged / Path(e.path).name)) for e in entries], token=False)
nbytes = sum(e.size for e in entries)
print(f"[stage] downloaded {nbytes / 2**30:.2f} GiB in {time.time() - t0:.0f}s", flush=True)

fs, root = fsspec.core.url_to_fs(dest)
total = 0
for path in sorted(staged.iterdir()):
    if path.suffix == ".safetensors":
        t1 = time.time()
        recast = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
                  for k, v in load_file(path).items()}
        out = Path("/tmp/bf16") / path.name
        out.parent.mkdir(parents=True, exist_ok=True)
        save_file(recast, str(out), metadata={"format": "pt"})
        total += out.stat().st_size
        fs.put_file(str(out), f"{root.rstrip('/')}/{path.name}")
        out.unlink()
        print(f"[stage]   {path.name}: bf16 in {time.time() - t1:.0f}s", flush=True)
    else:
        # config.json / tokenizer* / generation_config go across VERBATIM. Never
        # re-serialize them: transformers rewrites the rope key on the way out,
        # and a config that states rope only under `rope_parameters` reads as
        # rope_theta 10000 on a 500000-trained model -- silently, and worth
        # 0.76 nats/token (#180, #198).
        fs.put_file(str(path), f"{root.rstrip('/')}/{path.name}")

index_path = staged / "model.safetensors.index.json"
if index_path.exists() and total:
    index = json.loads(index_path.read_text())
    index["metadata"]["total_size"] = total
    with fsspec.open(f"{dest.rstrip('/')}/model.safetensors.index.json", "w") as fh:
        json.dump(index, fh)

cfg = json.loads((staged / "config.json").read_text())
if cfg.get("rope_theta") is None:
    raise SystemExit("[stage] FATAL: staged config.json has no top-level rope_theta")
print(f"[stage] rope_theta={cfg['rope_theta']} vocab_size={cfg.get('vocab_size')}")
print(f"[stage] DONE {total / 2**30:.2f} GiB in {(time.time() - t0) / 60:.1f} min -> {dest}")
print(f"[stage] prefix holds: {sorted(os.path.basename(f) for f in fs.ls(root, detail=False))}",
      flush=True)
PYEOF

BOOTSTRAP="mkdir -p /tmp/stage && cat > /tmp/stage/worker.py <<'EOS'
${PODSCRIPT}
EOS
uv run --no-project --with torch --with numpy --with 'huggingface_hub>=1.5' --with gcsfs \\
    --with safetensors \\
    python /tmp/stage/worker.py '${SRC}' '${DEST}'"

# `iris job run` bundles the CWD as the workspace, so submit from a small,
# dedicated directory -- a large tree (or /tmp) uploads everything and hangs.
WORKSPACE=${STAGE_WORKSPACE:-/tmp/exp208_submit}
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"
exec "$IRIS" --cluster=marin job run \
    --job-name "$NAME" --no-wait --no-sync --enable-extra-resources \
    --priority batch --region us-central1 \
    --cpu 4 --memory 32GB --disk 64GB \
    -- bash -lc "$BOOTSTRAP"
