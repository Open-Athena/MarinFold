#!/usr/bin/env bash
# Mirror a published HF checkpoint export into GCS us-central1, in bf16, from
# inside the marin cluster.
#
# Same reasoning as stage_from_hf_cw.py (the CoreWeave version): the workstation
# uplink is ~2.2 MB/s, so pushing the 3B export from here costs ~45 minutes,
# while a cluster pod pulls the same bytes from the HF CDN and writes them to
# in-region GCS in about a minute.
#
# Only the **weights** are mirrored. The small files that needed real repair —
# the transformers-5 -> 4.57 config downgrade and the tokenizer-class fix from
# prepare_hf_export.py — are uploaded from the workstation (a few hundred kB),
# because doing that repair on the pod would need transformers 4.57, and the
# marin image ships the 5.x that wrote the broken shapes in the first place.
#
# The pod uses `uv run --with` rather than the workspace env: it needs only
# torch (for the bf16 cast), huggingface_hub and gcsfs, and `--no-sync` skips
# marin's own multi-minute sync entirely.
#
#   ./stage_from_hf_gcs.sh <hf-repo> <path-in-repo> <gs://dest>
set -euo pipefail

REPO="${1:?usage: stage_from_hf_gcs.sh <hf-repo> <path-in-repo> <gs://dest>}"
PATH_IN_REPO="${2:?}"
DEST="${3:?}"
IRIS=${IRIS_BIN:-/home/bizon/git/marin-freshiris/.venv/bin/iris}
NAME="exp169-stage-$(basename "$DEST" | tr '_' '-')"

read -r -d '' PODSCRIPT <<'PYEOF' || true
import json, os, sys, time
from pathlib import Path

import fsspec
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

repo, path_in_repo, dest = sys.argv[1], sys.argv[2], sys.argv[3]

t0 = time.time()
local = snapshot_download(repo, allow_patterns=[f"{path_in_repo}/*"],
                          local_dir="/tmp/hf_src", max_workers=8)
src = Path(local) / path_in_repo
print(f"[stage] downloaded {repo}/{path_in_repo} in {time.time() - t0:.0f}s: "
      f"{sorted(p.name for p in src.iterdir())}", flush=True)

index = json.loads((src / "model.safetensors.index.json").read_text())
fs, root = fsspec.core.url_to_fs(dest)
total = 0
for shard in sorted(set(index["weight_map"].values())):
    t1 = time.time()
    recast = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
              for k, v in load_file(src / shard).items()}
    out = Path("/tmp/bf16") / shard
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(recast, str(out), metadata={"format": "pt"})
    size = out.stat().st_size
    total += size
    fs.put_file(str(out), f"{root.rstrip('/')}/{shard}")
    out.unlink()
    print(f"[stage]   {shard}: {size / 2**30:.3f} GiB in {time.time() - t1:.0f}s", flush=True)

index["metadata"]["total_size"] = total
with fsspec.open(f"{dest.rstrip('/')}/model.safetensors.index.json", "w") as fh:
    json.dump(index, fh)

listing = sorted(os.path.basename(f) for f in fs.ls(root, detail=False))
print(f"[stage] DONE {total / 2**30:.2f} GiB in {(time.time() - t0) / 60:.1f} min -> {dest}")
print(f"[stage] prefix now holds: {listing}", flush=True)
PYEOF

BOOTSTRAP="mkdir -p /tmp/stage && cat > /tmp/stage/worker.py <<'EOS'
${PODSCRIPT}
EOS
uv run --no-project --with torch --with huggingface_hub --with gcsfs --with safetensors \\
    python /tmp/stage/worker.py '${REPO}' '${PATH_IN_REPO}' '${DEST}'"

# `iris job run` bundles the CWD as the job's workspace, so it must be a small,
# dedicated directory — submitting from a large tree (or from /tmp) uploads the
# whole thing and appears to hang.
WORKSPACE=${STAGE_WORKSPACE:-/tmp/exp169_submit}
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"
exec "$IRIS" --cluster=marin job run \
    --job-name "$NAME" --no-wait --no-sync --enable-extra-resources \
    --priority batch --region us-central1 \
    --cpu 4 --memory 32GB --disk 64GB \
    -- bash -lc "$BOOTSTRAP"
