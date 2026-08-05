# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Stage an exp163 checkpoint + eval set from CoreWeave S3 to GCS, in **bf16**.

Why this exists: the CoreWeave H100 batch band starved for ~10h across two days, so
multi-draft generation moves to the marin **iris TPU** pool (v5p-8). Those pods read
**GCS**, not CoreWeave S3, and vLLM-on-TPU additionally requires the weights to be
**bf16 on disk** -- an fp32 export makes the ragged-paged attention kernel throw
``Expected ... v.dtype=float32`` (Qwen3 QK-norm yields bf16 k but fp32 v), and
``LLM(dtype="bfloat16")`` alone does NOT fix it.

Runs **cloud-side** on a marin CPU pod: the workstation uplink is ~2.5MB/s, so moving
5.9GB down and 2.9GB back up locally would take ~40min. The pod needs CoreWeave S3
credentials injected (``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` /
``AWS_ENDPOINT_URL``); GCS comes from the pod's own identity.

Three things are copied VERBATIM rather than re-saved, each for a reason that cost a
debug cycle in an earlier experiment:

* ``config.json`` -- carries ``rope_theta`` + ``rope_scaling``. Re-saving via
  transformers 5.12 rewrites the Llama3 rope under the newer ``rope_parameters`` key,
  and marin's older vLLM reads the old keys -> null -> silently DEFAULT rope, which
  degrades worse as sequence length grows. (This is the same bug that invalidated the
  exp163 R-precision numbers; the S3 configs have since been repaired in place.)
* ``tokenizer*.json`` / ``special_tokens_map.json`` -- transformers 5.12 re-saves a
  ``TokenizersBackend`` class the marin worker cannot import.
* the safetensors **index** is rewritten only in its ``total_size`` field, since the
  weight->file map is unchanged by a dtype cast.

Usage (as a marin CPU job; see dispatch notes in the module docstring of
``dispatch_rollouts.py``)::

    python -m stage_v3_to_gcs \\
        --model-src  s3://marin-us-east-02a/MarinFold/exp163/checkpoints/<run>/hf/step-404 \\
        --model-dst  gs://marin-us-east5/MarinFold/exp163/tpu/<run>-bf16/step-404 \\
        --data-src   s3://marin-us-east-02a/MarinFold/exp163/eval554 \\
        --data-dst   gs://marin-us-east5/MarinFold/exp163/eval554
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

VERBATIM = ("config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "generation_config.json")


def _s3():
    import s3fs
    return s3fs.S3FileSystem(config_kwargs={"s3": {"addressing_style": "virtual"}})


def _gcs():
    import gcsfs
    return gcsfs.GCSFileSystem()


def _strip(u: str) -> str:
    return u.split("://", 1)[1].rstrip("/") if "://" in u else u.rstrip("/")


def _fs(url: str):
    """Filesystem for a URL. The SOURCE is not always S3: checkpoints trained on the
    marin TPU pool are already on GCS, and only need the bf16 cast + rope repair."""
    return _gcs() if url.startswith("gs://") else _s3()


def stage_model(src: str, dst: str) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    s3, gcs = _fs(src), _gcs()
    S, D = _strip(src), _strip(dst)
    names = [p.rsplit("/", 1)[-1] for p in s3.ls(S)]
    shards = sorted(n for n in names if n.endswith(".safetensors"))
    if not shards:
        raise SystemExit(f"no safetensors under {src}")
    print(f"[stage] {len(shards)} shard(s): {shards}", flush=True)

    total_bytes = 0
    with tempfile.TemporaryDirectory() as tmp:
        for sh in shards:
            lp = os.path.join(tmp, sh)
            print(f"[stage] download {sh}", flush=True)
            s3.get(f"{S}/{sh}", lp)
            tensors = load_file(lp)
            out, n_cast = {}, 0
            for k, v in tensors.items():
                if v.dtype in (torch.float32, torch.float64, torch.float16):
                    out[k] = v.to(torch.bfloat16)
                    n_cast += 1
                else:
                    out[k] = v
            op = os.path.join(tmp, f"bf16-{sh}")
            # metadata format=pt keeps transformers/vLLM loaders happy
            save_file(out, op, metadata={"format": "pt"})
            sz = os.path.getsize(op)
            total_bytes += sz
            print(f"[stage] {sh}: cast {n_cast}/{len(tensors)} -> {sz/1e9:.2f} GB, upload", flush=True)
            gcs.put(op, f"{D}/{sh}")
            os.remove(lp)
            os.remove(op)

    for name in VERBATIM:
        if s3.exists(f"{S}/{name}"):
            with s3.open(f"{S}/{name}", "rb") as fh:
                blob = fh.read()
            with gcs.open(f"{D}/{name}", "wb") as fh:
                fh.write(blob)
            print(f"[stage] verbatim {name} ({len(blob)} B)", flush=True)

    idx = "model.safetensors.index.json"
    if s3.exists(f"{S}/{idx}"):
        with s3.open(f"{S}/{idx}", "rb") as fh:
            j = json.load(fh)
        j.setdefault("metadata", {})["total_size"] = total_bytes
        with gcs.open(f"{D}/{idx}", "wb") as fh:
            fh.write(json.dumps(j, indent=2).encode())
        print(f"[stage] index rewritten: total_size={total_bytes:,}", flush=True)

    # levanter's HF export ALWAYS writes the Llama3 rope under the newer
    # ``rope_parameters`` key and leaves top-level ``rope_theta``/``rope_scaling``
    # null. Readers older than transformers 5.x (marin's vLLM, and the transformers
    # 4.53.1 on the CUDA eval image) see null and silently fall back to DEFAULT rope
    # -- a 50x wrong base frequency whose error grows with sequence distance. That
    # bug invalidated an entire round of exp163 R-precision numbers.
    #
    # So translate it here, at the one place every checkpoint passes through. This
    # only RE-SPELLS the same rope: the values are copied out of rope_parameters and
    # the result is asserted non-null before we continue.
    cfg = json.loads(gcs.cat(f"{D}/config.json"))
    if cfg.get("rope_theta") is None and cfg.get("rope_parameters"):
        rp = dict(cfg["rope_parameters"])
        theta = rp.pop("rope_theta", None)
        if theta is None:
            raise SystemExit(f"[stage] FATAL: {D}/config.json rope_parameters has no "
                             "rope_theta to translate.")
        cfg["rope_theta"] = theta
        cfg["rope_scaling"] = rp
        cfg.pop("rope_parameters", None)
        with gcs.open(f"{D}/config.json", "wb") as fh:
            fh.write(json.dumps(cfg, indent=2).encode())
        print(f"[stage] REPAIRED rope: rope_parameters -> rope_theta={theta} + "
              f"rope_scaling={rp}", flush=True)
    if cfg.get("rope_theta") is None:
        raise SystemExit(f"[stage] FATAL: {D}/config.json has no top-level rope_theta; "
                         "marin's vLLM would fall back to default rope.")
    print(f"[stage] verify rope_theta={cfg['rope_theta']} rope_scaling={cfg.get('rope_scaling')}",
          flush=True)
    print(f"[stage] model done -> {dst}", flush=True)


def stage_data(src: str, dst: str) -> None:
    """targets.parquet + the per-target prompts dir (small, but 554 objects)."""
    s3, gcs = _s3(), _gcs()
    S, D = _strip(src), _strip(dst)
    if s3.exists(f"{S}/targets.parquet"):
        with s3.open(f"{S}/targets.parquet", "rb") as fh:
            blob = fh.read()
        with gcs.open(f"{D}/targets.parquet", "wb") as fh:
            fh.write(blob)
        print(f"[stage] targets.parquet ({len(blob)/1e6:.1f} MB)", flush=True)
    prompts = s3.find(f"{S}/prompts")
    print(f"[stage] {len(prompts)} prompt objects", flush=True)
    for n, p in enumerate(prompts):
        with s3.open(p, "rb") as fh:
            blob = fh.read()
        with gcs.open(f"{D}/prompts/{p.rsplit('/', 1)[-1]}", "wb") as fh:
            fh.write(blob)
        if (n + 1) % 100 == 0:
            print(f"[stage]   {n+1}/{len(prompts)}", flush=True)
    print(f"[stage] data done -> {dst}", flush=True)


def mirror(src_glob: str, dst: str) -> None:
    """Copy an arbitrary S3 glob to a GCS prefix, preserving basenames.

    Used for the contacts-v1 val split, which the TPU training run needs on GCS
    (the marin pods cannot see CoreWeave S3).
    """
    s3, gcs = _s3(), _gcs()
    pat = _strip(src_glob)
    files = sorted(s3.glob(pat))
    if not files:
        raise SystemExit(f"no objects matched {src_glob}")
    D = _strip(dst)
    print(f"[stage] mirror {len(files)} object(s) -> {dst}", flush=True)
    for n, f in enumerate(files):
        with s3.open(f, "rb") as fh:
            blob = fh.read()
        with gcs.open(f"{D}/{f.rsplit('/', 1)[-1]}", "wb") as fh:
            fh.write(blob)
        if (n + 1) % 10 == 0 or n + 1 == len(files):
            print(f"[stage]   {n+1}/{len(files)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-src")
    ap.add_argument("--model-dst")
    ap.add_argument("--data-src")
    ap.add_argument("--data-dst")
    ap.add_argument("--mirror-src", help="arbitrary S3 glob to copy verbatim")
    ap.add_argument("--mirror-dst")
    a = ap.parse_args()
    if a.model_src:
        stage_model(a.model_src, a.model_dst)
    if a.data_src:
        stage_data(a.data_src, a.data_dst)
    if a.mirror_src:
        mirror(a.mirror_src, a.mirror_dst)
    print("[stage] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
