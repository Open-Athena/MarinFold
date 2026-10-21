"""Make a lossless-for-bf16-inference checkpoint export beside its source in S3.

Run on a CPU in cw-us-east-02a. The original export is fp32, but the established
inference recipe loads bf16. Casting before transfer halves bytes without
changing any weight vLLM uses. Every serialized tensor is checked after saving.
"""

import hashlib
import json
from pathlib import Path

import fsspec
import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Cast and verify in-region, preserving model structure and tokenizer."""
    plan = json.loads((HERE / "plan.json").read_text())
    original = plan["source"]["uri"]
    target = ("s3://marin-us-east-02a/MarinFold/exp345/checkpoints/" +
              plan["checkpoint"]["run_name"] + "/step-479417/bf16")
    fs, root = fsspec.core.url_to_fs(original)
    _, out = fsspec.core.url_to_fs(target)
    entries, total_size, tensors_checked = [], 0, 0
    for entry in plan["files"]:
        name = entry["name"]
        path = HERE / name
        info = fs.info(root + "/" + name)
        assert info["size"] == entry["size"] and info["ETag"].strip('"') == entry["digest"]
        path.write_bytes(fs.cat_file(root + "/" + name))
        if name.endswith(".safetensors"):
            tensors = load_file(str(path))
            cast = {key: tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor for key, tensor in tensors.items()}
            converted = HERE / ("bf16-" + name)
            save_file(cast, str(converted), metadata={"format": "pt"})
            reread = load_file(str(converted))
            assert set(reread) == set(cast)
            for key, tensor in cast.items():
                assert torch.equal(reread[key], tensor)
                total_size += reread[key].numel() * reread[key].element_size()
                tensors_checked += 1
            path = converted
        elif name == "config.json":
            config = json.loads(path.read_text())
            config["torch_dtype"] = "bfloat16"
            path.write_text(json.dumps(config, indent=2) + "\n")
        elif name == "model.safetensors.index.json":
            index = json.loads(path.read_text())
            index["metadata"]["total_size"] = total_size
            path.write_text(json.dumps(index, indent=2) + "\n")
        raw = path.read_bytes()
        fs.pipe_file(out + "/" + name, raw)
        entries.append({"name": name, "size": len(raw), "digest": hashlib.sha256(raw).hexdigest(), "digest_kind": "sha256"})
        print(f"Verified and saved {name}: {len(raw):,} bytes", flush=True)
    plan["original_source"] = plan["source"]
    plan["original_files"] = plan["files"]
    plan["source"] = {"kind": "coreweave-s3", "uri": target}
    plan["files"] = entries
    plan["dtype_conversion"] = {"from": "float32", "to": "bfloat16", "tensors_roundtrip_verified": tensors_checked,
                                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    fs.pipe_file(out + "/conversion-identity.json", (json.dumps(plan, indent=2) + "\n").encode())
    print(target + "/conversion-identity.json", flush=True)


if __name__ == "__main__":
    main()
