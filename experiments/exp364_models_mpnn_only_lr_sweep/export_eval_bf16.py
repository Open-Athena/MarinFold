"""Save the TPU model in the evaluator's existing dtype before one RNO mirror."""

import hashlib
import json
from pathlib import Path

import fsspec
import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Cast float weights exactly as vLLM does, retaining source/checkpoint identity."""
    plan = json.loads((HERE / "data/contact_eval/plan.json").read_text())
    checkpoint = next(
        c for c in plan["checkpoints"] if c["trial"] == "lr5e-4-cooldown-tpu"
    )
    fs = fsspec.filesystem("s3", use_listings_cache=False)
    source_manifest = f"{plan['root']}/manifests/{checkpoint['label']}.json"
    source = json.loads(fs.cat_file(source_manifest))
    destination = checkpoint["hf_export"].replace("/hf/", "/hf-bf16/")
    scratch = Path("/tmp/exp364-bf16")
    scratch.mkdir(exist_ok=True)
    files = []
    parameter_bytes = 0
    torch.set_num_threads(4)
    for obj in source["files"]:
        local = scratch / obj["name"]
        fs.get_file(checkpoint["hf_export"] + "/" + obj["name"], str(local))
        with local.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != obj["sha256"]:
                raise ValueError(f"Source content changed: {obj['name']}")
        if local.suffix == ".safetensors":
            tensors = load_file(str(local))
            converted = {}
            for name, tensor in tensors.items():
                value = (
                    tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
                )
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite converted parameter: {name}")
                converted[name] = value
                parameter_bytes += value.numel() * value.element_size()
            save_file(converted, str(local), metadata={"format": "pt"})
            reloaded = load_file(str(local))
            if set(reloaded) != set(tensors) or any(
                not torch.equal(reloaded[name], value)
                for name, value in converted.items()
            ):
                raise ValueError("Round-trip converted tensors differ")
            del tensors, converted, reloaded
        elif obj["name"] == "config.json":
            config = json.loads(local.read_text())
            config["torch_dtype"] = "bfloat16"
            local.write_text(json.dumps(config, indent=2) + "\n")
    index_path = scratch / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["metadata"]["total_size"] = parameter_bytes
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    for local in sorted(scratch.iterdir()):
        with local.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        fs.put_file(str(local), destination + "/" + local.name)
        if fs.info(destination + "/" + local.name)["size"] != local.stat().st_size:
            raise ValueError("Incomplete BF16 export upload")
        files.append(
            {"name": local.name, "size": local.stat().st_size, "sha256": digest}
        )
    additional_bytes = sum(obj["size"] for obj in files)
    total_egress = checkpoint["model_egress_bytes"] + additional_bytes
    if total_egress >= 10_000_000_000:
        raise ValueError("Combined TPU evaluation transfers exceed the budget")
    manifest_uri = f"{plan['root']}/manifests/{checkpoint['label']}.bf16.json"
    checkpoint = {
        **checkpoint,
        "hf_export": destination,
        "export_manifest": manifest_uri,
        "evaluation_cluster": "cw-rno2a",
        "evaluation_region": "RNO2A",
        "model_egress_bytes": total_egress,
        "rno_mirror_bytes": additional_bytes,
    }
    manifest = {
        "checkpoint": checkpoint,
        "files": files,
        "dtype": "bfloat16",
        "inference_dtype": "bfloat16",
        "source_manifest": source_manifest,
        "source_dtype": "float32",
        "all_parameters_finite": True,
        "conversion": "floating tensors cast to existing inference dtype; integer tensors preserved; round-trip equality checked",
    }
    with fs.open(manifest_uri, "wt") as stream:
        json.dump(manifest, stream, indent=2)
    print("VERIFIED_BF16", manifest_uri, "mirror_bytes", additional_bytes, flush=True)


if __name__ == "__main__":
    main()
