"""Publish the evaluated exp277 weights and tokenizer, verifying source identity.

Usage: uv run --frozen python publish_checkpoint.py --submit
The CPU job reads 5.89 GB in-region and publishes one public HF copy.
"""

import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import fsspec
from huggingface_hub import HfFileSystem

HERE = Path(__file__).resolve().parent
RUN = "contacts-v1-exp277-m2-p06-full-epoch-1.5B"
PREFIX = f"checkpoints/{RUN}/hf/step-266344"
DEST = f"buckets/open-athena/MarinFold/{PREFIX}"
SOURCE = f"marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot/runs/{RUN}/hf/step-266344"


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    """Hash a file without loading its weights into memory."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        while block := source.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def publish() -> None:
    """Verify the evaluated source, repair its config, and publish all files."""
    # The submission bundles this existing library file without its heavy runtime.
    from model_config import repair_rope

    spec = json.loads((HERE / "checkpoint_source.json").read_text())
    fs = fsspec.filesystem("s3")
    listing = {Path(row["name"]).name: row for row in fs.ls(SOURCE, detail=True)}
    for file in spec["checkpoint_files"]:
        row = listing[file["name"]]
        if row["size"] != file["size"] or row["ETag"].strip('"') != file["digest"]:
            raise ValueError(f"Source identity mismatch: {file['name']}")
    manifest = {"source": spec, "published_uri": f"hf://{DEST}", "files": {}}
    hub = HfFileSystem(token=os.environ["HF_TOKEN"])
    with tempfile.TemporaryDirectory(prefix="exp277-publish-") as temporary:
        local = Path(temporary)
        for file in spec["checkpoint_files"]:
            name = file["name"]
            path = local / name
            fs.get_file(f"{SOURCE}/{name}", str(path))
            if path.stat().st_size != file["size"]:
                raise ValueError(f"Incomplete download: {name}")
            source_sha = digest_file(path)
            if name == "config.json":
                raw = json.loads(path.read_text())
                fixed = repair_rope(raw)
                if fixed["rope_theta"] != 500000 or fixed["vocab_size"] != 2845:
                    raise ValueError("Unexpected trained RoPE or vocabulary")
                path.write_text(json.dumps(fixed, indent=2) + "\n")
                manifest["rope_theta"] = fixed["rope_theta"]
            if name == "tokenizer.json":
                vocab = json.loads(path.read_text())["model"]["vocab"]
                if len(vocab) != 2845 or vocab["<contact>"] != 5:
                    raise ValueError("Unexpected contacts-v1 tokenizer")
            if name == "tokenizer_config.json":
                raw = json.loads(path.read_text())
                if raw.get("tokenizer_class") == "TokenizersBackend":
                    raw["tokenizer_class"] = "PreTrainedTokenizerFast"
                    path.write_text(json.dumps(raw, indent=2) + "\n")
            published_sha = digest_file(path)
            with path.open("rb") as reader, hub.open(f"{DEST}/{name}", "wb") as writer:
                shutil.copyfileobj(reader, writer, length=8 << 20)
            if hub.info(f"{DEST}/{name}")["size"] != path.stat().st_size:
                raise ValueError(f"Incomplete upload: {name}")
            manifest["files"][name] = {
                "size": path.stat().st_size,
                "sha256": published_sha,
                "source_sha256": source_sha,
                "source_etag": file["digest"],
            }
            print(
                f"Published {name}: {path.stat().st_size} bytes, sha256={published_sha}",
                flush=True,
            )
        manifest["config_repair_sha256"] = digest_file(HERE / "model_config.py")
        with hub.open(f"{DEST}/publication_manifest.json", "w") as writer:
            json.dump(manifest, writer, indent=2)
    print(
        f"COMPLETE https://huggingface.co/buckets/open-athena/MarinFold/tree/{PREFIX}",
        flush=True,
    )


def submit(attempt: int) -> None:
    """Submit publication to an in-region batch CPU job."""
    root = HERE.parents[1]
    credentials = configparser.ConfigParser()
    credentials.read(Path.home() / ".aws/credentials")
    cw = credentials["cw"]
    token = (
        os.environ.get("HF_TOKEN")
        or (Path.home() / ".cache/huggingface/token").read_text().strip()
    )
    env = {
        "HF_TOKEN": token,
        "FSSPEC_S3": json.dumps(
            {
                "key": cw["aws_access_key_id"],
                "secret": cw["aws_secret_access_key"],
                "endpoint_url": "http://cwlota.com",
                "config_kwargs": {"s3": {"addressing_style": "virtual"}},
            }
        ),
    }
    with tempfile.TemporaryDirectory(prefix="exp277-publish-bundle-") as temporary:
        bundle = Path(temporary)
        shutil.copyfile(__file__, bundle / "publish_checkpoint.py")
        shutil.copyfile(
            HERE / "data/checkpoint_source.json", bundle / "checkpoint_source.json"
        )
        shutil.copyfile(
            root / "marinfold/marinfold/inference/_config.py",
            bundle / "model_config.py",
        )
        (bundle / "pyproject.toml").write_text(
            '[project]\nname="exp277-publish"\nversion="0.1.0"\nrequires-python=">=3.12"\ndependencies=["s3fs==2026.2.0","huggingface-hub==1.27.0"]\n'
        )
        argv = [
            "/home/bizon/git/marin-freshiris/.venv/bin/iris",
            "--cluster=marin",
            "job",
            "run",
            "--target-cluster",
            "cw-us-east-02a",
            "--priority",
            "batch",
            "--enable-extra-resources",
            "--no-wait",
            "--job-name",
            f"exp277-publish-step266344-a{attempt:02d}",
            "--cpu",
            "4",
            "--memory",
            "16GB",
            "--disk",
            "32GB",
            "--timeout",
            "7200",
        ]
        for key, value in env.items():
            argv.extend(["-e", key, value])
        argv.extend(["--", "python", "publish_checkpoint.py"])
        if subprocess.run(argv, cwd=bundle, check=False).returncode:
            raise RuntimeError("Iris publication submission failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.submit:
        submit(args.attempt)
    else:
        publish()
