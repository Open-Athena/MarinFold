"""Stage explicitly selected pinned base models once into CoreWeave storage."""

import argparse
import hashlib
import json
from pathlib import Path

import fsspec
from common import MODELS, ROOT
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", choices=MODELS, required=True)
    args = parser.parse_args()
    reference = AutoTokenizer.from_pretrained(
        MODELS["0.8B"][0], revision=MODELS["0.8B"][1]
    )
    vocab = reference.get_vocab()
    for size in args.sizes:
        repo, revision = MODELS[size]
        dest = f"{ROOT}/base/{size}"
        fs, prefix = fsspec.core.url_to_fs(dest)
        if fs.exists(prefix + "/staged.json"):
            with fs.open(prefix + "/staged.json") as handle:
                existing = json.load(handle)
            if existing["revision"] != revision:
                raise ValueError("Existing staged revision differs")
            print(f"{size}: already staged {revision}", flush=True)
            continue
        local = Path(
            snapshot_download(
                repo,
                revision=revision,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.txt",
                    "*.jinja",
                    "*.model",
                ],
            )
        )
        tokenizer = AutoTokenizer.from_pretrained(local)
        if (
            tokenizer.get_vocab() != vocab
            or tokenizer.backend_tokenizer.to_str()
            != reference.backend_tokenizer.to_str()
        ):
            raise ValueError(f"{repo} tokenizer differs; token caches cannot be shared")
        files = []
        for file in sorted(local.rglob("*")):
            if file.is_file():
                relative = str(file.relative_to(local))
                with file.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                fs.put_file(str(file), prefix + "/" + relative)
                files.append(
                    {"name": relative, "bytes": file.stat().st_size, "sha256": digest}
                )
        with fs.open(prefix + "/staged.json", "w") as handle:
            json.dump(
                {"repo": repo, "revision": revision, "files": files}, handle, indent=2
            )
        print(
            f"{size}: staged {revision}, {sum(f['bytes'] for f in files)} bytes",
            flush=True,
        )


if __name__ == "__main__":
    main()
