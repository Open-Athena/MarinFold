"""Isolate the current HF bucket client from vLLM's older Transformers runtime."""

import argparse
from pathlib import Path

from huggingface_hub import HfFileSystem


def main() -> None:
    """Publish or test one exact object without changing model dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exists")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination")
    args = parser.parse_args()
    fs = HfFileSystem()
    if args.exists:
        print("true" if fs.exists(args.exists.removeprefix("hf://")) else "false")
        return
    fs.put_file(str(args.source), args.destination.removeprefix("hf://"))


if __name__ == "__main__":
    main()
