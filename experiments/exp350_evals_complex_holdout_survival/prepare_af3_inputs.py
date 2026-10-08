"""Create native-dimer AF3 inputs with cached ColabFold MSAs or no MSA."""

import argparse
import hashlib
import json
from pathlib import Path

import requests
from helico.msa_server import fetch_paired_and_unpaired

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Preserve chain order and record MSA provenance for every target."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--msa-cache", type=Path, required=True)
    parser.add_argument("--single-sequence", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    targets = json.loads(
        (HERE / "data/four_model_v1/predictor_inputs.json").read_text()
    )
    targets.sort(key=lambda t: t["L"])
    if args.limit:
        targets = targets[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    for target in targets:
        output = args.output / (target["stem"] + ".json")
        if output.exists():
            continue
        sequences = target["chain_sequences"]
        if args.single_sequence:
            paired, unpaired = ["", ""], ["", ""]
        else:
            key = hashlib.sha256(
                "\n".join(sorted(set(sequences))).encode()
            ).hexdigest()[:16]
            # Recover matching small, public archives before querying the server.
            for mode in ("np", "p"):
                dest = args.msa_cache / key / mode / "out.tar.gz"
                if dest.exists():
                    continue
                url = f"https://huggingface.co/datasets/timodonnell/helico-data/resolve/main/benchmarks/FoldBench/foldbench-msas-server/{key}/{mode}/out.tar.gz"
                response = requests.get(url, timeout=60)
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(response.content)
            paired, unpaired = fetch_paired_and_unpaired(sequences, args.msa_cache)
        proteins = [
            {
                "protein": {
                    "id": chain,
                    "sequence": seq,
                    "unpairedMsa": u,
                    "pairedMsa": p,
                    "templates": [],
                }
            }
            for chain, seq, p, u in zip(
                ("A", "B"), sequences, paired, unpaired, strict=True
            )
        ]
        record = {
            "name": target["stem"],
            "dialect": "alphafold3",
            "version": 1,
            "modelSeeds": [0],
            "sequences": proteins,
        }
        output.write_text(json.dumps(record, indent=2) + "\n")
        print(
            target["stem"],
            [x.count(">") for x in unpaired],
            [x.count(">") for x in paired],
            flush=True,
        )


if __name__ == "__main__":
    main()
