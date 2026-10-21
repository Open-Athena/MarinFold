"""Check complete raw samples, exact cutoffs, token ranges, and vote artifacts."""

import argparse
import gzip
import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

from contacts import limits, sequence_pairs, snapshots

HERE = Path(__file__).resolve().parent
TOKENIZER_URL = "https://huggingface.co/buckets/open-athena/MarinFold/resolve/checkpoints/contacts-v1-exp277-m2-p06-full-epoch-1.5B/hf/step-266344/tokenizer.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    with urllib.request.urlopen(TOKENIZER_URL) as response:
        content = response.read()
    publication = json.loads((HERE.parent / "exp277_models_single_mpnn_pilot/data/publication_manifest.json").read_text())
    assert hashlib.sha256(content).hexdigest() == publication["files"]["tokenizer.json"]["sha256"]
    tokenizer = json.loads(content)
    vocabulary = dict(tokenizer["model"]["vocab"])
    vocabulary.update({r["content"]: r["id"] for r in tokenizer["added_tokens"]})
    id_to_token = {value: key for key, value in vocabulary.items()}
    invalid = {value for key, value in vocabulary.items()
               if key.startswith("<p") and key.endswith(">") and key[2:-1].isdigit() and int(key[2:-1]) >= 2000}
    files = sorted((args.results / "complete").glob("*.json"))
    assert len(files) == 97
    out_of_ring: Counter = Counter()
    full_unfinished = 0
    identities: set[tuple[str, str, str, str]] = set()
    for completed, file in enumerate(files, start=1):
        result = json.loads(file.read_text())
        stem = result["stem"]
        timing = {r["mode"]: r for r in result["timings"]}
        identities.add((result["worker_sha256"], result["checkpoint"],
                        timing["full_n100"]["vllm_version"], timing["full_n100"]["transformers_version"]))
        length = timing["full_n100"]["n_residues"]
        caps = limits(length)
        with gzip.open(args.results / f"traces/{stem}.json.gz", "rt") as handle:
            trace = json.load(handle)
        assert len(trace["short"]) == 1000 and len(trace["full"]) == 100
        assert len(trace["n_term"]) == 1000
        matrices = {f"short_{label}_n{n}": np.zeros((length, length), np.int16)
                    for label in caps for n in (100, 1000)}
        matrices["full_n100"] = np.zeros((length, length), np.int16)
        for arm in ("short", "full"):
            for index, ids in enumerate(trace[arm]):
                out_of_ring.update(id_to_token[i] for i in ids if i in invalid)
                tokens = [id_to_token[i] for i in ids]
                start = trace["n_term"][index]
                if arm == "short":
                    state, count = snapshots(tokens, list(caps.values()))
                    assert count <= max(caps.values())
                    assert count == max(caps.values()) or trace["short_eos"][index]
                    conditions = [(f"short_{label}_n{n}", state[cap]["pairs"])
                                  for label, cap in caps.items() for n in (100, 1000) if index < n]
                else:
                    if not trace["full_eos"][index]:
                        full_unfinished += 1
                        continue
                    state, _ = snapshots(tokens, [len(ids) + 1])
                    conditions = [("full_n100", state[len(ids) + 1]["pairs"])]
                for mode, pairs in conditions:
                    for i, j in sequence_pairs(pairs, start, length):
                        matrices[mode][i, j] += 1
                        matrices[mode][j, i] += 1
        with np.load(args.results / f"scores/{stem}.npz") as saved:
            assert set(saved.files) == set(matrices)
            for mode, expected in matrices.items():
                if not np.array_equal(saved[mode], expected):
                    raise ValueError(f"raw-sample reconstruction disagrees for {stem}/{mode}; rebuild before analysis")
        if completed % 10 == 0:
            print(f"Verified raw samples and all vote matrices: {completed}/97", flush=True)
    assert len(identities) == 1, "mixed worker/checkpoint/runtime identities"
    report = dict(proteins=97, short_rollouts=97_000, full_rollouts=9_700,
                  full_unfinished=full_unfinished, vote_matrices_verified=97 * 11,
                  out_of_ring_tokens=dict(out_of_ring), tokenizer_sha256=hashlib.sha256(content).hexdigest(),
                  worker_checkpoint_runtime=next(iter(identities)))
    (HERE / "data/artifact_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
