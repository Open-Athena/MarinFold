"""Freeze a matched native-AFDB train/validation sample and canonical eval-val.

AFDB training candidates come from 64 uniformly sampled decontaminated shards;
validation candidates cover all 22 original shards. Select one entry per
structural cluster, then match validation on pLDDT round, length and confidence.
This estimates accuracy on native AFDB examples, not the full native/MPNN/ESM
mixture. The AFDB holdout was never decontaminated against later ESM sources.
"""

import argparse
import hashlib
import json
import math
import random
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from protocol import SEED, parse_training_document, stable_seed

HERE = Path(__file__).resolve().parent
BASE = "https://huggingface.co/buckets/open-athena/MarinFold/resolve/"
TRAIN_PREFIX = "data/document_structures/contacts_v1_decontam/train/"
VAL_PREFIX = "data/document_structures/contacts_v1/val/"
FOLDBENCH = "data/contacts-v1-foldbench-monomers-exp245/"


def fetch(relative: str) -> tuple[bytes, dict]:
    """Cache a bounded source file and record its exact bytes."""
    path = HERE / "scratch" / "sources" / relative
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(BASE + relative, timeout=120) as response:
            blob = response.read(50_000_001)
        if len(blob) > 50_000_000:
            raise ValueError(f"Unexpectedly large source: {relative}")
        path.write_bytes(blob)
    blob = path.read_bytes()
    return blob, {
        "path": relative,
        "url": BASE + relative,
        "bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
    }


def load_shard(relative: str) -> tuple[list[dict], dict]:
    """Parse eligible complete documents and preserve every exclusion count."""
    blob, provenance = fetch(relative)
    source = pq.read_table(pa.BufferReader(blob)).to_pylist()
    candidates = []
    counts: Counter = Counter()
    for index, row in enumerate(source):
        if row["truncated"]:
            counts["truncated"] += 1
            continue
        if not 30 <= row["seq_len"] <= 1000:
            counts["outside_30_1000"] += 1
            continue
        if row["contacts_emitted"] < 5:
            counts["fewer_than_5_contacts"] += 1
            continue
        sequence, contacts = parse_training_document(row)
        record = {
            key: row[key]
            for key in (
                "entry_id",
                "seq_len",
                "global_plddt",
                "round",
                "struct_cluster_id",
                "seq_cluster_id",
                "split",
                "num_tokens",
                "sha1",
            )
        }
        record.update(
            input_seq=sequence,
            contacts=[[i, j, 1.0] for i, j in contacts],
            source=relative,
            source_row=index,
            document_sha256=hashlib.sha256(row["document"].encode()).hexdigest(),
            sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
        )
        candidates.append(record)
    provenance.update(
        rows=len(source), eligible=len(candidates), exclusions=dict(counts)
    )
    print(json.dumps({"source": relative, "eligible": len(candidates)}), flush=True)
    return candidates, provenance


def deduplicate(records: list[dict]) -> list[dict]:
    """Select one seeded representative per structural cluster and sequence."""
    seen_clusters, seen_sequences = set(), set()
    selected = []
    for row in sorted(records, key=lambda r: stable_seed(r["entry_id"], "selection")):
        cluster, sequence = row["struct_cluster_id"], row["sequence_sha256"]
        if cluster in seen_clusters or sequence in seen_sequences:
            continue
        seen_clusters.add(cluster)
        seen_sequences.add(sequence)
        selected.append(row)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--diagnostic-n", type=int, default=32)
    args = parser.parse_args()
    train_shards = sorted(random.Random(SEED).sample(range(2067), 64))
    paths = [
        f"{TRAIN_PREFIX}contacts_v1-{i:05d}-of-02067.parquet" for i in train_shards
    ]
    paths += [f"{VAL_PREFIX}contacts_v1-{i:05d}-of-00022.parquet" for i in range(22)]
    with ThreadPoolExecutor(max_workers=12) as pool:
        loaded = list(pool.map(load_shard, paths))
    sources = [provenance for _, provenance in loaded]
    rows = [row for candidates, _ in loaded for row in candidates]
    train = deduplicate([row for row in rows if row["split"] == "train"])[: args.n]
    val_pool = deduplicate([row for row in rows if row["split"] == "val"])
    if len(train) != args.n:
        raise ValueError("Insufficient eligible training examples")
    train_clusters = {row["struct_cluster_id"] for row in train}
    train_sequences = {row["sequence_sha256"] for row in train}
    val_pool = [
        row for row in val_pool if row["sequence_sha256"] not in train_sequences
    ]
    if train_clusters & {row["struct_cluster_id"] for row in val_pool}:
        raise ValueError("AFDB structural clusters cross the original split")
    validation = []
    for index, target in enumerate(train):
        same_round = [row for row in val_pool if row["round"] == target["round"]]
        if not same_round:
            raise ValueError("Validation pool cannot match the training round")
        match = min(
            same_round,
            key=lambda row: (
                abs(math.log(row["seq_len"] / target["seq_len"]))
                + abs(row["global_plddt"] - target["global_plddt"]) / 100,
                stable_seed(row["entry_id"], "match"),
            ),
        )
        target["match_id"] = index
        match["match_id"] = index
        validation.append(match)
        val_pool.remove(match)
    records, manifest = [], []
    for dataset, cohort in (("afdb_train", train), ("afdb_val", validation)):
        diagnostic_ids = {
            r["entry_id"]
            for r in sorted(
                cohort, key=lambda r: stable_seed(r["entry_id"], "diagnostic")
            )[: args.diagnostic_n]
        }
        for row in cohort:
            records.append(
                dict(
                    dataset=dataset,
                    stem=row["entry_id"],
                    L=row["seq_len"],
                    input_seq=row["input_seq"],
                    contacts=row["contacts"],
                    resolved=list(range(row["seq_len"])),
                    diagnostic=row["entry_id"] in diagnostic_ids,
                )
            )
            manifest.append(
                {
                    **{
                        k: v
                        for k, v in row.items()
                        if k not in ("input_seq", "contacts")
                    },
                    "dataset": dataset,
                    "stem": row["entry_id"],
                    "diagnostic": row["entry_id"] in diagnostic_ids,
                }
            )
    targets_blob, targets_source = fetch(
        FOLDBENCH + "eval_targets_foldbench_monomers.parquet"
    )
    gt_blob, gt_source = fetch(FOLDBENCH + "gt_universe_scored.jsonl")
    sets_blob, sets_source = fetch(FOLDBENCH + "eval_sets.csv")
    sets = pd.read_csv(pa.BufferReader(sets_blob))
    stems = set(sets.loc[sets.eval_set == "eval-val", "stem"])
    targets = {
        r["stem"]: r for r in pq.read_table(pa.BufferReader(targets_blob)).to_pylist()
    }
    gt = [json.loads(line) for line in gt_blob.decode().splitlines() if line.strip()]
    chosen = [r for r in gt if r["stem"] in stems]
    if len(chosen) != 97:
        raise ValueError(f"Expected exactly 97 eval-val proteins, got {len(chosen)}")
    diagnostics = {
        r["stem"]
        for r in sorted(chosen, key=lambda r: stable_seed(r["stem"], "diagnostic"))[
            : args.diagnostic_n
        ]
    }
    for row in chosen:
        records.append(
            {
                **row,
                "dataset": "eval_val",
                "input_seq": targets[row["stem"]]["input_seq"],
                "diagnostic": row["stem"] in diagnostics,
            }
        )
        manifest.append(
            dict(
                dataset="eval_val",
                stem=row["stem"],
                seq_len=row["L"],
                diagnostic=row["stem"] in diagnostics,
            )
        )
    output = HERE / "data"
    output.mkdir(exist_ok=True)
    (output / "targets.json").write_text(json.dumps(records, separators=(",", ":")))
    pd.DataFrame(manifest).to_csv(output / "cohort_manifest.csv", index=False)
    (output / "input_provenance.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "train_shards": train_shards,
                "n_per_afdb_cohort": args.n,
                "diagnostic_n_per_cohort": args.diagnostic_n,
                "sources": sources + [targets_source, gt_source, sets_source],
                "original_afdb_validation_is_not_a_certified_esm_holdout": True,
                "ground_truth_afdb": "all nontruncated serialized training contacts, degree >= 0.001 and separation >= 6",
                "checkpoint_training_contract": "exp277 config.py uses all exp232 native AFDB training cache and original exp154 AFDB validation cache",
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {"units": len(records), "download_bytes": sum(s["bytes"] for s in sources)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
