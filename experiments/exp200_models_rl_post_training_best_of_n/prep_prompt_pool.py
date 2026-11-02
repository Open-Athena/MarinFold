# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build exp200's RL training pool: targets + resampled prompt realizations.

Runs cloud-side. The exp53 corpus is ~2,000 shards in us-east5 and the
workstation uplink is ~2.5 MB/s, so selection has to happen next to the data.

WHY AFDB ROUND 0, not ESM-Atlas. exp163 scaled its corpus with ESM-Atlas
(ESMFold2 distillation labels), which is fine when ground truth only has to rank
candidate drafts. exp200's reward is per-contact correctness: every wrong label
is a directly mis-signed gradient on a specific token, and the penalty term makes
that worse than merely noisy. Round 0 of the exp53 corpus is the highest-pLDDT
copy of each structure — the cleanest ground truth available — so that is what
this samples.

LEAKAGE. eval554 is entirely PDB-derived (denovo_pdb, foldbench100, cameo_hard,
casp_fm) while this pool is AFDB, so entry ids cannot collide by construction.
Sequences can. This excludes every exact sequence in eval554 and reports the
count; homology-level overlap is NOT addressed here and is a known open question
(see exp41, foldseek train-similarity).

``parse_doc`` and its regexes are vendored from exp98's ``select_targets.py``
rather than imported: iris bundles one directory, so a sibling-experiment import
works locally and fails on the pod.

    python prep_prompt_pool.py --n 10000 -k 16 \\
        --eval-targets gs://marin-us-east5/MarinFold/exp163/eval554/targets.parquet \\
        --out-targets gs://.../exp200/train/targets.parquet \\
        --out-prompts gs://.../exp200/train/prompts
"""

import argparse
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    build_document,
    residues_from_sequence,
)
from marinfold.document_structures.contacts_v1.parse import ONE_LETTER_TO_THREE

TRAIN_DIR = (
    "marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/documents/train"
)
N_SHARDS = 2067
NUM_POS = 2000
MIN_SEP = 6
BEGIN = "<begin_statements>"

CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")
NTERM_RE = re.compile(r"<n-term>\s+<p(\d+)>")
RES_RE = re.compile(r"<p(\d+)>\s+<([A-Z]{3})>")
THREE_TO_ONE = {three: one for one, three in ONE_LETTER_TO_THREE.items()}


def shard_path(i: int) -> str:
    return f"gs://{TRAIN_DIR}/contacts_v1-{i:05d}-of-0{N_SHARDS}.parquet"


def parse_doc(doc: str):
    """``(L, one-letter sequence, sorted GT pairs)`` or None. Vendored from exp98."""
    cut = doc.index(BEGIN) + len(BEGIN)
    prefix, struct = doc[:cut], doc[cut:]
    m = NTERM_RE.search(prefix)
    if not m:
        return None
    nterm = int(m.group(1))
    pos_in_seq = sorted(
        {int(p) for p in re.findall(r"<p(\d+)>", prefix)}, key=lambda p: (p - nterm) % NUM_POS
    )
    seqidx = {p: (p - nterm) % NUM_POS for p in pos_in_seq}
    res_of_pos = {int(p): aa for p, aa in RES_RE.findall(prefix)}
    if not all(p in res_of_pos for p in pos_in_seq):
        return None
    seq = "".join(THREE_TO_ONE.get(res_of_pos[p], "X") for p in pos_in_seq)
    gt = set()
    for a, b in CONTACT_RE.findall(struct):
        ia, ib = seqidx.get(int(a)), seqidx.get(int(b))
        if ia is None or ib is None or ia == ib or abs(ia - ib) < MIN_SEP:
            continue
        gt.add((min(ia, ib), max(ia, ib)))
    return len(pos_in_seq), seq, sorted(gt)


def shard_round(fs, i: int) -> int:
    with fs.open(shard_path(i), "rb") as fh:
        return pq.read_table(fh, columns=["round"]).column("round").to_pylist()[0]


def first_round0_shard(fs) -> int:
    """Binary-search the first round-0 shard; shards are ordered round-descending."""
    lo, hi = 0, N_SHARDS - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if shard_round(fs, mid) == 0:
            hi = mid
        else:
            lo = mid + 1
    if shard_round(fs, lo) != 0:
        raise ValueError("no round-0 shards found")
    return lo


def load_eval_sequences(path: str) -> set[str]:
    with fsspec.open(path, "rb") as fh:
        table = pq.read_table(fh, columns=["sequence"])
    return set(table["sequence"].to_pylist())


def collect(fs, shards, *, want: int, max_len: int, min_contacts: int, exclude: set[str]):
    """Read shards until `want` usable candidates are found."""
    cols = ["document", "seq_len", "entry_id", "global_plddt", "struct_cluster_id", "round", "truncated"]
    pool, n_leaked, n_shards = [], 0, 0
    for index in shards:
        if len(pool) >= want:
            break
        with fs.open(shard_path(index), "rb") as fh:
            table = pq.read_table(fh, columns=cols).to_pylist()
        n_shards += 1
        for row in table:
            if row["round"] != 0 or row["truncated"] or row["seq_len"] > max_len:
                continue
            parsed = parse_doc(row["document"])
            if parsed is None:
                continue
            length, sequence, gt = parsed
            if len(gt) < min_contacts:
                continue
            if sequence in exclude:
                n_leaked += 1
                continue
            pool.append(
                dict(
                    entry_id=row["entry_id"], L=length, sequence=sequence, n_gt=len(gt),
                    gt_contacts=[[i, j] for i, j in gt],
                    global_plddt=row["global_plddt"], struct_cluster_id=row["struct_cluster_id"],
                )
            )
        print(f"[prep] {n_shards} shards -> {len(pool)} candidates ({n_leaked} excluded)", flush=True)
    return pool, n_leaked, n_shards


def write_prompts(targets, out_dir: str, k: int, workers: int) -> int:
    """One parquet of `k` realizations per target.

    Deliberately one object per protein, matching what ``contacts_env`` reads.
    exp163 flagged that this does not scale past ~1M proteins; at 10k it is fine,
    and changing the layout would mean changing the env's reader in lockstep.
    """
    out_dir = out_dir.rstrip("/")

    def one(target) -> bool:
        residues = residues_from_sequence(target["sequence"])
        rows = []
        for r in range(k):
            # A fresh N-terminus and statement order per realization: the format's
            # nuisance symmetries, and exp82's settled rollout+resample recipe.
            # Mirrors gen_prompts_exp163.py exactly, including reading n_term_index
            # off the document rather than regexing the prefix back out of it.
            doc = build_document(f"{target['entry_id']}:r{r}", residues, [], config=GenerationConfig())
            if doc is None:
                raise RuntimeError(f"build_document returned None for {target['entry_id']}")
            prefix = doc.document[: doc.document.index(BEGIN) + len(BEGIN)]
            positions = [(doc.n_term_index + i) % NUM_POS for i in range(target["L"])]
            rows.append({"r": r, "L": target["L"], "prefix": prefix, "seq_positions": positions})
        with fsspec.open(f"{out_dir}/{target['entry_id']}.parquet", "wb") as fh:
            pq.write_table(pa.Table.from_pylist(rows), fh)
        return True

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, t) for t in targets]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 500 == 0:
                print(f"[prep] prompts {done}/{len(targets)}", flush=True)
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("-k", "--realizations", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--min-contacts", type=int, default=5)
    ap.add_argument("--pool-mult", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=200)
    ap.add_argument("--eval-targets", required=True, help="excluded by exact sequence")
    ap.add_argument("--out-targets", required=True)
    ap.add_argument("--out-prompts", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()

    t0 = time.time()
    fs = fsspec.filesystem("gcs")

    exclude = load_eval_sequences(a.eval_targets)
    print(f"[prep] excluding {len(exclude)} eval sequences", flush=True)

    start0 = first_round0_shard(fs)
    shards = list(range(start0, N_SHARDS))
    print(f"[prep] round-0 shards [{start0}, {N_SHARDS - 1}] ({len(shards)})", flush=True)
    random.Random(a.seed).shuffle(shards)

    pool, n_leaked, n_shards = collect(
        fs, shards, want=int(a.pool_mult * a.n), max_len=a.max_len,
        min_contacts=a.min_contacts, exclude=exclude,
    )
    if len(pool) < a.n:
        raise ValueError(f"only {len(pool)} candidates for --n {a.n}; raise --pool-mult")

    # Deduplicate by structural cluster so the pool is not many copies of one fold.
    by_cluster: dict[object, dict] = {}
    for row in pool:
        by_cluster.setdefault(row["struct_cluster_id"], row)
    unique = list(by_cluster.values())
    print(f"[prep] {len(pool)} candidates -> {len(unique)} distinct clusters", flush=True)

    rng = random.Random(a.seed)
    targets = rng.sample(unique, min(a.n, len(unique)))
    targets.sort(key=lambda r: r["entry_id"])

    with fsspec.open(a.out_targets, "wb") as fh:
        pq.write_table(pa.Table.from_pylist(targets), fh)
    print(f"[prep] wrote {len(targets)} targets -> {a.out_targets}", flush=True)

    written = write_prompts(targets, a.out_prompts, a.realizations, a.workers)
    lengths = [t["L"] for t in targets]
    contacts = [t["n_gt"] for t in targets]
    print(
        f"[prep] DONE {written} prompt files x {a.realizations} realizations in "
        f"{time.time() - t0:.0f}s | shards read {n_shards} | leaked-excluded {n_leaked} | "
        f"L mean {sum(lengths)/len(lengths):.0f} | n_gt mean {sum(contacts)/len(contacts):.0f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
