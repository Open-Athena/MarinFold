# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn exp200's contacts-v1 prompt pool into a SkyRL dataset — issue #208.

Reuses the pool built for the marin.rl path
(`gs://marin-us-central1/.../exp200/train/{targets.parquet,prompts/}`): 10,000
AFDB round-0 proteins, 16 resampled realizations each, zero exact-sequence
overlap with the 554-protein eval set. Nothing about that pool is
framework-specific, so the two harnesses train on identical data and their
results stay comparable.

SkyRL rows carry `prompt`, `env_class` and `extras`; this writes the per-protein
state `ContactsV1Env` needs into `extras`.

TWO THINGS THAT WILL BITE, BOTH FROM READING THE GENERATOR SOURCE.

**1. contacts-v1 has no chat format.** The prompt is a raw token sequence, and
the vocabulary has neither a chat template nor `<|im_end|>`. exp200 hit the same
wall on marin.rl — its renderer templated every prompt — and worked around it by
calling `llm.generate` with `TokensPrompt` directly. SkyRL's equivalent is
`custom_chat_template`, which must pass the prompt through verbatim.

**2. Setting `custom_chat_template` silently disables per-token rewards in the
stock generator.** `_build_per_token_rewards` reads:

```python
if self.custom_chat_template:
    reward_out = per_step_rewards[-1][0]        # <- a FLOAT
else:
    token_level_rewards = [0.0] * len(response_ids)
    ...
```

So the two things exp208 needs — a raw prompt and a dense reward — are mutually
exclusive in the stock implementation. `DenseContactsGenerator` overrides the
whole method and returns the dense vector on both paths, which is precisely why
the override is of `_build_per_token_rewards` rather than a narrower hook. If a
future SkyRL bump reorganises that method, this is the first thing to re-check:
the failure is silent and degrades to one scalar per trajectory.

    uv run python build_dataset.py --out data/skyrl_train.parquet --n 2000
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import pyarrow as pq_types  # noqa: F401  (kept for schema clarity)
import pyarrow.parquet as pq

POOL = "gs://marin-us-central1/protein-structure/MarinFold/exp200/train"
ENV_CLASS = "contacts_v1"


def load_targets(path: str, limit: int | None) -> dict[str, dict]:
    with fsspec.open(path, "rb") as fh:
        table = pq.read_table(fh, columns=["entry_id", "L", "gt_contacts"])
    out: dict[str, dict] = {}
    for entry_id, length, pairs in zip(table["entry_id"].to_pylist(),
                                       table["L"].to_pylist(),
                                       table["gt_contacts"].to_pylist()):
        # `gt_contacts` is a list of [i, j] PAIRS in sequence-index space, not a
        # flat [i0, j0, i1, ...] vector. Confusing the two silently halves the
        # ground truth and pairs unrelated residues.
        gt = sorted({(min(int(i), int(j)), max(int(i), int(j))) for i, j in pairs
                     if abs(int(i) - int(j)) >= 6})
        if gt:
            out[entry_id] = {"L": int(length), "gt": gt}
        if limit and len(out) >= limit:
            break
    return out


def prompts_for(prompts_path: str, entry_id: str) -> list[dict]:
    with fsspec.open(f"{prompts_path}/{entry_id}.parquet", "rb") as fh:
        table = pq.read_table(fh, columns=["r", "prefix", "seq_positions"])
    return [{"r": int(r), "prefix": p, "seq_positions": list(sp)}
            for r, p, sp in zip(table["r"].to_pylist(), table["prefix"].to_pylist(),
                                table["seq_positions"].to_pylist())]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2000, help="proteins")
    ap.add_argument("--realizations", type=int, default=1,
                    help="rows per protein; SkyRL's n_samples_per_prompt supplies the "
                         "rollout group, so 1 row per protein is usually right")
    ap.add_argument("--split", default="train")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    targets = load_targets(f"{a.pool}/targets.parquet", a.n)
    ids = sorted(targets)
    print(f"[dataset] {len(ids)} proteins with usable ground truth", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        realizations = list(pool.map(lambda e: prompts_for(f"{a.pool}/prompts", e), ids))

    rows = []
    for entry_id, reals in zip(ids, realizations):
        for row in reals[: a.realizations]:
            rows.append({
                # Raw contacts-v1 text. It is NOT a chat exchange; see the module
                # docstring for the custom_chat_template requirement.
                "prompt": json.dumps([{"role": "user", "content": row["prefix"]}]),
                "env_class": ENV_CLASS,
                "split": a.split,
                "extras": json.dumps({
                    "entry_id": entry_id,
                    "L": targets[entry_id]["L"],
                    "gt_contacts": targets[entry_id]["gt"],
                    "seq_positions": row["seq_positions"],
                    "realization": row["r"],
                }),
            })

    a.out.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    pq.write_table(pa.table({k: [r[k] for r in rows] for k in rows[0]}), a.out)
    print(f"[dataset] wrote {len(rows)} rows -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
