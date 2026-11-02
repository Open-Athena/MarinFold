# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The exp237 training prompts: #208's pool with the mode marker swapped — issue #237.

**Why reuse #208's pool rather than draw a new one.** #237 exists to test one
change — whether a reward computed on the object the metric scores behaves
differently from one computed on a rollout. Holding the data fixed is what makes
that comparison a comparison: same 10,000 AFDB proteins, same resampled
realizations, same harness, same eval. A fresh pool would confound the reward's
unit with the training distribution, and #208's negative result would stop being
the control.

The transformation is exactly one token. A contacts-v1 prompt is

    <contacts-v1> <begin_sequence> <pN> <AA> ... <begin_statements>

and the multi prompt is the same string with token 0 replaced by
``<contacts-v1.multi>``. #230 renamed vocab id 7 **in place**, so this is a
substitution of one id for another in a fixed-size vocabulary — no resize, no
drift — and it is the same swap #230's own eval workers make (`eval_agg_worker`,
`eval_modes_worker`). Everything else in the row, including ``seq_positions`` and
the ground truth, is untouched and must be: the ``<pN>`` rotation in the prompt is
what ``pos_to_seq`` inverts, so rebuilding the prompt would silently decouple
them.

The swap is asserted, not assumed: a row whose prompt does not start with the
plain marker is a hard failure, because the alternative is a run that trains the
plain mode while claiming to train multi.

    python build_multi_dataset.py --src ~/exp208_skyrl/data/skyrl_train_10k.parquet \\
        --out data/skyrl_multi_10k.parquet
"""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PLAIN_TOKEN = "<contacts-v1>"
MULTI_TOKEN = "<contacts-v1.multi>"

SCHEMA = pa.schema([
    ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    ("env_class", pa.string()),
    ("split", pa.string()),
    ("extras", pa.string()),
])


def to_multi(content: str) -> str:
    """Replace the leading plain marker with the multi marker.

    ``startswith`` rather than ``replace``: the plain marker is a substring of the
    multi marker, so a global replace on an already-converted prompt would produce
    ``<contacts-v1.multi.multi>``, which tokenizes to something else entirely and
    would only show up as inexplicably bad rollouts.
    """
    if content.startswith(MULTI_TOKEN):
        return content
    if not content.startswith(PLAIN_TOKEN):
        raise ValueError(f"prompt does not start with {PLAIN_TOKEN!r}: {content[:64]!r}")
    return MULTI_TOKEN + content[len(PLAIN_TOKEN):]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="#208's SkyRL prompt parquet")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=None, help="cap on rows, for a smoke set")
    ap.add_argument("--max-len", type=int, default=None,
                    help="drop proteins longer than this. A multi rollout needs context for "
                         "~22 sections, and a long protein spends it on the sequence header "
                         "instead; leaving it unset keeps #208's pool exactly")
    a = ap.parse_args()

    table = pq.read_table(a.src)
    rows = table.to_pylist()
    out_rows = []
    for row in rows:
        extras = json.loads(row["extras"])
        if a.max_len and int(extras["L"]) > a.max_len:
            continue
        prompt = [{"role": m["role"], "content": to_multi(m["content"])} for m in row["prompt"]]
        out_rows.append({"prompt": prompt, "env_class": row["env_class"],
                         "split": row["split"], "extras": row["extras"]})
        if a.n and len(out_rows) >= a.n:
            break

    if not out_rows:
        raise SystemExit("no rows survived the filters")
    head = out_rows[0]["prompt"][0]["content"]
    if not head.startswith(MULTI_TOKEN) or head.startswith(MULTI_TOKEN + PLAIN_TOKEN):
        raise SystemExit(f"the swap did not take: {head[:80]!r}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({k: [r[k] for r in out_rows] for k in out_rows[0]}, schema=SCHEMA), a.out)
    lengths = [int(json.loads(r["extras"])["L"]) for r in out_rows]
    print(f"[dataset] {len(out_rows)} rows -> {a.out}")
    print(f"[dataset] L: min {min(lengths)} median {sorted(lengths)[len(lengths) // 2]} "
          f"max {max(lengths)}")
    print(f"[dataset] prompt[0]: {head[:90]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
