# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Draw the protein pool as TWO DISJOINT halves -- one per document kind.

Supersedes the single-pool draw. Two design changes, both deliberate:

**Disjoint proteins.**  A protein appears in the multi half or the plain half,
never both.  The plain half needs no rollouts -- only ground truth -- so tying it
to the rollout-covered subset was a constraint that bought nothing and cost
coverage.  Disjoint halves let the corpus span twice the unique proteins, and
each protein's ground truth is then seen once per epoch instead of twice.

**One document per protein.**  Repetition, where it is needed at all, belongs in
the epoch loop rather than baked into the corpus as near-duplicate documents.

What must NOT change is that the token-0 marker stays the only thing predicting
mode.  Protein *identity* differing between halves is fine; the protein
*distribution* differing is not, because then the sequence section alone would
carry mode information.  So the split is **stratified by arm**: each arm is
divided in the same proportion, and both halves inherit the same length filter,
the same quality gates and the same 30 %-identity decontamination.

Within an arm, proteins that already have rollouts are preferred for the **multi**
side, so generation already paid for is not stranded on the plain side.

    uv run python split_targets.py --work /data/exp230_multi \\
        --n-per-half 700000 --rollouts gs://.../rollouts
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from corpus_sources import AFDB, ARMS, ESM_ATLAS, PDB_MONOMERS, eligible_shards, local_path
from select_targets import SCHEMA, load_droplist


def covered_target_ids(uri: str | None, log) -> set[str]:
    """target_ids that already have generated rollouts."""
    if not uri:
        return set()
    import fsspec

    fs = fsspec.core.url_to_fs(uri)[0]
    scheme = uri.split("://", 1)[0] if "://" in uri else None
    paths = sorted(fs.glob(f"{uri.rstrip('/')}/*.parquet"))
    done: set[str] = set()
    for p in paths:
        full = p if (scheme is None or "://" in p) else f"{scheme}://{p}"
        with fs.open(full, "rb") as fh:
            done |= set(pq.read_table(fh, columns=["target_id"]).column("target_id").to_pylist())
    log(f"[rollouts] {len(done):,} proteins already have rollouts ({len(paths)} parts)")
    return done


def collect_arm(spec, *, work: Path, drop: set, want: int, log) -> list[dict]:
    """Stream an arm's staged shards until ``want`` decontaminated proteins."""
    from corpus_sources import iter_corpus_rows

    staged = [p for p in eligible_shards(spec, work) if local_path(work, p).exists()]
    if not staged:
        raise SystemExit(f"no staged shards for {spec.arm!r} -- run stage.py --arm {spec.arm}")
    kept: list[dict] = []
    seen: set[str] = set()
    n_seen = n_drop = 0
    for path in staged:
        if len(kept) >= want:
            break
        for rec in iter_corpus_rows(spec, work=work, log=lambda *a: None, shards=[path]):
            n_seen += 1
            if (rec["arm"], rec["entry_id"]) in drop:
                n_drop += 1
                continue
            if rec["entry_id"] in seen:
                continue
            seen.add(rec["entry_id"])
            rec["target_id"] = f"{rec['arm']}:{rec['entry_id']}"
            kept.append(rec)
            if len(kept) >= want:
                break
    log(f"[{spec.arm}] {len(staged)} shards -> {n_seen:,} usable, {n_drop:,} contaminated, "
        f"{len(kept):,} kept")
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("/data/exp230_multi"))
    ap.add_argument("--n-per-half", type=int, default=700_000)
    ap.add_argument("--rollouts", default=None,
                    help="existing rollout prefix; its proteins are preferred for the multi half")
    ap.add_argument("--seed", type=int, default=230)
    # Arm quotas for the WHOLE pool (both halves). AFDB round-0 and the PDB
    # monomer corpus are finite, so these are ceilings, not targets; ESM-Atlas
    # absorbs whatever is left over.
    ap.add_argument("--max-afdb", type=int, default=700_000)
    ap.add_argument("--max-pdb", type=int, default=100_000)
    a = ap.parse_args()

    def log(*m):
        print(" ".join(str(x) for x in m), flush=True)

    drop = load_droplist(a.work)
    covered = covered_target_ids(a.rollouts, log)
    total = 2 * a.n_per_half

    # Quotas are SHARES of the total, not fixed ceilings. Taking AFDB up to a
    # fixed maximum first consumed the entire budget at small totals and left
    # ESM-Atlas with ZERO -- which would have dropped 94 % of what exp199 was
    # pretrained on out of the rehearsal half whose whole job is to preserve
    # that pretraining.
    #
    # PDB is finite (~27k survive decontamination) so it is taken first in full;
    # the two predicted corpora then split what remains evenly, which keeps the
    # halves looking like exp199's own pretraining mixture.
    pools: dict[str, list[dict]] = {}
    pools["pdb"] = collect_arm(PDB_MONOMERS, work=a.work, drop=drop,
                               want=min(a.max_pdb, total), log=log)
    remaining = max(0, total - len(pools["pdb"]))
    pools["afdb"] = collect_arm(AFDB, work=a.work, drop=drop,
                                want=min(a.max_afdb, remaining // 2), log=log)
    # ESM-Atlas absorbs whatever AFDB could not supply; AFDB round-0 is finite.
    rest = max(0, total - len(pools["pdb"]) - len(pools["afdb"]))
    pools["esm_atlas"] = collect_arm(ESM_ATLAS, work=a.work, drop=drop, want=rest, log=log)

    got = sum(len(v) for v in pools.values())
    if got < total:
        log(f"[warn] pool is {got:,}, short of {total:,} -- halves will be {got // 2:,} each")
    rng = random.Random(a.seed)

    multi: list[dict] = []
    plain: list[dict] = []
    for arm, recs in pools.items():
        rng.shuffle(recs)
        # Stratified: this arm contributes the SAME share to both halves, so no
        # arm statistic predicts the mode. Within the arm, rollout-covered
        # proteins go to multi first so existing generation is reused.
        half = len(recs) // 2
        recs.sort(key=lambda r: r["target_id"] not in covered)   # covered first
        multi.extend(recs[:half])
        plain.extend(recs[half: 2 * half])
        log(f"[{arm}] split {half:,} multi / {half:,} plain "
            f"({sum(1 for r in recs[:half] if r['target_id'] in covered):,} of the multi "
            f"side already have rollouts)")

    rng.shuffle(multi)
    rng.shuffle(plain)
    assert not ({r["target_id"] for r in multi} & {r["target_id"] for r in plain}), "halves overlap"

    for name, rows in (("targets_multi", multi), ("targets_plain", plain)):
        df = pd.DataFrame(rows)[[f.name for f in SCHEMA]]
        out = a.work / f"{name}.parquet"
        pq.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False), out)
        by_arm = df.groupby("arm").size().to_dict()
        log(f"[{name}] {len(df):,} proteins -> {out}   {by_arm}")

    need = sum(1 for r in multi if r["target_id"] not in covered)
    log(f"\n[generation] {need:,} of {len(multi):,} multi proteins still need rollouts")
    (Path(__file__).parent / "data" / "split.provenance.json").write_text(json.dumps({
        "n_per_half": a.n_per_half, "seed": a.seed,
        "multi": len(multi), "plain": len(plain),
        "already_covered": len([r for r in multi if r["target_id"] in covered]),
        "need_generation": need,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
