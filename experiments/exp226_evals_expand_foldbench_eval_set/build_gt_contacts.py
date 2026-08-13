# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 6 — ground-truth contacts for the eval2 proteins #89's universe misses.

23 of eval2's 307 proteins are #226's net-new FoldBench monomers, which are not
in #89's frozen ground-truth universe and so cannot be scored. This computes
their GT contacts the same way #89 computed the other 554's, and emits records
in #89's exact `gt_universe.jsonl` schema so the two files concatenate.

Faithfulness is the whole point here, so nothing is re-implemented:

* The contact definition comes from **#89's own `pyconfind_contacts.py`**,
  imported rather than copied — pyconfind in `native_only=True` mode with the
  C++ confind geometry defaults that `contacts_v1.GenerationConfig` generates
  with. Degree and separation thresholds are applied downstream at eval time,
  so every degree>0 contact is kept.
* Structures are the **RCSB biological-assembly mmCIFs** exp12 used for the
  FoldBench-100, at the same `-assembly1` URL.
* The output carries `resolved` — the candidate-pair universe — so the 23 are
  scored over the same kind of index space as the 554.

**The chain gotcha, again.** exp78's manifests pass FoldBench's raw `chain_id`
as `prefer_chain`, which is the mmCIF *label* asym id for 10 entries; that was
harmless for monomer assemblies because `extract_single_chain` then falls back
to the longest polymer chain and there is only one. This passes the resolved
**auth** chain id instead, which is what gemmi names the chain, so the selection
is intentional rather than incidental. Either way `alignment_identity` is the
check that matters, and it is asserted per protein.

    uv run --extra gt python build_gt_contacts.py
"""
import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# #89 owns the contact definition; import it rather than making a fourth copy
# of a module that already exists in exp74, exp78 and exp89. Same seam pattern
# and same reason as exp213_link.py — see that module's docstring.
EXP89_DIR = HERE.parent / "exp89_evals_contacts_v1_model_on_eval_set"
if not EXP89_DIR.is_dir():  # pragma: no cover - branch-layout guard
    raise SystemExit(f"exp89 directory not found at {EXP89_DIR}")
sys.path.insert(0, str(EXP89_DIR))

from pyconfind_contacts import PYCONFIND_KWARGS, compute_contacts  # noqa: E402

RCSB_ASSEMBLY_URL = "https://files.rcsb.org/download/{pdb}-assembly1.cif"

#: A wrong chain or a sequence/structure mismatch shows up here first. The 554
#: are near-1.0; anything materially below that is not a protein we should be
#: quietly adding to an eval set.
MIN_ALIGNMENT_IDENTITY = 0.90

#: A protein with no contacts at this geometry contributes nothing to a contact
#: metric and would silently dilute any average computed over the set.
MIN_CONTACTS = 1


def fetch_assembly(pdb_id: str, cache_dir: Path, *, retries: int = 4) -> Path:
    """Download `<pdb>-assembly1.cif`, cached. Same source exp12 used."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{pdb_id}-assembly1.cif"
    if out.exists() and out.stat().st_size > 0:
        return out
    url = RCSB_ASSEMBLY_URL.format(pdb=pdb_id)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as fh:
                out.write_bytes(fh.read())
            return out
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"failed to fetch {url}") from exc
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def load_targets(eval2: Path, targets: Path) -> list[dict]:
    """eval2 rows that have no ground truth yet, joined to their auth chain."""
    by_stem = {r["stem"]: r for r in csv.DictReader(targets.open())}
    rows = []
    for row in csv.DictReader(eval2.open()):
        if row["has_ground_truth"] == "1":
            continue
        target = by_stem[row["stem"]]
        auth = [c for c in target["auth_asym_ids"].split(";") if c]
        rows.append({
            "dataset": row["dataset"],
            "stem": row["stem"],
            "pdb_id": target["pdb_id"],
            "input_seq": row["input_seq"],
            "prefer_chain": auth[0] if auth else None,
            "best_identity": row["best_identity"],
            "designed_any": row["designed_any"],
        })
    return rows


def build_record(target: dict, cif: Path) -> dict:
    """One `gt_universe.jsonl` record, in #89's schema."""
    gt = compute_contacts(cif, target["input_seq"], stem=target["stem"],
                          prefer_chain=target["prefer_chain"])
    if gt.alignment_identity < MIN_ALIGNMENT_IDENTITY:
        raise ValueError(
            f"{target['stem']}: alignment identity {gt.alignment_identity:.3f} "
            f"below {MIN_ALIGNMENT_IDENTITY} on chain {gt.chain} — the structure "
            "does not match the eval sequence, so the chain selection is wrong."
        )
    if len(gt.contacts) < MIN_CONTACTS:
        raise ValueError(f"{target['stem']}: pyconfind found no contacts")
    return dict(
        dataset=target["dataset"],
        stem=target["stem"],
        L=int(gt.n_input_residues),
        n_resolved=int(gt.n_resolved_residues),
        gt_chain=gt.chain,
        gt_align_identity=round(float(gt.alignment_identity), 4),
        resolved=[int(p) for p in gt.resolved_positions],
        contacts=[[int(i), int(j), float(d)] for (i, j, d) in gt.contacts],
        # #89 carries exp65's strata here; the net-new monomers have none of
        # those axes computed, so the slot is present but empty rather than
        # absent, keeping the record shape identical.
        strata={},
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval2", type=Path, default=DATA / "eval2_manifest.csv")
    ap.add_argument("--targets", type=Path, default=DATA / "foldbench_targets.csv")
    ap.add_argument("--cif-cache", type=Path, default=Path("/data/exp226_gt/cif"),
                    help="assembly mmCIFs are inputs, not artifacts; kept out of git")
    ap.add_argument("--out", type=Path, default=DATA / "gt_universe_eval2_new.jsonl")
    ap.add_argument("--out-manifest", type=Path, default=DATA / "eval2_new_gt_manifest.csv")
    args = ap.parse_args()

    targets = load_targets(args.eval2, args.targets)
    print(f"[gt] {len(targets)} eval2 proteins need ground truth", flush=True)
    print(f"[gt] pyconfind geometry: {PYCONFIND_KWARGS}", flush=True)

    records, manifest = [], []
    for i, target in enumerate(targets, 1):
        cif = fetch_assembly(target["pdb_id"], args.cif_cache)
        record = build_record(target, cif)
        records.append(record)
        manifest.append({
            "dataset": target["dataset"], "stem": target["stem"],
            "gt_cif": cif.name, "gt_chain": record["gt_chain"],
            "foldbench_chain": target["prefer_chain"],
            "input_seq": target["input_seq"], "n_residues": record["L"],
            "n_resolved": record["n_resolved"],
            "gt_align_identity": record["gt_align_identity"],
            "n_contacts": len(record["contacts"]),
            "best_identity": target["best_identity"],
            "designed_any": target["designed_any"],
        })
        print(f"  [{i:2d}/{len(targets)}] {target['stem']}: L={record['L']} "
              f"resolved={record['n_resolved']} chain={record['gt_chain']} "
              f"id={record['gt_align_identity']:.3f} "
              f"contacts={len(record['contacts'])}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    with args.out_manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    coverage = [r["n_resolved"] / r["L"] for r in records]
    print(f"[gt] wrote {len(records)} records -> {args.out}", flush=True)
    print(f"[gt] alignment identity min "
          f"{min(r['gt_align_identity'] for r in records):.3f}; "
          f"resolved/L min {min(coverage):.2f}, median "
          f"{sorted(coverage)[len(coverage) // 2]:.2f}", flush=True)
    print(f"[gt] manifest -> {args.out_manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
