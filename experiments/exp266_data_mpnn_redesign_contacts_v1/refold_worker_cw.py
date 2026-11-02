# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Refold a sample of designs with ESMFold2 and score self-consistency.

The one assumption exp266 otherwise leaves untested: a ProteinMPNN sequence is
written onto an AFDB backbone and the contacts computed there, but nothing
checks the sequence would *fold* to that backbone. This measures it.

**ESMFold2** (`biohub/ESMFold2`, diffusion over ESMC) at `n_samples=1` — one
draw per sequence, not exp78's top-1-of-5. Best-of-5 answers "can this sequence
fold here if we look hard", and the question here is the plainer "does it",
which is also 5x cheaper. exp78 measured 42.9 s/protein at L 250-300 with 5
samples, so ~8.6 s at 1.

Reports both numbers the field uses, under the identity residue
correspondence (a design has its parent's length and order, so no alignment
search is needed — see `selfconsistency`):

* **per-sequence** self-consistency — the fraction of individual designs with
  scRMSD < 2 A. This is the number that bears on corpus label quality, since
  every design becomes its own document.
* **per-backbone designability** — the fraction of backbones with at least one
  passing design out of 8. This is what papers report, and it is the more
  flattering number; both are printed so they cannot be confused.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HF_MODEL_ID = "biohub/ESMFold2"   # == esm.models.esmfold2.ESMFOLD2_HF_REPO
PASS_RMSD = 2.0        # the field's designability gate
PASS_TM = 0.5          # "same fold"


def _log(message: str) -> None:
    print(f"[exp266-refold] {message}", file=sys.stderr, flush=True)


def load_model(device: str = "cuda"):
    """Load ESMFold2 once per task.

    The class lives in **`esm.models.esmfold2`**, spelled `EsmFold2Model`, and
    `from_pretrained` takes the device directly. exp78's
    `from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model`
    is stale against current `biohub/esm` — importing `esm` does not register
    anything into `transformers.models`, so that path raises ModuleNotFoundError
    no matter what order things are imported in.
    """
    import torch
    from esm.models.esmfold2 import ESMFOLD2_HF_REPO, EsmFold2Model

    dev = device if torch.cuda.is_available() else "cpu"
    model = EsmFold2Model.from_pretrained(ESMFOLD2_HF_REPO, device=dev)
    return model.eval()


def score_confidence(result) -> float:
    """Best-effort scalar confidence (higher = better); NaN if unavailable.

    Same accessor ladder as exp78's `_score_confidence`. Recorded for context
    only — the pass/fail here is scRMSD against the parent backbone, not the
    model's own opinion of its output.
    """
    import math

    for attr in ("ptm", "mean_plddt", "plddt", "confidence"):
        val = getattr(result, attr, None)
        if val is None and hasattr(result, "complex"):
            val = getattr(result.complex, attr, None)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            try:
                return float(np.asarray(val, dtype=float).mean())
            except Exception:  # noqa: BLE001
                continue
    return math.nan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--documents-glob", required=True,
                    help="Redesigned documents parquet glob (s3:// via fsspec).")
    ap.add_argument("--backbones-glob", required=True,
                    help="Staged backbones, for the reference CA coordinates.")
    ap.add_argument("--out", required=True, help="Output parquet.")
    ap.add_argument("--backbones", type=int, default=500,
                    help="Backbones to sample; every design of each is refolded, "
                         "so per-backbone designability is measurable too.")
    ap.add_argument("--num-sampling-steps", type=int, default=100)
    ap.add_argument("--num-loops", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--native-control", action="store_true",
                    help="Refold each sampled backbone's NATIVE sequence instead "
                         "of its designs. The control that makes a design pass "
                         "rate interpretable: if native sequences on these same "
                         "AFDB backbones also fail, the number is about ESMFold2 "
                         "and the metric, not about ProteinMPNN.")
    ap.add_argument("--files", type=int, default=2,
                    help="Document/backbone shard PAIRS this task reads. Reading "
                         "every shard to sample a few hundred backbones would "
                         "pull the whole ~58 GB staged corpus per task; each "
                         "task instead takes a seed-chosen slice of the shards "
                         "and samples inside it.")
    args = ap.parse_args()

    from selfconsistency import ca_coords_from_staged, ca_coords_from_structure, \
        kabsch_rmsd, tm_score

    rng = np.random.default_rng(args.seed)

    # Pick a slice of the DOCUMENT shards first. Stage A sorted the corpus by
    # length, so a random slice of shards is a random slice of length bands —
    # shuffle rather than take a contiguous block, or a task gets one length
    # regime and the self-consistency rate is confounded with length.
    doc_fs, _ = fsspec.core.url_to_fs(args.documents_glob)
    doc_files = sorted(doc_fs.glob(args.documents_glob))
    if not doc_files:
        raise FileNotFoundError(f"no documents match {args.documents_glob}")
    picked = rng.permutation(len(doc_files))[: max(1, args.files)]
    _log(f"reading {len(picked)} of {len(doc_files)} document shards")

    designs = []
    for i in picked:
        with doc_fs.open(doc_files[i], "rb") as h:
            t = pq.read_table(h, columns=["entry_id", "design_index", "seq_len",
                                          "n_term_index", "document",
                                          "mpnn_temperature", "mpnn_score",
                                          "identity_to_native"])
        designs.extend(t.to_pylist())

    # Sample by BACKBONE, keeping all 8 of its designs: that is what makes
    # per-backbone designability computable alongside the per-sequence rate.
    entry_ids = sorted({d["entry_id"] for d in designs})
    chosen = set(rng.choice(entry_ids, size=min(args.backbones, len(entry_ids)),
                            replace=False).tolist())
    designs = [d for d in designs if d["entry_id"] in chosen]

    if args.native_control:
        # One row per backbone, carrying the native sequence. Paired on exactly
        # the backbones the design arm uses, so the two rates are comparable.
        seen: set[str] = set()
        control = []
        for d in designs:
            if d["entry_id"] in seen:
                continue
            seen.add(d["entry_id"])
            control.append({**d, "design_index": -1, "mpnn_temperature": 0.0,
                            "mpnn_score": 0.0, "identity_to_native": 1.0,
                            "document": None})
        designs = control
        _log(f"NATIVE CONTROL: {len(designs)} native sequences")

    # Only now fetch the backbones we actually need, matching shard stems.
    bb_fs, _ = fsspec.core.url_to_fs(args.backbones_glob)
    bb_files = sorted(bb_fs.glob(args.backbones_glob))
    backbones: dict[str, dict] = {}
    for path in bb_files:
        stem = path.rsplit("/", 1)[-1].removesuffix(".parquet")
        if not any(stem in doc_files[i] for i in picked):
            continue
        with bb_fs.open(path, "rb") as h:
            t = pq.read_table(h, columns=["entry_id", "sequence", "coords_milli"])
        backbones.update({r["entry_id"]: r for r in t.to_pylist()
                          if r["entry_id"] in chosen})
    missing = chosen - set(backbones)
    if missing:
        raise ValueError(
            f"{len(missing)} sampled backbones have no staged row "
            f"(e.g. {sorted(missing)[:3]}); document and backbone shards "
            "are not paired as expected"
        )
    _log(f"{len(designs)} designs over {len(chosen)} backbones")

    from marinfold.document_structures.contacts_v1.read import sequence_from_document

    model = load_model()
    _log(f"ESMFold2 loaded; {args.num_sampling_steps} sampling steps, "
         f"{args.num_loops} loops, 1 sample per sequence")

    rows, started = [], time.perf_counter()
    for i, d in enumerate(designs, 1):
        seq = (backbones[d["entry_id"]]["sequence"] if args.native_control
               else sequence_from_document(d["document"], d["seq_len"],
                                           d["n_term_index"]))
        t0 = time.perf_counter()
        structure, confidence = _predict(model, seq, args, seed=args.seed)
        elapsed = time.perf_counter() - t0

        pred = ca_coords_from_structure(structure)
        ref = ca_coords_from_staged(backbones[d["entry_id"]])
        if len(pred) != len(ref):
            # Fail loud: a length mismatch means the sequence we refolded is not
            # the sequence the document asserts, which would silently corrupt
            # every number below it.
            raise ValueError(
                f"{d['entry_id']}#{d['design_index']}: refold has {len(pred)} "
                f"residues, backbone has {len(ref)}"
            )
        rmsd, _ = kabsch_rmsd(pred, ref)
        rows.append({
            "entry_id": d["entry_id"], "design_index": d["design_index"],
            "seq_len": d["seq_len"], "mpnn_temperature": d["mpnn_temperature"],
            "mpnn_score": d["mpnn_score"], "identity_to_native": d["identity_to_native"],
            "sc_rmsd": rmsd, "sc_tm": tm_score(pred, ref),
            "esmfold2_confidence": confidence,
            "esmfold2_seconds": elapsed,
        })
        if i % 50 == 0 or i == len(designs):
            rate = i / (time.perf_counter() - started)
            _log(f"{i}/{len(designs)} refolded ({rate:.2f}/s, "
                 f"{sum(r['sc_rmsd'] < PASS_RMSD for r in rows) / len(rows):.1%} pass)")

    _summarise(rows)
    with fsspec.open(args.out, "wb") as h:
        pq.write_table(pa.Table.from_pylist(rows), h, compression="zstd")
    _log(f"wrote {len(rows)} rows to {args.out}")
    return 0


def _predict(model, sequence: str, args, seed: int):
    """One ESMFold2 draw for one sequence -> (gemmi structure, confidence)."""
    import gemmi
    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        ProteinInput,
        StructurePredictionInput,
    )

    builder = ESMFold2InputBuilder()
    spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=sequence)])
    result = builder.fold(
        model, spi,
        num_loops=args.num_loops,
        num_sampling_steps=args.num_sampling_steps,
        num_diffusion_samples=1,
        seed=seed,
    )
    if isinstance(result, list):
        result = result[0]
    structure = gemmi.read_structure_string(
        result.complex.to_mmcif(), format=gemmi.CoorFormat.Mmcif
    )
    return structure, score_confidence(result)


def _summarise(rows: list[dict]) -> None:
    if not rows:
        return
    per_seq = sum(r["sc_rmsd"] < PASS_RMSD for r in rows) / len(rows)
    per_tm = sum(r["sc_tm"] > PASS_TM for r in rows) / len(rows)
    by_bb: dict[str, list[dict]] = {}
    for r in rows:
        by_bb.setdefault(r["entry_id"], []).append(r)
    per_bb = sum(any(x["sc_rmsd"] < PASS_RMSD for x in v) for v in by_bb.values()) / len(by_bb)
    _log(f"per-SEQUENCE self-consistency (scRMSD<2A): {per_seq:.1%}  "
         f"(scTM>0.5: {per_tm:.1%})")
    _log(f"per-BACKBONE designability (any of 8):     {per_bb:.1%} over {len(by_bb)}")
    print(json.dumps({"per_sequence_rmsd": per_seq, "per_sequence_tm": per_tm,
                      "per_backbone": per_bb, "n_designs": len(rows),
                      "n_backbones": len(by_bb)}))


if __name__ == "__main__":
    raise SystemExit(main())
