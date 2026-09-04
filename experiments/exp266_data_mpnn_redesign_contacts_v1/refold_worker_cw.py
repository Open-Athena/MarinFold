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

HF_MODEL_ID = "biohub/ESMFold2"
PASS_RMSD = 2.0        # the field's designability gate
PASS_TM = 0.5          # "same fold"


def _log(message: str) -> None:
    print(f"[exp266-refold] {message}", file=sys.stderr, flush=True)


def load_model():
    """Load ESMFold2 once per task."""
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    return model.cuda().eval() if torch.cuda.is_available() else model.eval()


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
    args = ap.parse_args()

    from selfconsistency import ca_coords_from_staged, ca_coords_from_structure, \
        kabsch_rmsd, tm_score

    bb_fs, _ = fsspec.core.url_to_fs(args.backbones_glob)
    backbones: dict[str, dict] = {}
    for path in sorted(bb_fs.glob(args.backbones_glob)):
        with bb_fs.open(path, "rb") as h:
            t = pq.read_table(h, columns=["entry_id", "sequence", "coords_milli"])
        backbones.update({r["entry_id"]: r for r in t.to_pylist()})
    _log(f"{len(backbones)} staged backbones available")

    # Sample backbones, not designs: keeping all 8 designs of a chosen backbone
    # is what makes per-backbone designability computable alongside the
    # per-sequence rate.
    rng = np.random.default_rng(args.seed)
    chosen = set(rng.choice(sorted(backbones), size=min(args.backbones, len(backbones)),
                            replace=False).tolist())

    doc_fs, _ = fsspec.core.url_to_fs(args.documents_glob)
    designs = []
    for path in sorted(doc_fs.glob(args.documents_glob)):
        with doc_fs.open(path, "rb") as h:
            t = pq.read_table(h, columns=["entry_id", "design_index", "seq_len",
                                          "n_term_index", "document",
                                          "mpnn_temperature", "mpnn_score",
                                          "identity_to_native"])
        designs.extend(r for r in t.to_pylist() if r["entry_id"] in chosen)
    _log(f"{len(designs)} designs over {len({d['entry_id'] for d in designs})} backbones")

    from marinfold.document_structures.contacts_v1.read import sequence_from_document

    model = load_model()
    _log(f"ESMFold2 loaded; {args.num_sampling_steps} steps, 1 sample per sequence")

    rows, started = [], time.perf_counter()
    for i, d in enumerate(designs, 1):
        seq = sequence_from_document(d["document"], d["seq_len"], d["n_term_index"])
        t0 = time.perf_counter()
        structure = _predict(model, seq, args)
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


def _predict(model, sequence: str, args):
    """One ESMFold2 draw for one sequence, as a gemmi structure."""
    import gemmi
    import torch

    with torch.no_grad():
        out = model.infer(
            [sequence],
            num_sampling_steps=args.num_sampling_steps,
            num_loops=args.num_loops,
        )
    cif = out[0] if isinstance(out, (list, tuple)) else out
    text = cif if isinstance(cif, str) else cif.to_cif()
    return gemmi.read_structure_string(text, format=gemmi.CoorFormat.Mmcif)


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
