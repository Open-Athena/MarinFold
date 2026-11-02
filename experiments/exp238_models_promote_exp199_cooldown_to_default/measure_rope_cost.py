# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""What the transformers-5 rope defect costs THIS checkpoint — issue #238.

Every checkpoint we republish carries a `PROVENANCE.md` stating, in nats per
token on real contacts-v1 documents, what a transformers-4.x reader loses when
`config.json` states rope only as `rope_parameters`. The number is measured per
checkpoint rather than carried over, because it is not a constant: the #117 and
#166 copies both measured ~0.76 nats on these three documents and #199's
p06-aug measured 0.437. Same defect, different cost — it depends on what the
weights learned to do with position.

The three documents are exp82's benchmark proteins (#82, #89): 1UBQ, 1QYS, and
7BNY chain A. They span the length range over which a wrong rope base goes from
mildly to badly wrong, which is the point — the damage growing with length is
the signature of the defect, and one short protein would understate it.

Reads the PUBLISHED bucket copy through the registry, not a staging directory,
so what is measured is what a user actually downloads. The model is loaded
twice: once with `config.json` as published (repaired), once with the 4.x rope
keys stripped back to the levanter export's shape, which is what a 4.x reader
effectively sees.

Structures come from the local RCSB mmCIF mirror, so the measurement cannot
silently drift with an updated PDB entry.

    cd marinfold && uv run --extra transformers --extra contacts-v1 python \\
        ../experiments/exp238_models_promote_exp199_cooldown_to_default/measure_rope_cost.py
"""

import argparse
import json
import sys
from pathlib import Path

import gemmi
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from marinfold.document_structures.contacts_v1.generate import generate_document
from marinfold.inference._tokenizer import load_tokenizer
from marinfold.registry import resolve_model

HERE = Path(__file__).resolve().parent
MMCIF_DIR = Path("/data/tim/af3-db/mmcif_files")

# exp82's benchmark set: (PDB id, chain or None for a single-chain entry).
# 7BNY is a large multi-chain entry, so it is pinned to chain A the way #82 and
# #89 score it; contacts-v1 generation rejects multi-chain input outright.
DOCUMENTS = (("1UBQ", None), ("1QYS", None), ("7BNY", "A"))


def structure(pdb_id: str, chain: str | None):
    """Load one entry from the local mirror, reduced to ``chain`` if given.

    Uses a gemmi ``Selection`` rather than removing chains in a loop: mutating
    a chain list while iterating it skips every other entry, which produces a
    structure that looks plausible and is missing half its atoms.
    """
    st = gemmi.read_structure(str(MMCIF_DIR / f"{pdb_id.lower()}.cif"))
    st.setup_entities()
    if chain is None:
        return st
    selected = gemmi.Selection(f"/1/{chain}").copy_structure_selection(st)
    selected.setup_entities()
    return selected


def as_published(config: dict) -> dict:
    """The config a transformers-4.x reader effectively sees WITHOUT the repair.

    Not simply "delete rope_theta": the point is to reproduce the levanter
    export — `rope_parameters` present, the 4.x keys absent — so the loader
    takes the same fallback path it takes on an unrepaired checkpoint.
    """
    if "rope_parameters" not in config:
        raise SystemExit("this config has no rope_parameters block; nothing to undo")
    return {k: v for k, v in config.items() if k not in ("rope_theta", "rope_scaling")}


def device() -> str:
    """CUDA when the installed torch can actually talk to the driver.

    The workstation ships a 12.2 driver, and whichever cu13 wheel `uv` resolves
    raises on `.cuda()` rather than reporting itself unavailable up front. Six
    fp32 forward passes over <700 tokens are a minute or two on CPU, so falling
    back is cheaper than pinning a cu121 wheel just for this.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def document_nll(model_dir: Path, config, tokenizer, documents: dict[str, str]) -> dict:
    """Mean NLL per token for each document under one config."""
    where = device()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), config=config, dtype=torch.float32,
    ).eval().to(where)
    out = {}
    with torch.no_grad():
        for name, text in documents.items():
            ids = torch.tensor([tokenizer.encode(text)], device=where)
            out[name] = (model(ids, labels=ids).loss.item(), ids.shape[1])
    del model
    if where == "cuda":
        torch.cuda.empty_cache()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="MODELS.yaml nickname or local dir (default: the default entry)")
    parser.add_argument("--out", type=Path, default=HERE / "data" / "rope_cost.json")
    args = parser.parse_args()

    model_dir = resolve_model(args.model)
    raw = json.loads((model_dir / "config.json").read_text())
    if raw.get("rope_theta") is None:
        raise SystemExit(f"{model_dir}/config.json has no top-level rope_theta — the "
                         f"repair did not happen, and this would measure nothing")

    tokenizer = load_tokenizer(model_dir)
    documents = {}
    for pdb_id, chain in DOCUMENTS:
        result = generate_document(structure(pdb_id, chain), entry_id=pdb_id)
        if result is None:
            raise SystemExit(f"{pdb_id} produced no contacts-v1 document")
        documents[pdb_id if chain is None else f"{pdb_id} ({chain})"] = result.document

    fixed = document_nll(model_dir, AutoConfig.for_model(**raw), tokenizer, documents)
    broken = document_nll(model_dir, AutoConfig.for_model(**as_published(raw)),
                          tokenizer, documents)

    rows = [dict(document=name, tokens=fixed[name][1], repaired=fixed[name][0],
                 as_published=broken[name][0],
                 delta=broken[name][0] - fixed[name][0])
            for name in documents]
    mean_repaired = sum(r["repaired"] for r in rows) / len(rows)
    mean_published = sum(r["as_published"] for r in rows) / len(rows)
    result = dict(model=str(model_dir), rope_theta=raw["rope_theta"], rows=rows,
                  mean_repaired=mean_repaired, mean_as_published=mean_published,
                  mean_delta=mean_published - mean_repaired)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")

    for r in rows:
        print(f"{r['document']:>14}  {r['tokens']:5d} tok  {r['repaired']:.3f} -> "
              f"{r['as_published']:.3f}  {r['delta']:+.2f}")
    print(f"{'mean':>14}         {mean_repaired:.3f} -> {mean_published:.3f}  "
          f"{result['mean_delta']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
