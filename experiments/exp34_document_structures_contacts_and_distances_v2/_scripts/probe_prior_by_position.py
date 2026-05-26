# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""P(mode | k statements already emitted) for the v1 1B model.

Follow-up to ``probe_begin_statements_prior.py``: that probe found
the model is well-calibrated at position 0 (right after
``<begin_statements>``) — long is the argmax in 26/30 proteins, not
medium. So the exp27 "~99% medium-bias" must come from elsewhere.

Two candidates:

A) **Positional drift.** Maybe at position 0 the model is fine, but
   after k contact/distance statements have been emitted the
   conditional prior shifts toward medium. We test by cutting a
   real v1 doc at the boundary right before statement #k and
   reading the next-token softmax, for k in {0, 1, 5, 20, 50}.
B) **Sampling artifact.** exp27 sampled trajectories with a
   logits processor that masked everything except contact tokens.
   Even a mild per-step preference for medium can compound across
   many samples; if (A) shows no drift, the 99% number is a
   sampling-with-masking compounding effect rather than a property
   of the raw softmax.

This script handles (A). Outputs ``data/begin_statements_prior_by_position.csv``
(one row per (entry_id, k_position)) and ``plots/10_prior_by_position.png``.
"""

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent

sys.path.insert(0, str(EXP_ROOT))
# Use the v1 generator + parser — the 1B model was trained on v1
# docs, so the in-distribution prefix is a v1 doc.
sys.path.insert(0, str(EXP_ROOT.parent / "exp1_document_structures_contacts_and_distances_v1"))

import generate as v1_generate
from parse import iter_parsed_structures, parse_structure
from build_summary import save_plot_with_meta


_MODE_TOKENS = [
    "<long-range-contact>",
    "<medium-range-contact>",
    "<short-range-contact>",
    "<distance>",
]
_PLDDT_TOKEN_PREFIX = "<plddt_"

_DEFAULT_PROTEIN_DIR = (
    EXP_ROOT.parent
    / "exp20_evals_marinfold_1b_foldbench"
    / "protenix_data" / "data" / "protenix-foldbench-monomers" / "gt"
)
_DEFAULT_CSV = EXP_ROOT / "data" / "begin_statements_prior_by_position.csv"
_DEFAULT_PNG = EXP_ROOT / "plots" / "10_prior_by_position.png"


# ---------------------------------------------------------------------------
# Statement-boundary helpers
# ---------------------------------------------------------------------------


def _statement_boundary_token_indices(tokens: list[str]) -> list[int]:
    """Indices of the *first* token of each statement in ``tokens``.

    Walks the tokens after ``<begin_statements>`` and groups them
    into statements by their starter token:

    - ``<distance>`` → 6 tokens (mode, p_i, p_j, atom_i, atom_j, d_X.X)
    - ``<{long|medium|short}-range-contact>`` → 3 tokens

    Free-floating ``<plddt_*>`` tokens (placed mid-statements by
    the v1 algorithm) are passed over without counting as a
    statement.

    Returns the start index of statement 0, 1, 2, ... ; doesn't
    include the index of the first non-statement token (``<end>`` /
    trailing pLDDT).
    """
    starts: list[int] = []
    bs = tokens.index("<begin_statements>")
    i = bs + 1
    while i < len(tokens):
        t = tokens[i]
        if t == "<end>":
            break
        if t.startswith(_PLDDT_TOKEN_PREFIX):
            i += 1
            continue
        starts.append(i)
        if t == "<distance>":
            i += 6
        elif t.endswith("-range-contact>"):
            i += 3
        else:
            # Defensive — should never happen on a well-formed v1 doc.
            raise ValueError(f"Unexpected token at position {i}: {t!r}")
    return starts


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def _build_v1_doc(structure, plddt_min: float) -> list[str]:
    """Generate a v1 doc for ``structure`` and return its token list."""
    cfg = v1_generate.GenerationConfig(residue_plddt_min=plddt_min)
    doc = v1_generate._generate_one(
        structure, context_length=8192, cfg=cfg,
    )
    if doc is None:
        return []
    return doc.split()


def stage_position(
    model_dir: Path, protein_dir: Path, *,
    k_positions: tuple[int, ...], num_proteins: int, residue_plddt_min: float,
    out_csv: Path, out_png: Path,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading 1B model from {model_dir} (device={device}) ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16,
    ).to(device).eval()
    mode_ids = {t: tokenizer.encode(t, add_special_tokens=False)[0] for t in _MODE_TOKENS}
    print("token ids:", mode_ids)

    # Disambiguate XXXX block names so each protein has its own seed
    # in the v1 generator (same fix as in sample_stats.py).
    import dataclasses
    _PLACEHOLDER_NAMES = frozenset({"", "XXXX", "UNNAMED"})

    rows: list[dict] = []
    processed = 0
    for structure in iter_parsed_structures(protein_dir):
        if structure.entry_id in _PLACEHOLDER_NAMES:
            structure = dataclasses.replace(
                structure, entry_id=structure.source_path.stem,
            )
        tokens = _build_v1_doc(structure, plddt_min=residue_plddt_min)
        if not tokens:
            continue
        # Statement boundaries are *content positions* in `tokens`.
        # The model's softmax at "position k" corresponds to running
        # the prompt = tokens[:starts[k]] and reading the next-token
        # distribution. If a requested k is past the doc's statement
        # count we skip it.
        starts = _statement_boundary_token_indices(tokens)
        if not starts:
            continue
        n_residues = len(structure.residues)
        for k in k_positions:
            if k >= len(starts):
                continue
            prompt_tokens = tokens[: starts[k]]
            prompt_ids = tokenizer.encode(
                " ".join(prompt_tokens), add_special_tokens=False,
            )
            ids = torch.tensor([prompt_ids], device=device)
            with torch.no_grad():
                logits = model(ids).logits[0, -1].float().cpu()
            probs = torch.softmax(logits, dim=-1)
            top_id = int(torch.argmax(probs))
            top_tok = tokenizer.decode([top_id]).strip()
            row = {
                "entry_id": structure.source_path.stem,
                "n_residues": n_residues,
                "n_statements_in_doc": len(starts),
                "k": k,
                "argmax_token": top_tok,
                "argmax_prob": float(probs[top_id]),
            }
            row.update({f"p_{t}": float(probs[mode_ids[t]]) for t in _MODE_TOKENS})
            rows.append(row)
        processed += 1
        if processed >= num_proteins:
            break
        if processed % 5 == 0:
            print(f"  ... {processed} proteins")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "entry_id", "n_residues", "n_statements_in_doc", "k",
        "argmax_token", "argmax_prob",
    ] + [f"p_{t}" for t in _MODE_TOKENS]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            r = dict(r)
            for k_, v in r.items():
                if isinstance(v, float):
                    r[k_] = f"{v:.6f}"
            w.writerow(r)
    print(f"\nwrote {out_csv}")

    # Aggregate + plot.
    _plot_drift(rows, out_png)


def _plot_drift(rows: list[dict], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np

    # Build per-k aggregates.
    ks = sorted({r["k"] for r in rows})
    per_k = {k: [r for r in rows if r["k"] == k] for k in ks}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 2]})

    colors = {
        "<long-range-contact>":   "#1f77b4",
        "<medium-range-contact>": "#ff7f0e",
        "<short-range-contact>":  "#2ca02c",
        "<distance>":             "#9467bd",
    }
    label_short = {
        "<long-range-contact>":   "long",
        "<medium-range-contact>": "medium",
        "<short-range-contact>":  "short",
        "<distance>":             "distance",
    }

    # Left: mean softmax mass per mode vs k.
    ax = axes[0]
    for tok in _MODE_TOKENS:
        means = []
        q25 = []
        q75 = []
        for k in ks:
            vals = [r[f"p_{tok}"] for r in per_k[k]]
            means.append(np.mean(vals))
            q25.append(np.percentile(vals, 25))
            q75.append(np.percentile(vals, 75))
        ax.plot(ks, means, marker="o", color=colors[tok],
                label=f"{label_short[tok]}", lw=2)
        ax.fill_between(ks, q25, q75, color=colors[tok], alpha=0.15)
    ax.set_xlabel("position k (# statements already in prompt)")
    ax.set_ylabel("P(token | prompt)  (raw softmax)")
    ax.set_title("Mean next-token softmax at the k'th inter-statement boundary")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)

    # Right: same data, but renormalized over only the three contact
    # modes (this is what exp27's logits-processor sees).
    ax = axes[1]
    for tok in [t for t in _MODE_TOKENS if t != "<distance>"]:
        means = []
        for k in ks:
            denom = [
                sum(r[f"p_{t}"] for t in _MODE_TOKENS if t != "<distance>")
                for r in per_k[k]
            ]
            vals = [
                (r[f"p_{tok}"] / d) if d > 0 else 0.0
                for r, d in zip(per_k[k], denom)
            ]
            means.append(np.mean(vals))
        ax.plot(ks, means, marker="o", color=colors[tok], label=label_short[tok], lw=2)
    ax.axhline(1/3, ls=":", color="grey", lw=1, label="uniform")
    ax.set_xlabel("position k")
    ax.set_ylabel("P(mode | prompt, contact only)")
    ax.set_title("Renormalized over the 3 contact modes\n(what the exp27 logits processor sees)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    save_plot_with_meta(
        fig, out_png,
        caption=(
            "Left: raw softmax mass on each candidate at the k'th "
            "inter-statement boundary in a real v1 doc. Right: same "
            "data but renormalized over the 3 contact-mode tokens "
            "only (matching the masking that exp27's logits "
            "processor applies). If a positional medium bias "
            "exists, the orange line on the right panel climbs as "
            "k increases."
        ),
    )
    plt.close(fig)
    print(f"wrote {out_png}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="1B",
                    help="Model nickname (MODELS.yaml) or local path.")
    ap.add_argument("--protein-dir", type=Path, default=_DEFAULT_PROTEIN_DIR)
    ap.add_argument("--num-proteins", type=int, default=20)
    ap.add_argument("--residue-plddt-min", type=float, default=0.0,
                    help="0.0 disables filter on PDB inputs; 70 mirrors AFDB training.")
    ap.add_argument("--k-positions", type=int, nargs="+",
                    default=[0, 1, 5, 20, 50],
                    help="Statement boundaries at which to read the softmax.")
    ap.add_argument("--out-csv", type=Path, default=_DEFAULT_CSV)
    ap.add_argument("--out-png", type=Path, default=_DEFAULT_PNG)
    args = ap.parse_args(argv)

    from marinfold.registry import resolve_model

    model_dir = resolve_model(args.model)
    print(f"resolved {args.model!r} -> {model_dir}")
    stage_position(
        model_dir, args.protein_dir,
        k_positions=tuple(args.k_positions),
        num_proteins=args.num_proteins,
        residue_plddt_min=args.residue_plddt_min,
        out_csv=args.out_csv, out_png=args.out_png,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
