# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-v1 PDF plot writers for ``infer`` / ``evaluate``.

Two top-level entry points — :func:`plot_infer_pdf` and
:func:`plot_evaluate_pdf` — open a multi-page PDF and write one page per
structure from the records / :class:`marinfold.EvalResult` the CLI already
produced.

- ``infer``: a single ``P(contact)`` heatmap per structure (the model's
  probability of emitting each residue pair as its next contact statement).
- ``evaluate``: ground-truth pyconfind contacts (left) next to the model's
  ``P(contact)`` (right), per structure — the exp89 benchmark-heatmap view.

The near-diagonal band (|i - j| < ``min_seq_separation``) is never a contact
in contacts-v1, so it renders as NaN (blank) in every panel.

``matplotlib`` is a base ``marinfold`` dependency; the import is kept at call
time anyway so the rest of the impl loads even in a stripped environment.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from marinfold import EvalResult


# --------------------------------------------------------------------------
# Matrix reconstruction
# --------------------------------------------------------------------------


def _reconstruct_matrix(
    pairs: Iterable[tuple[int, int]],
    values: Iterable[float],
    n_residues: int,
) -> np.ndarray:
    """N×N symmetric matrix filled at the 1-based (i, j) candidate pairs.

    Pairs are 1-indexed (i, j) with i < j, matching the record convention.
    Entries not in ``pairs`` (including the masked near-diagonal band and the
    diagonal) stay NaN, which renders transparent under matplotlib.
    """
    out = np.full((n_residues, n_residues), np.nan, dtype=np.float64)
    for (i, j), value in zip(pairs, values, strict=True):
        out[i - 1, j - 1] = value
        out[j - 1, i - 1] = value
    return out


# --------------------------------------------------------------------------
# Figure builders
# --------------------------------------------------------------------------


def figure_predicted_contacts(record: dict[str, Any]):
    """Single-panel ``P(contact)`` heatmap for one ``predict`` record."""
    import matplotlib.pyplot as plt

    entry_id = record["entry_id"]
    n = int(record["n_residues"])
    method = str(record.get("method", "pairwise"))
    label = _score_label(method)
    pred = _reconstruct_matrix(
        [tuple(p) for p in record["pairs"]], record["score"], n
    )

    vmax = _robust_vmax(pred)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(pred, cmap="magma", origin="lower", vmin=0, vmax=vmax,
                   interpolation="none")
    ax.set_title(f"MarinFold-cv1 · {label}\n({method} readout, from sequence)")
    ax.set_xlabel("residue j")
    ax.set_ylabel("residue i")
    fig.colorbar(im, ax=ax, fraction=0.046, label=label)
    fig.suptitle(f"{entry_id}  (L={n}{_method_detail(record)})")
    fig.tight_layout()
    return fig


def figure_gt_vs_predicted(
    *, entry_id: str, n_residues: int, gt: np.ndarray, pred: np.ndarray,
    score_label: str = "P(contact)",
):
    """Two-panel figure: GT contacts (left), the model's contact score (right)."""
    import matplotlib.pyplot as plt

    n_contacts = int(np.nansum(gt) // 2)
    vmax = _robust_vmax(pred)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.9))
    axes[0].imshow(gt, cmap="Greys", origin="lower", interpolation="none")
    axes[0].set_title(
        f"ground-truth contacts\n(L={n_residues}, {n_contacts} contacts, pyconfind)"
    )
    im = axes[1].imshow(pred, cmap="magma", origin="lower", vmin=0, vmax=vmax,
                        interpolation="none")
    axes[1].set_title(f"MarinFold-cv1 · {score_label}\n(from sequence)")
    for ax in axes:
        ax.set_xlabel("residue j")
        ax.set_ylabel("residue i")
    fig.colorbar(im, ax=axes[1], fraction=0.046, label=score_label)
    fig.suptitle(entry_id)
    fig.tight_layout()
    return fig


def _score_label(method: str) -> str:
    """Colour-bar / title label for the per-pair score of each method."""
    return "rollout vote score" if method == "rollout" else "P(contact)"


def _method_detail(record: dict[str, Any]) -> str:
    """Short title suffix describing the readout's ensemble / rollout count."""
    if record.get("method") == "rollout":
        return f", {int(record.get('n_rollouts', 0))} rollouts"
    k = int(record.get("ensemble_k", 1))
    return f", ×{k} ens" if k > 1 else ""


def _robust_vmax(matrix: np.ndarray) -> float:
    """99.5th-percentile colour-scale ceiling (falls back to max / 1.0)."""
    if not np.isfinite(matrix).any():
        return 1.0
    vmax = float(np.nanpercentile(matrix, 99.5))
    if vmax > 0:
        return vmax
    vmax = float(np.nanmax(matrix))
    return vmax if vmax > 0 else 1.0


# --------------------------------------------------------------------------
# PDF assemblers
# --------------------------------------------------------------------------


def plot_infer_pdf(out_path: Path, records: list[dict[str, Any]]) -> None:
    """Write one ``P(contact)`` heatmap page per structure."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise SystemExit("plot_infer_pdf: no records to plot")

    with PdfPages(out_path) as pdf:
        for record in records:
            fig = figure_predicted_contacts(record)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def plot_evaluate_pdf(out_path: Path, result: EvalResult) -> None:
    """Write one GT-vs-model contact-map page per structure.

    Reconstructs both panels from ``result.per_example`` (every scored
    candidate pair's ``score`` and ``gt``). N per structure comes from
    ``result.extras["per_structure_n_residues"]``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_by_entry: dict[str, int] = result.extras.get("per_structure_n_residues", {})
    score_label = _score_label(result.extras.get("method", "pairwise"))

    by_entry: dict[str, list[dict]] = {}
    entry_order: list[str] = []
    for row in result.per_example:
        entry_id = row["entry_id"]
        if entry_id not in by_entry:
            by_entry[entry_id] = []
            entry_order.append(entry_id)
        by_entry[entry_id].append(row)
    if not by_entry:
        raise SystemExit("plot_evaluate_pdf: EvalResult has no per_example rows to plot")

    with PdfPages(out_path) as pdf:
        for entry_id in entry_order:
            rows = by_entry[entry_id]
            n = int(n_by_entry.get(entry_id) or (max(max(r["i"], r["j"]) for r in rows)))
            pairs = [(int(r["i"]), int(r["j"])) for r in rows]
            gt = _reconstruct_matrix(pairs, [float(r["gt"]) for r in rows], n)
            pred = _reconstruct_matrix(pairs, [float(r["score"]) for r in rows], n)
            fig = figure_gt_vs_predicted(
                entry_id=entry_id, n_residues=n, gt=gt, pred=pred,
                score_label=score_label,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
