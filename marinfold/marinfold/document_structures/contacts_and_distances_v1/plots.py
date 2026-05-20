# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-and-distances-v1 PDF plot writers for ``infer`` / ``evaluate``.

The two top-level entry points — :func:`plot_infer_pdf` and
:func:`plot_evaluate_pdf` — open a multi-page PDF and write one
heatmap page per (structure, n_seeded) combination from the
records / :class:`marinfold.EvalResult` the CLI already produced.

Today there is one figure shape per command:

- ``infer``: a single predicted CA-CA expected-distance heatmap.
- ``evaluate``: GT (left) and predicted (right) side-by-side, sharing
  a color scale. Pairs masked out of the eval (non-finite GT or GT >
  ``distance_cap_angstrom``) render as NaN in both panels, so the
  heatmap shows exactly which pairs were scored.

Adding more plots later means: write another ``figure_*`` function
and append its output to the page list in the assembler.

``matplotlib`` is required for these helpers and is pulled in by the
``marinfold[contacts-and-distances-v1]`` extra. The import is at
call time so the rest of the impl loads without matplotlib
installed.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from marinfold import EvalResult


# Distance-bin saturation point (Å) — mirrors the
# ``_DISTANCE_MAX_A`` constant in ``inference.py``. Used as the color
# scale upper bound so brightness is comparable across structures.
_DISTANCE_MAX_A = 32.0


# --------------------------------------------------------------------------
# Matrix reconstruction
# --------------------------------------------------------------------------


def _reconstruct_matrix(
    pairs: Iterable[tuple[int, int]],
    values: Iterable[float],
    n_residues: int,
) -> np.ndarray:
    """N×N matrix filled at (i-1, j-1) and (j-1, i-1) with ``values``.

    Pairs are 1-indexed (i, j) with i < j, matching the per-pair
    convention used throughout the impl. The diagonal is 0; any
    (i, j) not in ``pairs`` stays NaN, which renders as transparent /
    grey under matplotlib's default cmap.
    """
    out = np.full((n_residues, n_residues), np.nan, dtype=np.float32)
    np.fill_diagonal(out, 0.0)
    for (i, j), v in zip(pairs, values, strict=True):
        out[i - 1, j - 1] = v
        out[j - 1, i - 1] = v
    return out


# --------------------------------------------------------------------------
# Figure builders
# --------------------------------------------------------------------------


def figure_predicted_heatmap(record: dict[str, Any]):
    """Single-panel predicted CA-CA expected-distance heatmap.

    ``record`` is one entry from :func:`inference.predict` — a dict
    with ``entry_id``, ``n_residues``, ``n_seeded``, ``query_atom``,
    ``pairs``, ``expected_distances``.
    """
    import matplotlib.pyplot as plt

    entry_id = record["entry_id"]
    n = int(record["n_residues"])
    n_seeded = int(record["n_seeded"])
    query_atom = record["query_atom"]
    pred = _reconstruct_matrix(
        record["pairs"], record["expected_distances"], n,
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(pred, vmin=0, vmax=_DISTANCE_MAX_A, cmap="viridis")
    ax.set_title(f"Predicted {query_atom}-{query_atom} (Å)")
    ax.set_xlabel("residue j")
    ax.set_ylabel("residue i")
    fig.colorbar(im, ax=ax, fraction=0.046, label="expected distance (Å)")
    fig.suptitle(f"{entry_id}  (n_residues={n}, n_seeded={n_seeded})")
    fig.tight_layout()
    return fig


def figure_gt_vs_predicted(
    *,
    entry_id: str,
    n_residues: int,
    n_seeded: int,
    query_atom: str,
    gt: np.ndarray,
    pred: np.ndarray,
    mae: float,
):
    """Two-panel figure: GT (left), predicted (right), shared color scale."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(gt, vmin=0, vmax=_DISTANCE_MAX_A, cmap="viridis")
    axes[0].set_title(f"GT {query_atom}-{query_atom} (Å)")
    im1 = axes[1].imshow(pred, vmin=0, vmax=_DISTANCE_MAX_A, cmap="viridis")
    axes[1].set_title(f"Predicted {query_atom}-{query_atom} (Å)")
    for ax in axes:
        ax.set_xlabel("residue j")
        ax.set_ylabel("residue i")
    # Right-edge colorbar shared between the two panels — they're on
    # the same scale, so one bar avoids duplicating the legend.
    fig.colorbar(im1, ax=axes, fraction=0.046, label="distance (Å)")
    mae_str = f"{mae:.2f} Å" if np.isfinite(mae) else "NaN"
    fig.suptitle(
        f"{entry_id}  (n_residues={n_residues}, n_seeded={n_seeded}, "
        f"MAE={mae_str})"
    )
    return fig


# --------------------------------------------------------------------------
# PDF assemblers
# --------------------------------------------------------------------------


def plot_infer_pdf(out_path: Path, records: list[dict[str, Any]]) -> None:
    """Write one predicted-heatmap page per (structure, n_seeded).

    ``records`` is the materialized output of :func:`inference.predict`
    (the CLI lists it before calling us so the writer + plotter both
    see the same data).
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        raise SystemExit("plot_infer_pdf: no records to plot")

    with PdfPages(out_path) as pdf:
        for record in records:
            fig = figure_predicted_heatmap(record)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def plot_evaluate_pdf(out_path: Path, result: EvalResult) -> None:
    """Write one GT-vs-predicted page per (structure, n_seeded).

    Reconstructs the GT and predicted matrices from
    ``result.per_example`` (which holds every evaluated pair's GT and
    expected distance). N per structure comes from
    ``result.extras["per_structure_n_residues"]``, populated by
    :func:`inference.evaluate`. MAE per (structure, n_seeded) comes
    from ``result.extras["per_structure_mae"]``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    extras = result.extras
    n_residues_by_entry: dict[str, int] = extras["per_structure_n_residues"]
    mae_by_n_by_entry: dict[Any, dict[str, float]] = extras["per_structure_mae"]
    query_atom = extras.get("query_atom", "CA")
    seed_n_values = list(extras.get("seed_n_values", []))
    if not seed_n_values:
        # Fall back to whatever n_seeded values appear in the rows.
        seed_n_values = sorted({int(r["n_seeded"]) for r in result.per_example})

    # Bucket per_example rows by (entry_id, n_seeded). The MAE-by-n
    # dict's keys can be ints or stringified ints depending on whether
    # extras went through a JSON round-trip; ``_lookup_mae`` handles
    # both.
    by_key: dict[tuple[str, int], list[dict]] = {}
    entry_order: list[str] = []
    seen_entries: set[str] = set()
    for row in result.per_example:
        entry_id = row["entry_id"]
        n_seeded = int(row["n_seeded"])
        by_key.setdefault((entry_id, n_seeded), []).append(row)
        if entry_id not in seen_entries:
            seen_entries.add(entry_id)
            entry_order.append(entry_id)

    if not by_key:
        raise SystemExit("plot_evaluate_pdf: EvalResult has no per_example rows to plot")

    with PdfPages(out_path) as pdf:
        for entry_id in entry_order:
            n = int(n_residues_by_entry[entry_id])
            for n_seeded in seed_n_values:
                rows = by_key.get((entry_id, int(n_seeded)))
                if not rows:
                    continue
                pairs = [(int(r["i"]), int(r["j"])) for r in rows]
                gt_vals = [float(r["gt_angstrom"]) for r in rows]
                pred_vals = [float(r["expected_angstrom"]) for r in rows]
                gt_mat = _reconstruct_matrix(pairs, gt_vals, n)
                pred_mat = _reconstruct_matrix(pairs, pred_vals, n)
                mae = _lookup_mae(mae_by_n_by_entry, n_seeded, entry_id)
                fig = figure_gt_vs_predicted(
                    entry_id=entry_id,
                    n_residues=n,
                    n_seeded=int(n_seeded),
                    query_atom=query_atom,
                    gt=gt_mat,
                    pred=pred_mat,
                    mae=mae,
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)


def _lookup_mae(
    mae_by_n_by_entry: dict[Any, dict[str, float]],
    n_seeded: int,
    entry_id: str,
) -> float:
    """Pull MAE out of ``per_structure_mae`` regardless of int/str key shape."""
    for key in (n_seeded, str(n_seeded)):
        inner = mae_by_n_by_entry.get(key)
        if inner is not None and entry_id in inner:
            return float(inner[entry_id])
    return float("nan")
