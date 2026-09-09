"""Geometry checks for designed monomers; thresholds are recorded in every result."""

import numpy as np


def confidence_percent(normalized_confidence: float) -> float:
    """Convert the pinned ESMFold API's confidence into the corpus's 0–100 unit."""
    if not 0 <= normalized_confidence <= 1:
        raise ValueError("Expected normalized ESMFold confidence")
    return 100 * normalized_confidence


def aligned_rmsd(reference: np.ndarray, moving: np.ndarray) -> float:
    """Return full-chain RMSD under a proper rotation, excluding reflections."""
    if (
        reference.shape != moving.shape
        or reference.ndim != 2
        or reference.shape[1] != 3
    ):
        raise ValueError("RMSD requires matching (residues, 3) arrays")
    if (
        len(reference) < 3
        or not np.isfinite(reference).all()
        or not np.isfinite(moving).all()
    ):
        raise ValueError("RMSD requires at least three finite coordinate triplets")
    x = reference.astype(np.float64) - reference.mean(axis=0)
    y = moving.astype(np.float64) - moving.mean(axis=0)
    u, _, vh = np.linalg.svd(y.T @ x)
    handedness = np.eye(3)
    handedness[2, 2] = np.linalg.det(u @ vh)
    aligned = y @ u @ handedness @ vh
    return float(np.sqrt(np.mean(np.sum((x - aligned) ** 2, axis=1))))


def ca_geometry(coordinates: np.ndarray) -> dict[str, float | int | bool]:
    """Check C-alpha chain continuity and severe nonlocal steric clashes."""
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) < 3:
        raise ValueError("Expected at least three C-alpha coordinates")
    if not np.isfinite(coordinates).all():
        raise ValueError("Nonfinite coordinates")
    distances = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :], axis=-1
    )
    adjacent = np.diag(distances, 1)
    nonlocal_pairs = np.triu(np.ones(distances.shape, dtype=bool), k=3)
    clash_count = int(np.sum((distances < 2.5) & nonlocal_pairs))
    break_count = int(np.sum((adjacent < 3.4) | (adjacent > 4.2)))
    return {
        "ca_clashes": clash_count,
        "ca_chain_breaks": break_count,
        "ca_step_min": float(adjacent.min()),
        "ca_step_max": float(adjacent.max()),
        "ca_geometry_pass": clash_count == 0 and break_count == 0,
    }
