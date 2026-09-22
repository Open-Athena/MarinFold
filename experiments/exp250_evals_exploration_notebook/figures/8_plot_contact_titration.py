#!/usr/bin/env python
# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""8c · plot — what one more contact is worth.

Draws the dataset :mod:`8_make_titration_data` wrote. No GPU, no model, nothing
folded: every structure on screen was predicted once and stored.

``top7_rollout_consensus`` ends where MarinFold ends — a contact map. This picks
it up there and asks what the map is *for*. Three panels, one frame per contact:

* **left** — the contact map filling up, best-ranked contact first, each one
  coloured by whether the deposited structure agrees.
* **centre** — Helico's prediction from exactly those contacts, superposed on
  the deposited structure.
* **right** — lDDT against the number of contacts supplied, drawn as it goes.

The protein is ``8ubs_A``, where the whole distance is visible: Helico alone
manages 0.22 lDDT, and the same model given MarinFold's contacts reaches the
oracle ceiling.

**Superposition.** Kabsch on CA atoms with outlier rejection, refined from
several starting sets — the whole chain, and contiguous windows along it — and
the fit that ends with the most CA within ``TRIM_CUTOFF`` wins. Plain least
squares over all 150 is the wrong tool for the early frames: with most of the
chain misplaced it splits the difference and puts *everything* slightly wrong, so
a correctly folded core never appears to settle. Outlier rejection seeded from
that same smeared fit is no better — no pair is within the cutoff, so the search
stops before it starts. Seeding from fragments is what finds the core, because a
partly-correct prediction is usually correct in stretches.

What is drawn under the panel is how many CA land within ``TRIM_CUTOFF``
afterwards, not the set the last cycle was fitted over. Those differ, and the
fitted set is the flattering one.

**Rendering** is PyMOL, once per frame, into ``.cache/8_titration_frames/``.
Cached on the coordinates' digest, so re-running to change the matplotlib half
costs nothing. The prediction is drawn as a uniform tube rather than a cartoon:
cartoon geometry re-derives secondary structure per frame, and a marginal helix
flickering between helix and loop reads as the model changing its mind when it is
the renderer changing its mind. The deposited structure is static, so it gets a
real cartoon.

    .venv/bin/python 8_plot_contact_titration.py
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")

import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.patches import Rectangle                               # noqa: E402

import figlib                                                          # noqa: E402

DATASET = "8_contact_titration"
DPI = 100
#: 64 rather than 128: the flat-shaded render uses few distinct colours, and the palette is the
#: one lever on file size that costs nothing visible here.
PALETTE_COLORS = 64

#: Shared with `7_plot_rollout_animation`: a predicted contact is red when the deposited structure
#: agrees and slate when it does not, wherever it is drawn.
HIT = "#C44E52"
MISS = "#7E97B3"
GROUND_TRUTH = "#2F2F2F"
BAND = "#F7F7F7"
DIAGONAL = "#D5D5D5"
PREDICTED_3D = "#C44E52"      # the tube, matching the contacts that produced it
DEPOSITED_3D = "#B8B8B8"

#: Superposition. 2 A is PyMOL's `align` cutoff; the refinement never fits fewer than
#: `TRIM_MIN_FRACTION` of the CA atoms, so a seed that starts badly still has something to fit and
#: moves, rather than stalling on an empty selection.
TRIM_CYCLES, TRIM_CUTOFF, TRIM_MIN_FRACTION = 8, 2.0, 0.15

#: Frame timing, milliseconds. The fold appears between the 10th and 20th contact and the rest is
#: a plateau, so the first SLOW_FRAMES are held and the plateau is run through.
SLOW_FRAMES, SLOW_MS, FAST_MS, HOLD_MS = 24, 230, 60, 2600

RAY_SIZE = 900                # PyMOL render, square, cropped to its ink afterwards
PYMOL = os.environ.get("PYMOL") or shutil.which("pymol") or str(Path.home() / "pymolenv/bin/pymol")

THREE_LETTER = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
    "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
    "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


# --------------------------------------------------------------------------------------------
# The dataset
# --------------------------------------------------------------------------------------------


def load_dataset() -> dict:
    """Everything the two halves of this script read, plus what `describe` printed."""
    metadata = figlib.describe(DATASET)
    directory = figlib.require(DATASET, "pred_coords.npy", "gt_coords.npy", "atom_index.csv",
                               "metrics.csv", "ranked_contacts.csv", "true_contacts.npy",
                               "sequence.txt")
    sequence = (directory / "sequence.txt").read_text().strip()
    return dict(
        metadata=metadata, directory=directory, sequence=sequence, length=len(sequence),
        pred_coords=np.load(directory / "pred_coords.npy"),
        gt_coords=np.load(directory / "gt_coords.npy"),
        atom_index=pd.read_csv(directory / "atom_index.csv"),
        metrics=pd.read_csv(directory / "metrics.csv"),
        contacts=pd.read_csv(directory / "ranked_contacts.csv"),
        true_contacts=np.load(directory / "true_contacts.npy"),
        stem=metadata["parameters"]["protein"],
        pdb_id=metadata["parameters"]["pdb_id"].split("-")[0].upper(),
    )


# --------------------------------------------------------------------------------------------
# Superposition
# --------------------------------------------------------------------------------------------


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rotation + translation taking ``mobile`` onto ``target``."""
    mobile_centre, target_centre = mobile.mean(axis=0), target.mean(axis=0)
    covariance = (mobile - mobile_centre).T @ (target - target_centre)
    left, _, right = np.linalg.svd(covariance)
    # Reflections are not rotations: flipping the least-significant axis when the determinant is
    # negative is what keeps this a proper superposition rather than a mirror image that fits
    # slightly better.
    sign = np.sign(np.linalg.det(right.T @ left.T))
    correction = np.diag([1.0, 1.0, sign])
    rotation = right.T @ correction @ left.T
    return rotation, target_centre - rotation @ mobile_centre


def _refine(mobile: np.ndarray, target: np.ndarray, keep: np.ndarray, cutoff: float,
            cycles: int, floor: int):
    """Grow a superposition out of one starting set: fit, re-select what is close, repeat.

    When fewer than ``floor`` pairs are within ``cutoff`` the selection falls back to the
    ``floor`` closest, so a cycle always has something to fit and the search moves instead of
    stalling on a seed that started badly.
    """
    for _ in range(cycles):
        rotation, translation = kabsch(mobile[keep], target[keep])
        distance = np.linalg.norm(mobile @ rotation.T + translation - target, axis=1)
        proposed = distance < cutoff
        if proposed.sum() < floor:
            proposed = np.zeros_like(keep)
            proposed[np.argsort(distance)[:floor]] = True
        if np.array_equal(proposed, keep):
            break
        keep = proposed
    rotation, translation = kabsch(mobile[keep], target[keep])
    distance = np.linalg.norm(mobile @ rotation.T + translation - target, axis=1)
    return (rotation, translation), distance < cutoff


def robust_superpose(mobile: np.ndarray, target: np.ndarray, *, cycles: int = TRIM_CYCLES,
                     cutoff: float = TRIM_CUTOFF,
                     min_fraction: float = TRIM_MIN_FRACTION) -> tuple[np.ndarray, np.ndarray]:
    """Best superposition found from several starting sets: ``(rotation, translation)`` and core.

    Iterative outlier rejection alone is not robust, because it inherits whatever the first fit
    gave it. Seeded from all 150 Cα with half the chain misplaced, the first fit is smeared, *no*
    pair is within the cutoff, and the search stops before it starts — reporting no core for a
    prediction whose other half is exactly right. That is the case this exists for.

    So the refinement is run from several seeds — the whole chain, and contiguous windows along
    it, the way a structural aligner seeds from fragments — and the one that ends with the most
    Cα within ``cutoff`` wins. A contiguous window is the right shape of guess: a partly-correct
    prediction is usually correct in stretches.

    The returned mask is what ends up within ``cutoff``, which is what the figure reports. It is
    not the set the last cycle was fitted over: those differ, and the fitted set is the flattering
    one.
    """
    length = len(mobile)
    floor = max(3, int(min_fraction * length))
    window = max(floor, length // 4)
    seeds = [np.ones(length, bool)]
    for start in range(0, max(1, length - window + 1), max(1, window // 2)):
        seed = np.zeros(length, bool)
        seed[start:start + window] = True
        seeds.append(seed)

    best, best_core = None, None
    for seed in seeds:
        transform, core = _refine(mobile, target, seed, cutoff, cycles, floor)
        if best_core is None or core.sum() > best_core.sum():
            best, best_core = transform, core
    return best, best_core


def superpose_all(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Every predicted structure moved onto the deposited one, and how much of it lands close.

    The fit is computed on CA atoms and applied to every atom: a superposition weighted by how
    many atoms a residue happens to have is a superposition that cares more about tryptophan.
    """
    is_ca = ((data["atom_index"].atom_name == "CA")
             & (data["atom_index"].entity_type == "protein")).to_numpy()
    target = data["gt_coords"][is_ca].astype(np.float64)
    aligned = np.empty_like(data["pred_coords"])
    core = np.empty(len(data["pred_coords"]), int)
    for k, coords in enumerate(data["pred_coords"]):
        coords = coords.astype(np.float64)
        (rotation, translation), keep = robust_superpose(coords[is_ca], target)
        aligned[k] = (coords @ rotation.T + translation).astype(np.float32)
        core[k] = int(keep.sum())
    return aligned, core


# --------------------------------------------------------------------------------------------
# Rendering the structures (PyMOL)
# --------------------------------------------------------------------------------------------


def pdb_lines(coords: np.ndarray, atom_index: pd.DataFrame, sequence: str) -> str:
    """Protein atoms as a PDB string. Ligand atoms are dropped — they are not part of the fold.

    Residue names come from the sequence rather than the atom table, which does not carry them;
    without them PyMOL sees UNK everywhere and refuses to build a cartoon.
    """
    lines = []
    serial = 0
    for (position, row), point in zip(atom_index.iterrows(), coords, strict=True):
        if row.entity_type != "protein":
            continue
        serial += 1
        residue = THREE_LETTER.get(sequence[int(row.res_seq_id)], "UNK")
        name = row.atom_name
        # PDB column rules: a 1-3 character atom name starts in column 14, a 4-character one in
        # column 13. Left-justifying every name puts CA in carbon-alpha's column for some readers
        # and in calcium's for others.
        spelled = f"{name:<4s}" if len(name) == 4 else f" {name:<3s}"
        # Columns, because PDB is a fixed-width format and being one short is not an error
        # anywhere: 13-16 atom name, 17 altLoc, 18-20 resName, 21 blank, 22 chain, 23-26 resSeq,
        # 27-30 iCode + blanks, 31-54 coordinates. Omitting the altLoc blank shifts resSeq by one
        # character, every atom parses into its own residue, and PyMOL draws 150 disconnected
        # cartoon stubs instead of a chain.
        lines.append(
            f"ATOM  {serial:5d} {spelled} {residue:>3s} {row.chain_id}"
            f"{int(row.res_seq_id) + 1:4d}    "
            f"{point[0]:8.3f}{point[1]:8.3f}{point[2]:8.3f}  1.00  0.00"
            f"{row.element:>12s}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def render_structures(data: dict, aligned: np.ndarray, cache: Path) -> list[Path]:
    """One PyMOL render per frame, cached on the coordinates that produced it.

    The deposited structure is loaded once and the camera fixed on it, so the view is identical in
    every frame and the prediction is the only thing that moves.
    """
    digest = hashlib.sha256(aligned.tobytes()).hexdigest()[:16]
    frames_dir = cache / digest
    reference = frames_dir / "deposited.png"
    expected = [frames_dir / f"{k:04d}.png" for k in range(len(aligned))]
    if reference.exists() and all(path.exists() for path in expected):
        print(f"structures    cached at {frames_dir}")
        return expected, reference
    if not Path(PYMOL).exists() and shutil.which(PYMOL) is None:
        raise SystemExit(
            f"PyMOL not found at {PYMOL!r}. Install it with:\n"
            f"    conda create -y -p ~/pymolenv -c conda-forge pymol-open-source\n"
            f"or point the PYMOL environment variable at an existing binary.")

    frames_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="titration-pdb-"))
    (scratch / "deposited.pdb").write_text(
        pdb_lines(data["gt_coords"], data["atom_index"], data["sequence"]))
    for k, coords in enumerate(aligned):
        (scratch / f"pred{k:04d}.pdb").write_text(
            pdb_lines(coords, data["atom_index"], data["sequence"]))

    script = [
        # The camera is set once and never touched again, so every frame shares it and the
        # prediction is the only thing that moves. `auto_zoom` is PyMOL's default-on habit of
        # reframing on each `load`, which would re-centre the view 151 times.
        "set auto_zoom, off",
        f"load {scratch / 'deposited.pdb'}, deposited",
        "hide everything",
        "show cartoon, deposited",
        "dss deposited",
        "set cartoon_fancy_helices, 1",
        f"color 0x{DEPOSITED_3D[1:]}, deposited",
        # ray_trace_mode 3 is the flat, outlined, poster look. It is chosen for the GIF as much as
        # for the style: smooth shading is thousands of near-identical colours that a 128-entry
        # palette cannot hold and inter-frame compression cannot exploit.
        "set ray_trace_mode, 3",
        "set ray_trace_color, grey40",
        "set ray_opaque_background, 1",
        "bg_color white",
        "set antialias, 2",
        "set ray_shadows, 0",
        "set cartoon_tube_radius, 0.5",
        "orient deposited",
        "zoom deposited, 7",       # generous margin: an unfolded prediction is much larger
        # The deposited structure alone, first: `crop_to_ink` takes the shared crop box from it.
        f"ray {RAY_SIZE}, {RAY_SIZE}",
        f"png {reference}, dpi={DPI}",
    ]
    for k in range(len(aligned)):
        script += [
            f"load {scratch / f'pred{k:04d}.pdb'}, pred",
            "hide everything, pred",
            "show cartoon, pred",
            # A uniform tube, not a cartoon: see the module docstring. `cartoon tube` overrides
            # whatever secondary structure would otherwise be assigned, per frame.
            "cartoon tube, pred",
            f"color 0x{PREDICTED_3D[1:]}, pred",
            f"ray {RAY_SIZE}, {RAY_SIZE}",
            f"png {frames_dir / f'{k:04d}.png'}, dpi={DPI}",
            "delete pred",
        ]
    script_path = scratch / "render.pml"
    script_path.write_text("\n".join(script))
    print(f"structures    rendering {len(aligned)} frames with PyMOL ...")
    result = subprocess.run([PYMOL, "-cq", str(script_path)], capture_output=True, text=True)
    missing = [path for path in [reference, *expected] if not path.exists()]
    if result.returncode != 0 or missing:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise RuntimeError(f"PyMOL rendered {len(expected) - len(missing)} of {len(expected)} "
                           f"frames; its output is above")
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"structures    {len(expected)} frames in {frames_dir}")
    return expected, reference


def crop_to_ink(paths: list[Path], reference: Path, margin: float = 0.12) -> list[np.ndarray]:
    """Crop every render to one box, taken from ``reference`` and padded by ``margin``.

    One shared box, because a per-frame crop makes the structure jitter. The box comes from the
    deposited structure alone rather than from the union over frames: with no contacts the
    prediction sprawls over several times the fold's extent, and a box that contains it would
    shrink the fold to a speck in all 150 frames where it is right. A prediction wider than the
    box is clipped, which reads as what it is.
    """
    images = [np.asarray(Image.open(path).convert("RGB")) for path in paths]
    ink = np.any(np.asarray(Image.open(reference).convert("RGB")) < 245, axis=-1)
    rows, columns = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    pad = int(margin * max(rows[-1] - rows[0], columns[-1] - columns[0]))
    top, bottom = max(rows[0] - pad, 0), min(rows[-1] + pad, ink.shape[0])
    left, right = max(columns[0] - pad, 0), min(columns[-1] + pad, ink.shape[1])
    return [image[top:bottom, left:right] for image in images]


# --------------------------------------------------------------------------------------------
# The frame
# --------------------------------------------------------------------------------------------


class Frame:
    """One reusable figure: contact map, structure, curve. Artists are updated, not rebuilt."""

    SIZE = (12.6, 4.8)
    MAP_RECT = (0.038, 0.155, 3.5 / SIZE[0], 3.5 / SIZE[1])
    STRUCTURE_RECT = (0.335, 0.135, 3.9 / SIZE[0], 3.9 / SIZE[1])
    CURVE_RECT = (0.705, 0.235, 0.265, 0.545)

    def __init__(self, data: dict, header: str, subheader: str):
        figlib.figure_style(DPI)
        self.data = data
        length = data["length"]
        self.figure = plt.figure(figsize=self.SIZE, dpi=DPI)
        self.figure.patch.set_facecolor("white")
        self.figure.text(0.038, 0.955, header, fontsize=12.5, va="top")
        self.figure.text(0.038, 0.905, subheader, fontsize=9.5, va="top", color="#555555")

        # --- the contact map ----------------------------------------------------------------
        self.axis = self.figure.add_axes(self.MAP_RECT)
        self.canvas = self.base_canvas()
        self.image = self.axis.imshow(self.canvas, origin="lower", interpolation="nearest",
                                      vmin=0, vmax=1)
        self.axis.plot([0, length - 1], [0, length - 1], color=DIAGONAL, lw=0.6, zorder=2)
        self.axis.set(xlabel="residue", ylabel="residue")
        self.axis.set_xticks([0, length // 2, length - 1])
        self.axis.set_yticks([0, length // 2, length - 1])
        for spine in ("top", "right"):
            self.axis.spines[spine].set_visible(True)
        # By the last frame both triangles are full, and a bare label sits in a thicket of
        # cells. The panel is mirrored, so neither corner can be kept clear; a barely-there
        # backing keeps the words legible while the contacts under them still read.
        backing = dict(boxstyle="square,pad=0.25", facecolor="white", alpha=0.78,
                       edgecolor="none")
        self.axis.text(0.035, 0.96, "MarinFold\ncontacts supplied", transform=self.axis.transAxes,
                       ha="left", va="top", fontsize=9, color=HIT, linespacing=1.4, bbox=backing)
        self.axis.text(0.965, 0.045,
                       f"Experimentally-determined\nstructure ({data['pdb_id']})",
                       transform=self.axis.transAxes, ha="right", va="bottom", fontsize=9,
                       color=GROUND_TRUTH, linespacing=1.4, bbox=backing)
        self.highlight = Rectangle((-10, -10), 3.2, 3.2, fill=False, edgecolor="#111111", lw=1.1,
                                   zorder=3)
        self.axis.add_patch(self.highlight)
        self.map_status = self.figure.text(0.038, 0.062, "", fontsize=10, va="center")
        self.map_detail = self.figure.text(0.038, 0.022, "", fontsize=9, va="center",
                                           color="#555555")

        # --- the structures -----------------------------------------------------------------
        self.structure = self.figure.add_axes(self.STRUCTURE_RECT)
        self.structure.set_axis_off()
        self.structure_image = None
        self.structure_status = self.figure.text(0.353, 0.062, "", fontsize=10, va="center")
        self.structure_detail = self.figure.text(0.353, 0.022, "", fontsize=9, va="center",
                                                 color="#555555")
        # The key lives inside the structure axes, in its own colours, the way the map's two
        # corner labels do. As swatches on the figure it sat at the same height as the subtitle
        # and the two collided at whatever width the subtitle happened to be.
        for row, (colour, label) in enumerate(
                ((DEPOSITED_3D, "deposited structure"), (PREDICTED_3D, "Helico prediction"))):
            self.structure.text(0.98, 0.985 - row * 0.045, label, ha="right", va="top",
                                transform=self.structure.transAxes, fontsize=9, color=colour)

        # --- the curve ----------------------------------------------------------------------
        self.curve_axis = self.figure.add_axes(self.CURVE_RECT)
        self.curve_axis.set(xlabel="MarinFold contacts supplied", ylabel="lDDT")
        self.curve_axis.set_xlim(0, len(data["metrics"]) - 1)
        self.curve_axis.set_ylim(0, 1.0)
        self.curve, = self.curve_axis.plot([], [], color=HIT, lw=1.8, zorder=3)
        self.curve_point, = self.curve_axis.plot([], [], "o", color=HIT, ms=4.5, zorder=4)
        self.curve_label = self.curve_axis.text(0.96, 0.06, "", ha="right", va="bottom",
                                                transform=self.curve_axis.transAxes,
                                                fontsize=10, color=HIT)

    def base_canvas(self) -> np.ndarray:
        """White, the unscorable band, and the deposited structure's contacts below the diagonal.

        With ``origin="lower"`` element ``[i, j]`` draws at ``x = j, y = i``, so the visually
        upper-left triangle is ``i > j``. The supplied contacts go there and the experimental map
        goes opposite it — the mirrored layout `1_plot_top7_heatmap` uses.
        """
        from matplotlib.colors import to_rgb

        length = self.data["length"]
        canvas = np.ones((length, length, 3))
        index = np.arange(length)
        canvas[np.abs(np.subtract.outer(index, index)) < figlib.MIN_SEPARATION] = to_rgb(BAND)
        lower_right = np.triu(np.ones((length, length), bool), k=1)
        canvas[self.data["true_contacts"] & lower_right] = to_rgb(GROUND_TRUTH)
        return canvas

    def draw(self) -> np.ndarray:
        self.figure.canvas.draw()
        return np.asarray(self.figure.canvas.buffer_rgba())[..., :3].copy()


def frames(frame: Frame, data: dict, renders: list[np.ndarray], core: np.ndarray):
    """Yield ``(rgb, milliseconds)``, one frame per contact count."""
    from matplotlib.colors import to_rgb

    metrics = data["metrics"]
    contacts = data["contacts"]
    canvas = frame.base_canvas()
    length = data["length"]
    hits = 0

    for k in range(len(metrics)):
        if k:
            row = contacts.iloc[k - 1]
            hits += int(row.true_contact)
            canvas[int(row.seq_j), int(row.seq_i)] = to_rgb(HIT if row.true_contact else MISS)
            frame.highlight.set_xy((int(row.seq_i) - 1.6, int(row.seq_j) - 1.6))
        frame.image.set_data(canvas)

        if frame.structure_image is None:
            frame.structure_image = frame.structure.imshow(renders[k])
        else:
            frame.structure_image.set_data(renders[k])

        measured = metrics.iloc[k]
        frame.map_status.set_text(f"{k} of {len(metrics) - 1} contacts supplied")
        frame.map_detail.set_text(
            f"{hits} of {k} are in the deposited structure ({hits / k:.0%})" if k else
            f"{int(np.triu(data['true_contacts'], figlib.MIN_SEPARATION).sum())} "
            f"experimental contacts to find")
        frame.structure_status.set_text(f"lDDT {measured.lddt:.3f}")
        frame.structure_detail.set_text(
            f"backbone RMSD {measured.rmsd:.1f} Å · {core[k]} of {length} "
            f"Cα within {TRIM_CUTOFF:.0f} Å")
        frame.curve.set_data(np.arange(k + 1), metrics.lddt.to_numpy()[:k + 1])
        frame.curve_point.set_data([k], [measured.lddt])
        frame.curve_label.set_text(f"{measured.lddt:.3f}")
        yield frame.draw(), (SLOW_MS if k <= SLOW_FRAMES else FAST_MS)


def write_gif(name: str, images, durations) -> None:
    """Save frames under one shared palette, plus the last frame as a PNG still."""
    figlib.OUTPUT.mkdir(parents=True, exist_ok=True)
    sample = np.concatenate(images[:: max(1, len(images) // 12)], axis=0)
    palette = Image.fromarray(sample).quantize(colors=PALETTE_COLORS,
                                               method=Image.Quantize.MEDIANCUT)
    quantized = [Image.fromarray(rgb).quantize(palette=palette, dither=Image.Dither.NONE)
                 for rgb in images]
    path = figlib.OUTPUT / f"{name}.gif"
    quantized[0].save(path, save_all=True, append_images=quantized[1:], duration=durations,
                      loop=0, optimize=True, disposal=1)
    Image.fromarray(images[-1]).save(figlib.OUTPUT / f"{name}.png")
    print(f"wrote {path}  {len(images)} frames · {sum(durations) / 1000:.1f}s · "
          f"{path.stat().st_size / 2**20:.2f} MiB")


def main() -> None:
    """Superpose, render, animate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=str(figlib.FIGURES / ".cache/8_titration_frames"),
                        help="where PyMOL renders are kept between runs")
    arguments = parser.parse_args()

    data = load_dataset()
    metrics = data["metrics"]
    aligned, core = superpose_all(data)
    print(f"\nsuperposition core {core.min()}-{core.max()} of {data['length']} CA "
          f"(median {int(np.median(core))})")
    print(f"lDDT          {metrics.lddt.iloc[0]:.3f} with no contacts -> "
          f"{metrics.lddt.iloc[-1]:.3f} with {len(metrics) - 1} · "
          f"best {metrics.lddt.max():.3f} at k={int(metrics.lddt.idxmax())}")
    reached = metrics.index[metrics.lddt >= 0.9 * metrics.lddt.max()]
    print(f"contacts      {int(reached[0])} reach 90% of the best lDDT · "
          f"top-{len(metrics) - 1} precision {data['contacts'].true_contact.mean():.3f}")

    renders = crop_to_ink(*render_structures(data, aligned, Path(arguments.cache)))
    frame = Frame(data, f"{data['pdb_id']} · one contact at a time",
                  "MarinFold's contacts, best first, folded by Helico at every step")
    images, durations = zip(*frames(frame, data, renders, core))
    images, durations = list(images), list(durations)
    durations[-1] = HOLD_MS
    write_gif("contact_titration_8ubs", images, durations)
    plt.close(frame.figure)

    # The curve on its own, as a still: it is the quantitative claim the animation makes and a
    # document that cannot show a GIF still needs it.
    figure, axis = plt.subplots(figsize=(3.6, 2.8), layout="constrained")
    axis.plot(metrics.k, metrics.lddt, color=HIT, lw=1.8)
    axis.set(xlabel="MarinFold contacts supplied", ylabel="lDDT", ylim=(0, 1.0),
             xlim=(0, metrics.k.max()))
    figlib.save_figure(figure, "contact_titration_lddt", DPI)
    plt.close(figure)

    summary = {"lddt_k0": float(metrics.lddt.iloc[0]), "lddt_max": float(metrics.lddt.max()),
               "k_best": int(metrics.lddt.idxmax()),
               "k_90pct": int(reached[0]), "core_median": int(np.median(core))}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
