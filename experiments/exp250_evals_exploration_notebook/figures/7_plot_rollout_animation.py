#!/usr/bin/env python
# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""7 · plot — the Top7 contact map, animated.

Draws the dataset written by :mod:`7_make_rollout_animation_data`. No GPU, no
model: every statement is replayed from the stored rollouts.

Two GIFs, because they answer different questions:

* ``top7_rollout_emission`` — **one** rollout, one contact statement per frame,
  in the order the model wrote them. This is the animated counterpart of
  ``top7_maps``: the same protein and the same recipe, before the votes are
  taken.
* ``top7_rollout_consensus`` — that rollout, then the other 99 accumulating into
  the vote map the published figure shows, with the top-R precision of the
  running consensus drawn beside it.

The pair is the argument for the recipe. A single rollout is right about half the
time; the frames that follow are what turns that into 0.67.

**Which rollout.** The one whose F1 against the experimental contacts is closest
to the median over all 100, ties broken by the statement count closest to the
median — a typical rollout by construction, not a flattering one. The selection
is recomputed here and printed, so it moves with the dataset instead of being a
hard-coded index that quietly stops being typical.

Unlike the manuscript panels in this directory, these carry their own header and
labels: a GIF is a standalone asset that travels without the caption a figure in
a document keeps beside it.

    .venv/bin/python 7_plot_rollout_animation.py
"""

import json

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")   # frames are rasterised off-screen; never an interactive window

import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.cm import ScalarMappable                               # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb   # noqa: E402
from matplotlib.patches import Rectangle                               # noqa: E402

import figlib                                                          # noqa: E402

DATASET = "7_rollout_animation"
DPI = 100                        # frame pixels = figure inches x DPI; 9.6 x 5.4 in -> 960 x 540
PALETTE_COLORS = 128             # one palette for every frame: no inter-frame flicker, smaller GIF

#: Cell colours. `HIT` and `HEAT`'s upper end are the same red, so a contact that survives the
#: vote in the consensus GIF is the colour it was when the rollout emitted it.
GROUND_TRUTH = "#2F2F2F"
HIT = "#C44E52"
MISS = "#7E97B3"
BAND = "#F7F7F7"                 # |i - j| < 6: never scored, so never drawn on
DIAGONAL = "#D5D5D5"
NEWEST = "#111111"
FADED = "#9A9A9A"
#: The published map's colours (`1_plot_top7_heatmap`), so the consensus GIF ends on that figure.
HEAT = LinearSegmentedColormap.from_list("votes", ["#FFFFFF", "#F4C36B", "#C44E52", "#3B0A0C"])

#: Frame timing, milliseconds. The first statements are held long enough to read the tokens; the
#: rest run at a pace that gets through ~80 of them without outstaying the loop.
SLOW_STATEMENTS, SLOW_MS, FAST_MS = 6, 420, 95
REPLAY_MS = 55                   # the same statements again in the consensus GIF, at a gallop
FAST_ROLLOUT_MS = 55             # the consensus GIF's per-rollout frames
TRANSITION_MS = 1300             # the beat between the one rollout and the other 99
HOLD_MS = 2200                   # the last frame of either GIF

STREAM_LINES = 12                # statements visible in the token panel


def load_dataset():
    """The stored rollouts plus everything derived from them, as one namespace-ish dict."""
    metadata = figlib.describe(DATASET)
    directory = figlib.require(DATASET, "statements.csv", "rollouts.csv", "votes.npy",
                               "score.npy", "ground_truth.json")
    statements = pd.read_csv(directory / "statements.csv")
    rollouts = pd.read_csv(directory / "rollouts.csv")
    # A truncated rollout stopped because it ran out of token budget rather than
    # because the model wrote <end>, and its map is a rollout cut off mid-sentence.
    truncated = int(rollouts.truncated.sum())
    if truncated:
        print(f"note: {truncated} of {len(rollouts)} rollouts hit the token budget "
              "instead of writing <end> — their maps are incomplete")
    record = json.loads((directory / "ground_truth.json").read_text())
    length = int(record["L"])
    truth = figlib.true_matrix(length, record["contacts"])
    votes = np.load(directory / "votes.npy").astype(np.int64)
    # `score` is `votes + a pairwise tie-break in [0, 0.5)`; the tie-break does not depend on the
    # rollouts, so subtracting the votes back out recovers it and it can be re-added to a partial
    # vote matrix. That is what makes the running-precision curve below the same metric the
    # published number is, rather than a vote count with ties broken by array order.
    tiebreak = np.load(directory / "score.npy").astype(np.float64) - votes

    statements = statements.assign(
        separation=statements.seq_j - statements.seq_i,
        duplicate=statements.duplicated(subset=["rollout", "seq_i", "seq_j"]),
    )
    statements["scorable"] = ((statements.seq_i >= 0)
                              & (statements.separation >= figlib.MIN_SEPARATION))
    statements["hit"] = [bool(truth[i, j]) if scorable else False
                         for i, j, scorable in zip(statements.seq_i, statements.seq_j,
                                                   statements.scorable)]
    return dict(metadata=metadata, statements=statements, record=record,
                length=length, truth=truth, votes=votes, tiebreak=tiebreak,
                pdb_id=record["stem"].split("_")[0].upper(),
                n_rollouts=int(metadata["parameters"]["n_rollouts"]),
                n_true=int(np.triu(truth, figlib.MIN_SEPARATION).sum()))


def choose_rollout(data) -> int:
    """The median rollout: F1 closest to the median F1, then statement count closest to median.

    A rollout is scored on the distinct, scorable pairs it emitted — the same set that casts its
    votes — against the experimental contacts.
    """
    per_rollout = figlib.rollout_accuracy(data["statements"], data["truth"])
    chosen = figlib.median_rollout(per_rollout)
    row = per_rollout.loc[chosen]
    print(f"\nrollouts      precision {per_rollout.precision.mean():.3f} mean, "
          f"{per_rollout.precision.min():.3f}-{per_rollout.precision.max():.3f} over "
          f"{len(per_rollout)} · {per_rollout.n.median():.0f} statements median")
    print(f"featured      rollout {chosen}: {int(row.n)} distinct contacts, "
          f"{int(row.hits)} in the structure (precision {row.precision:.3f}, "
          f"recall {row.recall:.3f}, F1 {row.f1:.3f} against a median F1 of "
          f"{per_rollout.f1.median():.3f})")
    return chosen


def running_precision(data, order) -> list[float]:
    """Top-R precision of the consensus after each rollout in ``order`` is added.

    ``R`` is the number of experimental contacts, so this is the R-precision the repository
    quotes — the same cut, on a vote matrix that is still filling up.
    """
    counted = data["statements"][data["statements"].scorable & ~data["statements"].duplicate]
    by_rollout = {rollout: frame[["seq_i", "seq_j"]].to_numpy()
                  for rollout, frame in counted.groupby("rollout")}
    votes = np.zeros_like(data["votes"])
    curve = []
    for rollout in order:
        pairs = by_rollout.get(rollout)
        if pairs is not None:
            votes[pairs[:, 0], pairs[:, 1]] += 1
            votes[pairs[:, 1], pairs[:, 0]] += 1
        metrics = figlib.score_metrics(votes + data["tiebreak"], data["record"])
        curve.append(float(metrics[(metrics.range == "all") & (metrics.cut == "R")].value.iloc[0]))
    return curve


# --------------------------------------------------------------------------------------------
# The frame
# --------------------------------------------------------------------------------------------


def base_canvas(data) -> np.ndarray:
    """The map every frame starts from: white, the unscorable band, and the experimental contacts.

    With ``origin="lower"`` element ``[i, j]`` is drawn at ``x = j, y = i``, so the visually
    upper-left triangle is ``i > j`` and the lower-right is ``i < j``. The rollout is drawn in the
    upper-left and the experimental structure in the lower-right — the mirrored layout
    ``1_plot_top7_heatmap`` uses. Getting the two round the wrong way mirrors the figure while
    leaving every label reading correctly, which is as invisible as it sounds.
    """
    length = data["length"]
    canvas = np.ones((length, length, 3))
    index = np.arange(length)
    band = np.abs(np.subtract.outer(index, index)) < figlib.MIN_SEPARATION
    resolved = np.zeros(length, bool)
    resolved[np.asarray(data["record"]["resolved"])] = True
    canvas[band | ~np.outer(resolved, resolved)] = to_rgb(BAND)
    lower_right = np.triu(np.ones((length, length), bool), k=1)
    canvas[data["truth"] & lower_right] = to_rgb(GROUND_TRUTH)
    return canvas


class Frame:
    """One reusable matplotlib figure whose artists are updated per frame.

    Rebuilding the figure per frame is the obvious way to do this and costs about a second a
    frame; updating a fixed set of artists costs milliseconds, which is the difference between
    iterating on the design and not.
    """

    #: Figure geometry, in inches and figure fractions. Stated once because three of these
    #: rectangles have to agree: the map, the colour key beside it and the panel to their right.
    SIZE = (9.6, 5.4)
    MAP_INCHES = 4.0
    MAP_RECT = (0.045, 0.145, MAP_INCHES / SIZE[0], MAP_INCHES / SIZE[1])
    BAR_RECT = (0.487, 0.145, 0.013, MAP_INCHES / SIZE[1])
    RIGHT_RECT = (0.555, 0.145, 0.425, MAP_INCHES / SIZE[1])
    CURVE_RECT = (0.615, 0.265, 0.345, 0.44)
    HIGHLIGHT = 3.2      # cells across, for the box marking the newest statement

    def __init__(self, data, header: str, subheader: str, right: str):
        """``right`` selects the right-hand panel: ``"stream"`` or ``"curve"``."""
        figlib.figure_style(DPI)
        self.data = data
        length = data["length"]
        self.figure = plt.figure(figsize=self.SIZE, dpi=DPI)
        self.figure.patch.set_facecolor("white")

        # Title and subtitle as two artists rather than one two-line string: the second line says
        # what this particular GIF does and reads as the smaller of the two, which one text
        # object at one size cannot express.
        self.figure.text(0.045, 0.988, header, fontsize=12.5, va="top")
        self.figure.text(0.045, 0.938, subheader, fontsize=9.5, va="top", color="#555555")

        # Explicit rectangles rather than a layout engine: the map has to be the same square of
        # pixels in every frame, and a layout engine resizes an axes around whatever text it
        # happens to be carrying that frame.
        self.axis = self.figure.add_axes(self.MAP_RECT)
        self.image = self.axis.imshow(base_canvas(data), origin="lower", interpolation="nearest",
                                      vmin=0, vmax=1)
        self.axis.plot([0, length - 1], [0, length - 1], color=DIAGONAL, lw=0.6, zorder=2)
        self.axis.set(xlabel="residue", ylabel="residue")
        self.axis.set_xticks([0, length // 2, length - 1])
        self.axis.set_yticks([0, length // 2, length - 1])
        for spine in ("top", "right"):
            self.axis.spines[spine].set_visible(True)
        self.corner = self.axis.text(0.035, 0.96, "", transform=self.axis.transAxes, ha="left",
                                     va="top", fontsize=9, color=HIT, linespacing=1.4)
        self.axis.text(0.965, 0.045, f"Experimentally-determined\nstructure ({data['pdb_id']})",
                       transform=self.axis.transAxes, ha="right", va="bottom", fontsize=9,
                       color=GROUND_TRUTH, linespacing=1.4)
        # The newest statement's cell, outlined. Off-screen until the first statement lands.
        # Wider than the cell it marks: a 1-cell outline on a ~4 px cell is a smudge.
        self.highlight = Rectangle((-10, -10), self.HIGHLIGHT, self.HIGHLIGHT, fill=False,
                                   edgecolor=NEWEST, lw=1.1, zorder=3)
        self.axis.add_patch(self.highlight)

        self.status = self.figure.text(0.045, 0.062, "", fontsize=10, va="center")
        self.detail = self.figure.text(0.045, 0.022, "", fontsize=9, va="center", color="#555555")

        # The vote scale, beside the map and hidden until the map is showing votes. In the
        # single-rollout phase a cell is one of two colours and a continuous scale beside it
        # would be a scale for something that is not on the screen.
        self.bar_axis = self.figure.add_axes(self.BAR_RECT)
        bar = self.figure.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=HEAT),
                                   cax=self.bar_axis, ticks=[0, 0.5, 1.0])
        bar.set_label(f"fraction of the {data['n_rollouts']} rollouts\nasserting the contact",
                      fontsize=8, linespacing=1.5)
        bar.ax.set_yticklabels(["0", "", "1"], fontsize=8)
        self.bar_axis.set_visible(False)

        self.stream = self.figure.add_axes(self.RIGHT_RECT)
        self.stream.set_axis_off()
        self.stream.set(xlim=(0, 1), ylim=(0, 1))
        self.stream.set_visible(right == "stream")
        self.stream.text(0, 1.0, "what the model wrote", fontsize=9.5, va="top", color="#333333")
        self.stream.text(0.655, 1.0, "residues", fontsize=8.5, va="top", color="#999999")
        self.stream.plot([0, 1], [0.955, 0.955], color="#DDDDDD", lw=0.8, clip_on=False)
        self.lines = [self.stream.text(0.045, 0.895 - row * 0.058, "", fontsize=8.8, va="center",
                                       family="monospace")
                      for row in range(STREAM_LINES)]
        # The newest statement is the top line. The fade alone did not say so — a light blue
        # "not in it" line reads as more prominent than a dark red one below it whatever their
        # alphas are — so the cursor says it in a way colour cannot argue with.
        self.stream.plot([0.012], [0.895], marker=">", ms=5, color=NEWEST, clip_on=False)
        self._legend()

        self.curve_axis = self.figure.add_axes(self.CURVE_RECT)
        self.curve_axis.set_visible(right == "curve")
        self.curve_axis.set(xlabel="rollouts voting", ylabel="R-precision")
        self.curve_axis.set_xlim(0, data["n_rollouts"])
        self.curve, = self.curve_axis.plot([], [], color=HIT, lw=1.6, zorder=3)
        self.curve_point, = self.curve_axis.plot([], [], "o", color=HIT, ms=4.5, zorder=4)
        self.curve_label = self.curve_axis.text(0.97, 0.06, "", ha="right", va="bottom",
                                                transform=self.curve_axis.transAxes,
                                                fontsize=9.5, color=HIT)

    def _legend(self) -> None:
        """Two swatches under the token stream, in the map's own colours."""
        for row, (color, label) in enumerate(
                ((HIT, "Present in the experimental structure"), (MISS, "Not present"))):
            y = 0.10 - row * 0.075
            self.stream.add_patch(Rectangle((0.02, y - 0.018), 0.026, 0.036, facecolor=color,
                                            edgecolor="none", transform=self.stream.transAxes))
            self.stream.text(0.062, y, label, fontsize=9, va="center", color="#333333")

    def show_curve(self, curve) -> None:
        """Swap the token stream for the running-precision panel, with one rollout marked."""
        self.stream.set_visible(False)
        self.bar_axis.set_visible(True)
        self.curve_axis.set_visible(True)
        self.curve_axis.set_ylim(0, max(curve) * 1.25)
        # What a single rollout is worth, so the rise off it is the thing the panel shows.
        self.curve_axis.axhline(curve[0], color="#B0B0B0", lw=0.9, ls=":", zorder=2)
        self.curve_axis.text(self.data["n_rollouts"], curve[0], f" one rollout: {curve[0]:.3f}",
                             fontsize=8.5, color="#777777", ha="right", va="bottom")

    def draw(self) -> np.ndarray:
        """Rasterise the current state to an RGB array."""
        self.figure.canvas.draw()
        return np.asarray(self.figure.canvas.buffer_rgba())[..., :3].copy()


# --------------------------------------------------------------------------------------------
# The two GIFs
# --------------------------------------------------------------------------------------------


def statement_text(row) -> str:
    """One emitted statement as the model wrote it, with the residues it resolves to."""
    tokens = f"<{row.kind}> <p{row.pos_a}> <p{row.pos_b}>"
    if row.seq_i < 0:
        return f"{tokens:30s}  off-protein"
    # +1 so the residues read in the 1-based numbering a sequence is usually discussed in.
    return f"{tokens:30s}  {row.seq_i + 1:>3d} · {row.seq_j + 1:<3d}"


def emission_frames(frame: Frame, data, rollout: int, *, slow_statements=SLOW_STATEMENTS,
                    slow_ms=SLOW_MS, fast_ms=FAST_MS):
    """Yield ``(rgb, milliseconds)`` for one rollout, one statement per frame.

    The first frame is the empty map beside the experimental structure, so the animation opens on
    the question rather than part-way through the answer.
    """
    canvas = base_canvas(data)
    emitted = data["statements"][data["statements"].rollout == rollout].sort_values("order")
    history: list[tuple[str, str]] = []
    hits = 0

    frame.image.set_data(canvas)
    frame.corner.set_text("MarinFold\none rollout")
    frame.status.set_text(f"statement 0 of {len(emitted)}")
    frame.detail.set_text(f"the sequence is already written; {data['n_true']} contacts to find")
    yield frame.draw(), slow_ms

    for position, row in enumerate(emitted.itertuples(index=False), start=1):
        color = HIT if row.hit else MISS
        hits += int(row.hit)
        if row.seq_i >= 0:
            # The statement is symmetric; it is drawn once, in the rollout's triangle.
            canvas[row.seq_j, row.seq_i] = to_rgb(color)
            offset = 0.5 * Frame.HIGHLIGHT
            frame.highlight.set_xy((row.seq_i - offset, row.seq_j - offset))
        frame.image.set_data(canvas)

        # Newest statement on top, older ones fading down the panel so the stream reads as a
        # stream rather than as a table that happens to change.
        history.append((statement_text(row) + ("  (repeat)" if row.duplicate else ""), color))
        recent = list(reversed(history[-STREAM_LINES:]))
        for depth, line in enumerate(frame.lines):
            text, line_color = recent[depth] if depth < len(recent) else ("", FADED)
            line.set_text(text)
            line.set_color(line_color)
            line.set_alpha(1.0 if depth == 0 else max(0.14, 0.70 - 0.055 * (depth - 1)))

        frame.status.set_text(f"statement {position} of {len(emitted)}")
        frame.detail.set_text(f"{hits} of {position} are in the experimental structure "
                              f"({hits / position:.0%})")
        yield frame.draw(), (slow_ms if position <= slow_statements else fast_ms)


def consensus_frames(frame: Frame, data, order, curve):
    """Yield ``(rgb, milliseconds)`` as rollouts accumulate into the vote map."""
    counted = data["statements"][data["statements"].scorable & ~data["statements"].duplicate]
    by_rollout = {rollout: group[["seq_i", "seq_j"]].to_numpy()
                  for rollout, group in counted.groupby("rollout")}
    canvas = base_canvas(data)
    votes = np.zeros_like(data["votes"])
    # The rollout's own triangle, minus the band that is never scored: the cells the vote map is
    # allowed to paint. Everything outside it keeps whatever `base_canvas` put there.
    index = np.arange(data["length"])
    scorable = (np.tril(np.ones((data["length"],) * 2, bool), k=-1)
                & (np.abs(np.subtract.outer(index, index)) >= figlib.MIN_SEPARATION))

    frame.highlight.set_xy((-10, -10))
    for position, rollout in enumerate(order, start=1):
        pairs = by_rollout.get(rollout)
        if pairs is not None:
            votes[pairs[:, 0], pairs[:, 1]] += 1
            votes[pairs[:, 1], pairs[:, 0]] += 1
        confidence = votes / data["n_rollouts"]
        canvas[scorable] = HEAT(confidence[scorable])[:, :3]
        frame.image.set_data(canvas)
        frame.corner.set_text(f"MarinFold\n{position} rollout" + ("s" if position > 1 else ""))
        frame.status.set_text(f"rollout {position} of {len(order)}")
        frame.detail.set_text(f"R-precision {curve[position - 1]:.3f} over "
                              f"{data['n_true']} experimental contacts")
        frame.curve.set_data(np.arange(1, position + 1), curve[:position])
        frame.curve_point.set_data([position], [curve[position - 1]])
        frame.curve_label.set_text(f"{curve[position - 1]:.3f}")
        yield frame.draw(), FAST_ROLLOUT_MS


def write_gif(name: str, frames, durations) -> None:
    """Save frames as a GIF under one shared palette, plus the final frame as a PNG poster.

    Quantising each frame on its own palette makes flat areas shimmer between frames and costs
    size; one palette built from the whole sequence does neither.
    """
    figlib.OUTPUT.mkdir(parents=True, exist_ok=True)
    sample = np.concatenate(frames[:: max(1, len(frames) // 12)], axis=0)
    palette = Image.fromarray(sample).quantize(colors=PALETTE_COLORS,
                                               method=Image.Quantize.MEDIANCUT)
    images = [Image.fromarray(rgb).quantize(palette=palette, dither=Image.Dither.NONE)
              for rgb in frames]
    path = figlib.OUTPUT / f"{name}.gif"
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0,
                   optimize=True, disposal=1)
    Image.fromarray(frames[-1]).save(figlib.OUTPUT / f"{name}.png")
    seconds = sum(durations) / 1000
    print(f"wrote {path}  {len(frames)} frames · {seconds:.1f}s · "
          f"{path.stat().st_size / 2**20:.2f} MiB")


def main() -> None:
    """Build both GIFs from the stored dataset."""
    data = load_dataset()
    featured = choose_rollout(data)
    header = "Top7 de novo designed protein"

    frame = Frame(data, header, "Showing the contacts as one rollout emits them",
                  right="stream")
    frames, durations = zip(*emission_frames(frame, data, featured))
    frames, durations = list(frames), list(durations)
    durations[-1] = HOLD_MS
    write_gif("top7_rollout_emission", frames, durations)
    plt.close(frame.figure)

    # The consensus GIF replays the same rollout — at a gallop, since it is the same statements
    # in the same order — and then keeps going through the other 99 in index order.
    order = [featured] + [r for r in range(data["n_rollouts"]) if r != featured]
    curve = running_precision(data, order)
    print(f"consensus     {curve[0]:.3f} after 1 rollout -> {curve[-1]:.3f} after {len(order)}")

    frame = Frame(data, header,
                  f"Showing one rollout, then consensus across {len(order)} rollouts",
                  right="stream")
    frames, durations = zip(*emission_frames(frame, data, featured, slow_statements=0,
                                             slow_ms=SLOW_MS, fast_ms=REPLAY_MS))
    frames, durations = list(frames), list(durations)

    # A beat on the finished rollout before the other 99 arrive: without it the map jumps from
    # two colours to a continuous scale in one frame and the change reads as a glitch.
    frame.detail.set_text(f"now do that {len(order) - 1} more times, and count the votes")
    frames.append(frame.draw())
    durations.append(TRANSITION_MS)

    frame.show_curve(curve)
    for rgb, milliseconds in consensus_frames(frame, data, order, curve):
        frames.append(rgb)
        durations.append(milliseconds)
    durations[-1] = HOLD_MS
    write_gif("top7_rollout_consensus", frames, durations)
    plt.close(frame.figure)


if __name__ == "__main__":
    main()
