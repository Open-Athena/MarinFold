# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Finalize the two full exp166 checkpoint evals from their durable HF parts.

This is the compact exp166 counterpart to PR #170's fetch, metric, summary,
paired-statistics, and plotting scripts. It deliberately does not import that
experiment. Sparse integer vote parts are lossless: the script reconstructs
each dense score matrix exactly, scores it with exp89's metric semantics, and
stores all matrices in one tar archive per checkpoint so later analyses do not
need another TPU run.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import shutil
import tarfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from sklearn.metrics import roc_auc_score

from checkpoint_specs import CHECKPOINTS, HF_BUCKET_ROOT

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
DATA = HERE / "data"
PLOTS = HERE / "plots"
SCRATCH = REPO_ROOT / "scratch" / "exp166-analysis"
HISTORICAL_EXP146 = DATA / "historical_exp146_rprecision.csv.gz"
HISTORICAL_EXP146_SHA256 = (
    "daf8b31cddae3898bee11b6c7214daa83e29630626718f886654689c97b6e1f0"
)

RANGES = {"all": (6, None), "short": (6, 11), "medium": (12, 23), "long": (24, None)}
CUTS = (
    ("L", lambda length, n_true: length),
    ("L/2", lambda length, n_true: max(1, length // 2)),
    ("L/5", lambda length, n_true: max(1, length // 5)),
    ("R", lambda length, n_true: n_true),
)
STRATA = ("neff_tier", "fold_verdict", "seq_leakage", "msa_neff", "length")
CONTROL_R_PRECISION = 0.5344
CONTROL_TOLERANCE = 0.006

MODELS = {
    "exp117-control": "exp117_control_step35679",
    "exp166": "exp166_aaaug_step35679",
}

LOSS_POINTS = (
    {
        "model": "marinfold-cv1-exp75-rollout",
        "label": "#75",
        "val_loss": 2.756602,
        "color": "#9b938c",
        "origin": "historical evaluation",
        "generated_in_this_evaluation": False,
        "r_precision_source": (
            "experiments/exp82_evals_contacts_v1_contact_prediction/"
            "data/where_we_stand_rows.csv.gz"
        ),
        "val_loss_source": "PR #170 / issue #169 training-run record",
        "marker": "o",
        "fit_in_1_5b_trend": True,
    },
    {
        "model": MODELS["exp117-control"],
        "label": "#117",
        "val_loss": 2.703709,
        "color": "#eb6834",
        "origin": "this evaluation",
        "generated_in_this_evaluation": True,
        "r_precision_source": "data/exp166_rows.csv.gz",
        "val_loss_source": "exp117 training-run record",
        "marker": "o",
        "fit_in_1_5b_trend": True,
    },
    {
        "model": "exp146_3b_e8_step17839",
        "label": "#146 · 3B",
        "val_loss": 2.702478,
        "color": "#9b938c",
        "origin": "historical evaluation",
        "generated_in_this_evaluation": False,
        "r_precision_source": "data/historical_exp146_rprecision.csv.gz",
        "val_loss_source": "PR #170 / issue #169 training-run record",
        "marker": "s",
        "fit_in_1_5b_trend": False,
    },
    {
        "model": MODELS["exp166"],
        "label": "#166",
        "val_loss": 2.664179,
        "color": "#eb6834",
        "origin": "this evaluation",
        "generated_in_this_evaluation": True,
        "r_precision_source": "data/exp166_rows.csv.gz",
        "val_loss_source": "open-athena/marinfold-exp166 checkpoint inventory",
        "marker": "o",
        "fit_in_1_5b_trend": True,
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(fs: HfFileSystem, path: str) -> Any:
    with fs.open(path, "r") as source:
        return json.load(source)


def read_parquet_parts(fs: HfFileSystem, pattern: str) -> pd.DataFrame:
    paths = sorted(fs.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no parquet parts match {pattern}")
    frames = []
    for path in paths:  # Intentionally sequential: this VM is resource-limited.
        with fs.open(path, "rb") as source:
            frames.append(pq.read_table(source).to_pandas())
    return pd.concat(frames, ignore_index=True)


def load_ground_truth(fs: HfFileSystem, prefix: str) -> list[dict[str, Any]]:
    path = f"{prefix}/inputs/gt_universe.jsonl"
    with fs.open(path, "r") as source:
        records = [json.loads(line) for line in source]
    keys = [(record["dataset"], record["stem"]) for record in records]
    if len(records) != 554 or len(set(keys)) != 554:
        raise ValueError(
            f"expected 554 unique ground-truth units, got {len(set(keys))}"
        )
    return records


def true_matrix(length: int, contacts: Iterable[Sequence[float]]) -> np.ndarray:
    matrix = np.zeros((length, length), dtype=bool)
    for raw_i, raw_j, degree in contacts:
        i, j = int(raw_i), int(raw_j)
        if float(degree) >= 0.001 and j - i >= 6 and i < j < length:
            matrix[i, j] = True
    return matrix


def metric_rows(score: np.ndarray, record: dict[str, Any]) -> list[dict[str, Any]]:
    """Exact exp89/PR #170 precision and AUC semantics for one protein."""

    length = int(record["L"])
    resolved = np.asarray(record["resolved"], dtype=np.int64)
    a, b = np.triu_indices(len(resolved), k=1)
    pair_i, pair_j = resolved[a], resolved[b]
    separation = pair_j - pair_i
    scores = score[pair_i, pair_j]
    truth = true_matrix(length, record["contacts"])[pair_i, pair_j].astype(int)

    rows = []
    for range_name, (low, high) in RANGES.items():
        selected = separation >= low
        if high is not None:
            selected &= separation <= high
        range_scores, range_truth = scores[selected], truth[selected]
        n_candidate, n_true = int(range_scores.size), int(range_truth.sum())
        order = np.argsort(-range_scores, kind="mergesort") if n_candidate else None
        ranked_truth = range_truth[order] if n_candidate else None
        for cut, cut_size in CUTS:
            target = int(cut_size(length, n_true))
            if n_candidate == 0 or target <= 0:
                precision, n_top = math.nan, 0
            else:
                n_top = min(target, n_candidate)
                precision = float(ranked_truth[:n_top].sum()) / n_top
            rows.append(
                {
                    "range": range_name,
                    "cut": cut,
                    "precision": precision,
                    "n_candidate": n_candidate,
                    "n_true": n_true,
                    "n_top": n_top,
                }
            )
        auc = (
            float(roc_auc_score(range_truth, range_scores))
            if n_candidate and 0 < n_true < n_candidate
            else math.nan
        )
        rows.append(
            {
                "range": range_name,
                "cut": "AUC",
                "precision": auc,
                "n_candidate": n_candidate,
                "n_true": n_true,
                "n_top": n_candidate,
            }
        )
    return rows


def dense_scores(
    votes: pd.DataFrame, timing_keys: set[tuple[str, str]]
) -> dict[tuple[str, str], list[tuple[int, int, int]]]:
    triplets: dict[tuple[str, str], list[tuple[int, int, int]]] = defaultdict(list)
    seen: set[tuple[str, str, int, int]] = set()
    for row in votes.itertuples(index=False):
        key = (str(row.dataset), str(row.stem))
        pair = (*key, int(row.i), int(row.j))
        if pair in seen:
            raise ValueError(f"duplicate sparse vote pair: {pair}")
        if key not in timing_keys:
            raise ValueError(f"vote row has no completion marker: {key}")
        seen.add(pair)
        triplets[key].append((int(row.i), int(row.j), int(row.votes)))
    return triplets


def add_matrix(archive: tarfile.TarFile, name: str, matrix: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, score=matrix.astype(np.float16))
    payload = buffer.getvalue()
    info = tarfile.TarInfo(f"{name}.npz")
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def score_model(
    fs: HfFileSystem,
    model_key: str,
    ground_truth: Sequence[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, Path, dict[str, Any]]:
    spec = CHECKPOINTS[model_key]
    label = MODELS[model_key]
    prefix = f"{HF_BUCKET_ROOT}/scores/{spec.output_name}"
    manifest = read_json(fs, f"{prefix}/manifest.json")
    if manifest["n_targets"] != 554 or manifest["n_rollouts"] != 100:
        raise ValueError(f"unexpected run manifest for {model_key}: {manifest}")

    timings = read_parquet_parts(fs, f"{prefix}/parts/timings-*.parquet")
    complete = timings[timings["complete"]]
    timing_keys = set(zip(complete["dataset"], complete["stem"], strict=True))
    if len(timings) != 554 or len(timing_keys) != 554:
        raise ValueError(
            f"{model_key}: expected 554 unique complete timings, got "
            f"{len(timings)} rows / {len(timing_keys)} keys"
        )
    votes = read_parquet_parts(fs, f"{prefix}/parts/votes-*.parquet")
    triplets = dense_scores(votes, timing_keys)

    SCRATCH.mkdir(parents=True, exist_ok=True)
    archive_path = SCRATCH / f"score-matrices-{label}.tar"
    rows: list[dict[str, Any]] = []
    with tarfile.open(archive_path, "w") as archive:
        for record in ground_truth:
            key = (record["dataset"], record["stem"])
            if key not in timing_keys:
                raise ValueError(f"{model_key}: missing completed unit {key}")
            length = int(record["L"])
            matrix = np.zeros((length, length), dtype=np.float64)
            for i, j, value in triplets.get(key, []):
                matrix[i, j] = matrix[j, i] = value
            add_matrix(archive, f"{key[0]}__{key[1]}", matrix)
            strata = record.get("strata", {}) or {}
            base = dict(
                dataset=key[0],
                stem=key[1],
                n_residues=length,
                model=label,
                mode="single_seq",
                predictor="lm",
                **{name: strata.get(name) for name in STRATA},
            )
            rows.extend({**base, **metric} for metric in metric_rows(matrix, record))

    timings = timings.assign(model=label)
    run_stats = {
        "model": label,
        "output_prefix": prefix,
        "manifest": manifest,
        "n_timing_rows": len(timings),
        "n_vote_rows": len(votes),
        "n_rollouts": int(timings["n_rollouts"].sum()),
        "n_stopped_rollouts": int(timings["stopped_rollouts"].sum()),
        "matrix_archive": archive_path.name,
        "matrix_archive_bytes": archive_path.stat().st_size,
        "matrix_archive_sha256": sha256(archive_path),
    }
    return pd.DataFrame(rows), timings, archive_path, run_stats


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["model", "range", "cut"])["precision"]
    return grouped.agg(
        mean_precision="mean", n_valid="count", n_units="size"
    ).reset_index()


def paired(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for range_name in RANGES:
        for cut in ("R", "L", "L/2", "L/5", "AUC"):
            selected = rows[(rows["range"] == range_name) & (rows["cut"] == cut)]
            wide = selected.pivot(
                index=["dataset", "stem"], columns="model", values="precision"
            )
            pair = wide[[MODELS["exp166"], MODELS["exp117-control"]]].dropna()
            delta = (pair[MODELS["exp166"]] - pair[MODELS["exp117-control"]]).to_numpy()
            sem = float(delta.std(ddof=1) / np.sqrt(delta.size))
            records.append(
                {
                    "range": range_name,
                    "cut": cut,
                    "n_paired": delta.size,
                    "mean_candidate": float(pair[MODELS["exp166"]].mean()),
                    "mean_control": float(pair[MODELS["exp117-control"]].mean()),
                    "mean_delta": float(delta.mean()),
                    "sem": sem,
                    "ci_low": float(delta.mean() - 1.96 * sem),
                    "ci_high": float(delta.mean() + 1.96 * sem),
                    "candidate_win_rate": float((delta > 0).mean()),
                    "tie_rate": float((delta == 0).mean()),
                }
            )
    return pd.DataFrame(records)


def values(
    rows: pd.DataFrame, model: str, mode: str, predictor: str, range_name: str
) -> np.ndarray:
    selected = rows[
        (rows["model"] == model)
        & (rows["mode"] == mode)
        & (rows["predictor"] == predictor)
        & (rows["range"] == range_name)
        & (rows["cut"] == "R")
    ]["precision"].to_numpy(dtype=float)
    return selected[np.isfinite(selected)]


def draw_boxplot(
    axis: plt.Axes,
    series: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    title: str,
    *,
    width: float = 0.62,
) -> None:
    boxes = axis.boxplot(
        series,
        widths=width,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "#111",
            "markersize": 5,
        },
        medianprops={"color": "#111", "linewidth": 1.4},
        flierprops={"marker": ".", "markersize": 2.5, "alpha": 0.3},
    )
    for box, color in zip(boxes["boxes"], colors, strict=True):
        box.set_facecolor(color)
        box.set_alpha(0.85)
    for x, data in enumerate(series, start=1):
        axis.text(x, 1.035, f"{data.mean():.3f}", ha="center", va="bottom", fontsize=9)
    axis.set_xticks(range(1, len(labels) + 1), labels, fontsize=8.5)
    axis.set_ylabel("R-precision (top-R ranked pairs)")
    axis.set_ylim(-0.02, 1.10)
    axis.grid(axis="y", color="#d8d7d2", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title(title, pad=14)


def save_boxplot(
    series: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    out: Path,
    title: str,
    *,
    provenance: list[dict[str, Any]] | None = None,
    source_note: str | None = None,
) -> None:
    if provenance is not None and len(provenance) != len(series):
        raise ValueError("provenance must have one record per series")
    fig, axis = plt.subplots(figsize=(10.8, 5.8 if source_note else 5.5))
    draw_boxplot(axis, series, labels, colors, title)
    if source_note:
        fig.text(0.5, 0.045, source_note, ha="center", fontsize=8.5, weight="bold")
    fig.text(
        0.5,
        0.01,
        "box = median and IQR · whiskers = 1.5×IQR · ◆ = mean",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.08 if source_note else 0.04, 1, 1))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    metadata = {
        "metric": "R-precision",
        "labels": labels,
        "counts": [len(data) for data in series],
        "means": [float(data.mean()) for data in series],
    }
    if provenance is not None:
        metadata["provenance"] = provenance
    if source_note:
        metadata["source_note"] = source_note
    out.with_suffix(out.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2)
    )


def save_loss_scatter(
    points: list[dict[str, Any]],
    out: Path | None,
    *,
    baseline: dict[str, Any],
    axis: plt.Axes | None = None,
    title: str = "Validation loss versus contact R-precision",
) -> dict[str, float | str]:
    """Plot the three 1.5B checkpoints without PR #170's paired-delta panel."""

    if len(points) < 2:
        raise ValueError("loss scatter requires at least two points")
    if axis is None and out is None:
        raise ValueError("loss scatter requires an output path or an axis")
    xs = [float(point["val_loss"]) for point in points]
    ys = [float(point["r_precision"]) for point in points]
    errors = [float(point["ci95"]) for point in points]
    baseline_r = float(baseline["r_precision"])
    fit_points = [point for point in points if point.get("fit_in_1_5b_trend", True)]
    if len(fit_points) < 2:
        raise ValueError("loss scatter requires at least two fitted 1.5B points")
    fit_xs = [float(point["val_loss"]) for point in fit_points]
    fit_ys = [float(point["r_precision"]) for point in fit_points]
    slope, intercept = np.polyfit(fit_xs, fit_ys, 1)
    crossover_loss = float((baseline_r - intercept) / slope)

    standalone = axis is None
    if axis is None:
        fig, axis = plt.subplots(figsize=(7.8, 5.3))
    else:
        fig = axis.figure
    fit_grid = np.linspace(max(fit_xs), crossover_loss, 100)
    axis.plot(
        fit_grid,
        slope * fit_grid + intercept,
        color="#52514e",
        linewidth=1.8,
        zorder=1,
    )
    axis.axhline(
        baseline_r,
        color="#2a78d6",
        linestyle="--",
        linewidth=1.5,
        zorder=1,
    )
    axis.axvline(
        crossover_loss,
        color="#2a78d6",
        linestyle=":",
        linewidth=1.0,
        alpha=0.8,
        zorder=1,
    )
    axis.scatter(
        crossover_loss,
        baseline_r,
        marker="X",
        s=90,
        color="#2a78d6",
        edgecolor="white",
        linewidth=1.4,
        zorder=4,
    )
    axis.errorbar(
        xs,
        ys,
        yerr=errors,
        fmt="none",
        ecolor="#8a8880",
        elinewidth=1.2,
        capsize=4,
        zorder=2,
    )
    label_offsets = (
        (0, 54, "center"),
        (-52, 36, "right"),
        (10, -66, "left"),
        (-18, 58, "center"),
    )
    for index, (point, x, y) in enumerate(zip(points, xs, ys, strict=True)):
        dx, dy, alignment = label_offsets[index % len(label_offsets)]
        axis.scatter(
            x,
            y,
            s=125,
            marker=point.get("marker", "o"),
            color=point["color"],
            edgecolor="white",
            linewidth=2,
            zorder=3,
        )
        axis.annotate(
            f"{point['label']}\nloss {x:.4f} · R {y:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=alignment,
            fontsize=8.5,
            arrowprops={
                "arrowstyle": "-",
                "color": "#b7b5ae",
                "linewidth": 1,
                "shrinkA": 4,
                "shrinkB": 13,
            },
        )

    axis.text(
        0.02,
        baseline_r + 0.006,
        f"Protenix-v2 single-seq · R = {baseline_r:.3f}",
        transform=axis.get_yaxis_transform(),
        color="#2a78d6",
        fontsize=9,
        va="bottom",
    )
    axis.text(
        0.02,
        0.045,
        (
            f"1.5B best fit: R = {slope:.3f} × loss + {intercept:.3f}\n"
            f"Protenix-v2 crossover: expected loss ≈ {crossover_loss:.3f}"
        ),
        transform=axis.transAxes,
        ha="left",
        fontsize=8.5,
        linespacing=1.5,
        color="#52514e",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 3},
    )
    axis.set_xlim(max(xs) + 0.012, min(min(xs), crossover_loss) - 0.012)
    axis.set_xlabel("contacts-v1 validation loss (lower is better →)")
    axis.set_ylabel("Mean all-range R-precision (n = 554)")
    axis.set_title(title, pad=14)
    axis.grid(color="#d8d7d2", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    low = min(y - error for y, error in zip(ys, errors, strict=True))
    axis.set_ylim(low - 0.035, baseline_r + 0.07)
    fit = {
        "equation": "R-precision = slope * validation_loss + intercept",
        "slope": float(slope),
        "intercept": float(intercept),
        "baseline_crossover_loss": crossover_loss,
    }
    if not standalone:
        return fit
    assert out is not None
    fig.text(
        0.5,
        0.025,
        "Contact scores generated here: #117 and #166 · Historical contact score: #75",
        ha="center",
        fontsize=8.5,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    out.with_suffix(out.suffix + ".meta.json").write_text(
        json.dumps(
            {
                "figure": "loss_vs_rprecision",
                "metric": "R-precision, range=all",
                "points": points,
                "fit": fit,
                "baseline": baseline,
                "source_note": (
                    "Contact scores generated here: #117 and #166; "
                    "historical contact score: #75"
                ),
            },
            indent=2,
        )
    )
    return fit


def save_combined_plot(
    box_series: list[np.ndarray],
    box_labels: list[str],
    box_colors: list[str],
    box_provenance: list[dict[str, Any]],
    loss_points: list[dict[str, Any]],
    baseline: dict[str, Any],
    out: Path,
) -> None:
    if len(box_provenance) != len(box_series):
        raise ValueError("provenance must have one record per box series")
    fig, (box_axis, loss_axis) = plt.subplots(
        1,
        2,
        figsize=(16.2, 6.5),
        gridspec_kw={"width_ratios": [1.0, 1.18]},
    )
    draw_boxplot(
        box_axis,
        box_series,
        box_labels,
        box_colors,
        "A · Contact R-precision distributions",
        width=0.28,
    )
    fit = save_loss_scatter(
        loss_points,
        None,
        baseline=baseline,
        axis=loss_axis,
        title="B · Validation loss versus mean R-precision",
    )
    fig.suptitle("Where MarinFold stands on contacts-v1", fontsize=14, y=0.985)
    source_note = (
        "Generated here: #117 and #166 · Historical evaluations: #75, #146, "
        "and Protenix-v2 single-sequence"
    )
    fig.text(0.5, 0.042, source_note, ha="center", fontsize=8.5, weight="bold")
    fig.text(
        0.5,
        0.012,
        (
            "A: box = median and IQR · whiskers = 1.5×IQR · ◆ = mean.   "
            "B: error bars = 95% CI · line = least-squares fit through the "
            "three 1.5B checkpoints."
        ),
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95), w_pad=2.5)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    out.with_suffix(out.suffix + ".meta.json").write_text(
        json.dumps(
            {
                "figure": "where_we_stand_rprecision",
                "metric": "R-precision, range=all",
                "boxplot": {
                    "labels": box_labels,
                    "counts": [len(data) for data in box_series],
                    "means": [float(data.mean()) for data in box_series],
                    "provenance": box_provenance,
                },
                "scatter": {
                    "points": loss_points,
                    "fit": fit,
                    "baseline": baseline,
                },
                "source_note": source_note,
            },
            indent=2,
        )
    )


def make_plots(rows: pd.DataFrame) -> list[Path]:
    exp89 = pd.read_csv(
        REPO_ROOT
        / "experiments/exp89_evals_contacts_v1_model_on_eval_set/data/contact_precision_all.csv"
    )
    prior = pd.read_csv(
        REPO_ROOT
        / "experiments/exp82_evals_contacts_v1_contact_prediction/data/where_we_stand_rows.csv.gz"
    )
    if sha256(HISTORICAL_EXP146) != HISTORICAL_EXP146_SHA256:
        raise ValueError("historical exp146 rows checksum mismatch")
    exp146 = pd.read_csv(HISTORICAL_EXP146)
    combined = pd.concat([rows, exp89, prior, exp146], ignore_index=True)
    panel = [
        (
            "marinfold-cv1-exp75-rollout",
            "single_seq",
            "lm",
            "MarinFold #75\n1.5B · E8",
            "#9b938c",
            {
                "origin": "historical evaluation",
                "generated_in_this_evaluation": False,
                "source": (
                    "experiments/exp82_evals_contacts_v1_contact_prediction/"
                    "data/where_we_stand_rows.csv.gz"
                ),
            },
        ),
        (
            MODELS["exp117-control"],
            "single_seq",
            "lm",
            "Control #117\n1.5B · E16",
            "#eb6834",
            {
                "origin": "this evaluation",
                "generated_in_this_evaluation": True,
                "source": "data/exp166_rows.csv.gz",
            },
        ),
        (
            "exp146_3b_e8_step17839",
            "single_seq",
            "lm",
            "MarinFold #146\n3B · E8",
            "#9b938c",
            {
                "origin": "historical evaluation",
                "generated_in_this_evaluation": False,
                "source": "data/historical_exp146_rprecision.csv.gz",
                "upstream_experiment": "exp169 / PR #170",
            },
        ),
        (
            MODELS["exp166"],
            "single_seq",
            "lm",
            "Candidate #166\nAA augmentation",
            "#eb6834",
            {
                "origin": "this evaluation",
                "generated_in_this_evaluation": True,
                "source": "data/exp166_rows.csv.gz",
            },
        ),
        (
            "protenix-v2",
            "single_seq",
            "structure",
            "Protenix-v2\nsingle-seq",
            "#2a78d6",
            {
                "origin": "historical evaluation",
                "generated_in_this_evaluation": False,
                "source": (
                    "experiments/exp89_evals_contacts_v1_model_on_eval_set/"
                    "data/contact_precision_all.csv"
                ),
                "upstream_experiment": "exp74",
            },
        ),
    ]
    data, labels, colors, provenance = [], [], [], []
    for model, mode, predictor, label, color, series_provenance in panel:
        selected = values(combined, model, mode, predictor, "all")
        if selected.size:
            data.append(selected)
            labels.append(label)
            colors.append(color)
            provenance.append({"label": label, **series_provenance})
    order = sorted(range(len(data)), key=lambda index: float(data[index].mean()))
    data = [data[index] for index in order]
    labels = [labels[index] for index in order]
    colors = [colors[index] for index in order]
    provenance = [provenance[index] for index in order]
    where = PLOTS / "where_we_stand_rprecision.png"

    candidate_control = PLOTS / "candidate_vs_control_rprecision.png"
    paired_series, paired_labels = [], []
    range_labels = (
        ("all", "All"),
        ("short", "Short"),
        ("medium", "Medium"),
        ("long", "Long"),
    )
    for range_name, label in range_labels:
        paired_series.extend(
            [
                values(rows, MODELS["exp117-control"], "single_seq", "lm", range_name),
                values(rows, MODELS["exp166"], "single_seq", "lm", range_name),
            ]
        )
        paired_labels.extend([f"Control\n{label}", f"Candidate\n{label}"])
    save_boxplot(
        paired_series,
        paired_labels,
        ["#f7bda4", "#eb6834"] * len(range_labels),
        candidate_control,
        "Exp166 candidate versus exp117 initialization by contact range",
    )

    loss_points = []
    for spec in LOSS_POINTS:
        selected = values(combined, spec["model"], "single_seq", "lm", range_name="all")
        if selected.size != 554:
            raise ValueError(
                f"expected 554 all-range R rows for {spec['model']}, "
                f"got {selected.size}"
            )
        ci95 = 1.96 * float(selected.std(ddof=1)) / np.sqrt(selected.size)
        loss_points.append(
            {
                **spec,
                "r_precision": float(selected.mean()),
                "ci95": ci95,
                "n_proteins": int(selected.size),
            }
        )
    protenix = values(
        combined, "protenix-v2", "single_seq", "structure", range_name="all"
    )
    if protenix.size != 554:
        raise ValueError(
            f"expected 554 all-range R rows for Protenix-v2, got {protenix.size}"
        )
    baseline = {
        "label": "Protenix-v2 single-sequence",
        "r_precision": float(protenix.mean()),
        "n_proteins": int(protenix.size),
        "origin": "historical evaluation",
        "source": (
            "experiments/exp89_evals_contacts_v1_model_on_eval_set/"
            "data/contact_precision_all.csv"
        ),
        "upstream_experiment": "exp74",
    }
    save_combined_plot(
        data,
        labels,
        colors,
        provenance,
        loss_points,
        baseline,
        where,
    )

    return [
        where,
        where.with_suffix(where.suffix + ".meta.json"),
        candidate_control,
        candidate_control.with_suffix(candidate_control.suffix + ".meta.json"),
    ]


def upload(fs: HfFileSystem, source: Path, relative: str) -> None:
    destination = f"{HF_BUCKET_ROOT}/derived/{relative}"
    with source.open("rb") as local, fs.open(destination, "wb") as remote:
        shutil.copyfileobj(local, remote, length=1024 * 1024)
    print(f"[upload] {source} -> {destination}")


def run(*, should_upload: bool) -> None:
    token = None
    if should_upload:
        import os

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN is required with --upload")
    fs = HfFileSystem(token=token)
    candidate_prefix = f"{HF_BUCKET_ROOT}/scores/{CHECKPOINTS['exp166'].output_name}"
    ground_truth = load_ground_truth(fs, candidate_prefix)

    model_rows, timing_frames, archives, run_stats = [], [], [], []
    for key in ("exp117-control", "exp166"):
        rows, timings, archive, stats = score_model(fs, key, ground_truth)
        model_rows.append(rows)
        timing_frames.append(timings)
        archives.append(archive)
        run_stats.append(stats)

    rows = pd.concat(model_rows, ignore_index=True)
    timings = pd.concat(timing_frames, ignore_index=True)
    summary = aggregate(rows)
    paired_rows = paired(rows)

    DATA.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    outputs = {
        "exp166_rows.csv.gz": rows,
        "exp166_summary.csv": summary,
        "exp166_paired.csv": paired_rows,
        "exp166_timings.csv.gz": timings,
    }
    for name, frame in outputs.items():
        frame.to_csv(DATA / name, index=False)

    plots = make_plots(rows)
    control = summary[
        (summary["model"] == MODELS["exp117-control"])
        & (summary["range"] == "all")
        & (summary["cut"] == "R")
    ]["mean_precision"].item()
    if abs(control - CONTROL_R_PRECISION) > CONTROL_TOLERANCE:
        raise RuntimeError(
            f"control harness check failed: {control:.6f} vs {CONTROL_R_PRECISION:.4f} "
            f"(tolerance {CONTROL_TOLERANCE:.3f})"
        )

    artifact_paths = (
        [DATA / name for name in outputs] + [HISTORICAL_EXP146] + plots + archives
    )
    derived_manifest = {
        "control_reference_r_precision": CONTROL_R_PRECISION,
        "control_observed_r_precision": control,
        "control_tolerance": CONTROL_TOLERANCE,
        "control_check_passed": True,
        "runs": run_stats,
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in artifact_paths
        },
    }
    manifest_path = DATA / "derived_manifest.json"
    manifest_path.write_text(json.dumps(derived_manifest, indent=2, sort_keys=True))
    artifact_paths.append(manifest_path)

    headline = summary[
        summary["range"].isin(["all", "long"]) & summary["cut"].isin(["R", "L", "AUC"])
    ].pivot(index="model", columns=["range", "cut"], values="mean_precision")
    print("\n", headline.round(4).to_string())
    print(
        "\n",
        paired_rows[
            (paired_rows["range"] == "all") & (paired_rows["cut"] == "R")
        ].to_string(index=False),
    )

    if should_upload:
        for path in artifact_paths:  # Intentionally one upload at a time.
            if path in archives:
                relative = f"score-matrices/{path.name}"
            elif path.parent == PLOTS:
                relative = f"plots/{path.name}"
            else:
                relative = path.name
            upload(fs, path, relative)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    run(should_upload=args.upload)


if __name__ == "__main__":
    main()
