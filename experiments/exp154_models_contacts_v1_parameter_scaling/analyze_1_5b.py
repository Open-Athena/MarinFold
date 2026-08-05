# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit the 1.5B Gaussian process and render its partial dependence.

Run after ``fetch_wandb.py`` from this directory:

    uv run --with matplotlib --with numpy --with pandas --with scikit-learn \
        python analyze_1_5b.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
SOURCE_CSV = HERE / "data" / "wandb_runs.csv"
PLOTS_DIR = HERE / "plots"

FEATURES = (
    "epochs_grade",
    "weight_decay_grade",
    "learning_rate_grade",
    "batch_size_grade",
)
FEATURE_LABELS = {
    "epochs_grade": "Epochs",
    "weight_decay_grade": "Weight decay",
    "learning_rate_grade": "Learning rate",
    "batch_size_grade": "Batch size",
}
ROBUST_Z_THRESHOLD = 2.5
MAD_TO_STANDARD_DEVIATION = 1.4826


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save a figure as 150 dpi PNG and SVG."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    svg_path = PLOTS_DIR / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


def ordinal_map(values: pd.Series) -> dict[float, float]:
    """Map configured values to consecutive ordinal grades."""
    return {float(value): float(index) for index, value in enumerate(sorted(values.unique()))}


def identify_divergent_cells(cells: pd.DataFrame) -> pd.DataFrame:
    """Screen 1.5B eight-epoch cells for joint train/validation divergence."""
    screen = cells.loc[cells["epochs"] == 8].copy()
    if screen[["train_loss", "val_loss"]].isna().any().any():
        raise ValueError("divergence screen requires final train and validation loss for every eight-epoch cell")

    for metric in ("train_loss", "val_loss"):
        values = screen[metric].to_numpy(dtype=float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if mad <= 0:
            raise ValueError(f"cannot robustly standardize {metric}: median absolute deviation is {mad}")
        scale = MAD_TO_STANDARD_DEVIATION * mad
        screen[f"{metric}_robust_z"] = (values - median) / scale
        screen[f"{metric}_threshold"] = median + ROBUST_Z_THRESHOLD * scale

    screen["is_divergent"] = (
        (screen["train_loss_robust_z"] > ROBUST_Z_THRESHOLD)
        & (screen["val_loss_robust_z"] > ROBUST_Z_THRESHOLD)
    )
    if len(screen) != 51 or int(screen["is_divergent"].sum()) != 5:
        raise ValueError(
            f"expected 5 divergent cells among 51 eight-epoch cells, found "
            f"{int(screen['is_divergent'].sum())} among {len(screen)}"
        )
    return screen


def load_model_cells() -> tuple[pd.DataFrame, dict[str, dict[float, float]], pd.DataFrame]:
    """Load 1.5B cells and apply the plotted divergence screen."""
    runs = pd.read_csv(SOURCE_CSV)
    runs = runs.loc[runs["model_size"] == "1_5b"].copy()
    key = ["epochs", "weight_decay", "learning_rate", "batch_size"]
    cells = runs.loc[runs.groupby(key)["val_loss"].idxmin()].copy()
    divergence_screen = identify_divergent_cells(cells)
    divergent_run_ids = set(divergence_screen.loc[divergence_screen["is_divergent"], "run_id"])

    value_maps = {
        "epochs_grade": {float(value): float(np.log2(value)) for value in sorted(cells["epochs"].unique())},
        "weight_decay_grade": ordinal_map(cells["weight_decay"]),
        "learning_rate_grade": ordinal_map(cells["learning_rate"]),
        "batch_size_grade": ordinal_map(cells["batch_size"]),
    }
    source_columns = {
        "epochs_grade": "epochs",
        "weight_decay_grade": "weight_decay",
        "learning_rate_grade": "learning_rate",
        "batch_size_grade": "batch_size",
    }
    for feature, source in source_columns.items():
        cells[feature] = cells[source].map(value_maps[feature])

    model_cells = cells.loc[~cells["run_id"].isin(divergent_run_ids)].sort_values(key).reset_index(drop=True)
    if len(model_cells) != 105:
        raise ValueError(f"expected 105 non-divergent 1.5B cells, found {len(model_cells)}")
    if set(model_cells["model_size"]) != {"1_5b"}:
        raise ValueError("Gaussian-process input contains a model size other than 1.5B")
    return model_cells, value_maps, divergence_screen


def plot_divergence_screen(screen: pd.DataFrame) -> None:
    """Plot the robust dual-tail rule used to exclude divergent GP inputs."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(13.8, 6.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[2.25, 1], wspace=0.1)
    ax = fig.add_subplot(grid[0, 0])
    table_ax = fig.add_subplot(grid[0, 1])
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.14)

    converged = screen.loc[~screen["is_divergent"]]
    divergent = screen.loc[screen["is_divergent"]].sort_values("train_loss_robust_z")
    all_z = screen[["train_loss_robust_z", "val_loss_robust_z"]].to_numpy()
    lower = min(-1.0, float(all_z.min()) - 0.4)
    upper = max(ROBUST_Z_THRESHOLD + 0.5, float(all_z.max()) + 0.6)

    ax.fill_between(
        [ROBUST_Z_THRESHOLD, upper],
        ROBUST_Z_THRESHOLD,
        upper,
        color="#FEE2E2",
        alpha=0.6,
        zorder=0,
    )
    ax.plot([lower, upper], [lower, upper], color="#94A3B8", linewidth=1, linestyle=":", zorder=1)
    ax.axvline(ROBUST_Z_THRESHOLD, color="#DC2626", linewidth=1.1, linestyle="--")
    ax.axhline(ROBUST_Z_THRESHOLD, color="#DC2626", linewidth=1.1, linestyle="--")
    ax.scatter(
        converged["train_loss_robust_z"],
        converged["val_loss_robust_z"],
        s=42,
        color="#2563EB",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.78,
        label="Retained",
        zorder=2,
    )
    ax.scatter(
        divergent["train_loss_robust_z"],
        divergent["val_loss_robust_z"],
        s=72,
        marker="X",
        color="#DC2626",
        edgecolor="white",
        linewidth=0.7,
        label="Excluded from GP",
        zorder=3,
    )
    for number, (_, row) in enumerate(divergent.iterrows(), start=1):
        ax.annotate(
            str(number),
            (row["train_loss_robust_z"], row["val_loss_robust_z"]),
            xytext=(7, 6),
            textcoords="offset points",
            color="#991B1B",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Final training loss (robust z-score)")
    ax.set_ylabel("Validation loss (robust z-score)")
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    table_ax.axis("off")
    table_ax.set_title("Excluded configurations", loc="left", fontsize=14, pad=12)
    table_rows = []
    for number, (_, row) in enumerate(divergent.iterrows(), start=1):
        table_rows.append(
            [
                str(number),
                f"{row['learning_rate']:.2g}",
                f"{row['weight_decay']:g}",
                str(int(row["batch_size"])),
                f"{row['val_loss']:.3f}",
            ]
        )
    table = table_ax.table(
        cellText=table_rows,
        colLabels=["#", "LR", "WD", "BS", "Val loss"],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.1, 0.24, 0.18, 0.18, 0.3],
        bbox=[0, 0.26, 1, 0.62],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#FEE2E2")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#FFF7F7" if row % 2 else "white")
    table_ax.text(
        0,
        0.17,
        "A cell is excluded only when both robust z-scores exceed 2.5.",
        transform=table_ax.transAxes,
        color="#475569",
        fontsize=9.5,
        va="top",
        wrap=True,
    )

    fig.suptitle("Divergence screen for 1.5B eight-epoch cells", x=0.04, ha="left", fontsize=20)
    fig.text(
        0.04,
        0.87,
        "Joint upper-tail screen using median/MAD standardization of final W&B training loss and validation loss.",
        color="#475569",
        fontsize=10.5,
    )
    save_figure(fig, "divergence_screen")


def make_gpr(random_state: int) -> GaussianProcessRegressor:
    """Construct the ordinal-input Matérn Gaussian process."""
    signal = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=np.ones(len(FEATURES)),
        length_scale_bounds=(0.15, 30.0),
        nu=2.5,
    )
    kernel = signal + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 0.5))
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=4,
        random_state=random_state,
    )


def fit_gpr(x: np.ndarray, y: np.ndarray, random_state: int) -> tuple[StandardScaler, GaussianProcessRegressor]:
    """Standardize ordinal coordinates and fit a Gaussian process."""
    scaler = StandardScaler().fit(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = make_gpr(random_state).fit(scaler.transform(x), y)
    return scaler, model


def format_level(feature: str, raw_value: float) -> str:
    """Format a configured value for a partial-dependence axis."""
    if feature == "learning_rate_grade":
        return f"{raw_value:.2g}"
    if feature in {"epochs_grade", "batch_size_grade"}:
        return str(int(raw_value))
    return f"{raw_value:g}"


def integrated_partial_dependence(
    x: np.ndarray,
    feature_index: int,
    levels: np.ndarray,
    scaler: StandardScaler,
    model: GaussianProcessRegressor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact centered GP partial dependence and posterior covariance.

    The nuisance-design distribution is the empirical distribution of the 105
    unique non-divergent cells. Because integration and centering are linear,
    the resulting posterior follows directly from the GP joint posterior.
    """
    evaluations = []
    for level in levels:
        block = x.copy()
        block[:, feature_index] = level
        evaluations.append(block)
    evaluation_matrix = np.vstack(evaluations)
    mean, covariance = model.predict(scaler.transform(evaluation_matrix), return_cov=True)

    # WhiteKernel contributes observation noise only on the test diagonal. PDP
    # describes the latent response surface, so remove that term before applying
    # the integration operator.
    noise = float(model.kernel_.k2.noise_level) * float(model._y_train_std**2)
    covariance = covariance.copy()
    covariance.flat[:: len(covariance) + 1] -= noise

    n_rows = len(x)
    n_levels = len(levels)
    averaging = np.zeros((n_levels, n_rows * n_levels))
    for index in range(n_levels):
        averaging[index, index * n_rows:(index + 1) * n_rows] = 1.0 / n_rows
    pd_mean = averaging @ mean
    pd_covariance = averaging @ covariance @ averaging.T

    frequencies = np.asarray([(x[:, feature_index] == level).mean() for level in levels])
    centering = np.eye(n_levels) - np.ones((n_levels, 1)) @ frequencies[None, :]
    return centering @ pd_mean, centering @ pd_covariance @ centering.T


def plot_gpr_partial_dependence(
    x: np.ndarray,
    value_maps: dict[str, dict[float, float]],
    scaler: StandardScaler,
    model: GaussianProcessRegressor,
) -> None:
    """Plot exact GP posterior partial dependence for every ordinal input."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.2), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.12, hspace=0.42, wspace=0.16)

    for feature_index, (feature, ax) in enumerate(zip(FEATURES, axes.flat, strict=True)):
        raw_to_grade = value_maps[feature]
        raw_levels = np.asarray(sorted(raw_to_grade))
        grades = np.asarray([raw_to_grade[value] for value in raw_levels])
        effect, covariance = integrated_partial_dependence(x, feature_index, grades, scaler, model)
        standard_error = np.sqrt(np.maximum(np.diag(covariance), 0))
        lower = effect - 1.96 * standard_error
        upper = effect + 1.96 * standard_error
        positions = np.arange(len(grades))

        ax.fill_between(positions, lower, upper, color="#93C5FD", alpha=0.35, linewidth=0)
        ax.plot(positions, effect, color="#2563EB", marker="o", markersize=4, linewidth=1.6)
        ax.axhline(0, color="#64748B", linewidth=0.8)
        labels = [format_level(feature, value) for value in raw_levels]
        ax.set_xticks(positions, labels)
        if len(labels) > 8:
            ax.tick_params(axis="x", rotation=45)
        ax.set_title(FEATURE_LABELS[feature], fontsize=14)
        ax.set_xlabel("Configured value (ordinal spacing)")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0, 0].set_ylabel("Centered effect on validation loss")
    axes[1, 0].set_ylabel("Centered effect on validation loss")
    fig.suptitle("Gaussian-process partial dependence", x=0.05, ha="left", fontsize=20)
    fig.text(
        0.05,
        0.89,
        "Exact integration of the joint GP posterior over the observed nuisance-design distribution; "
        "bands are conditional 95% credible intervals.",
        color="#475569",
        fontsize=10.5,
    )
    save_figure(fig, "gpr_partial_dependence")


def main() -> int:
    """Screen divergence, fit the 1.5B GP, and write both figures."""
    cells, value_maps, divergence_screen = load_model_cells()
    x = cells.loc[:, FEATURES].to_numpy(dtype=float)
    y = cells["val_loss"].to_numpy(dtype=float)
    plot_divergence_screen(divergence_screen)
    scaler, gpr = fit_gpr(x, y, 154)
    plot_gpr_partial_dependence(x, value_maps, scaler, gpr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
